# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
import torch.nn.functional as F

from vllm import _custom_ops as ops
from vllm.model_executor.custom_op import CustomOp
from vllm.model_executor.layers.quantization.utils.quant_utils import GroupShape
from vllm.platforms import current_platform
import hifloat8_quant as hif8_cast

_HIF8_MAX = 2.0**15*1.25
_HIF8_MIN = -2.0**15*1.25
_HIF8_MIN_SCALING_FACTOR = 1.0 / (_HIF8_MAX * 512.0)


@CustomOp.register("quant_hif8_fake")
class QuantFakeHiF8(CustomOp):
    """
    Fake quantize input tensor to HiF8 (per-tensor, per-token, or per-group).
    This CustomOp supports both static and dynamic quantization.
    """

    # Scale target for KV cache dynamic scaling: maps x_max to this value,
    # placing the majority of values in the HiF8 3-bit sweet spot [0.125, 16).
    _KV_SCALE_TARGET = 16.0

    def __init__(
        self,
        static: bool,
        group_shape: GroupShape,
        num_token_padding: int | None = None,
        column_major_scales: bool = False,
        use_ue8m0: bool | None = None,  # for Torch compile
        use_dynamic_scale: bool = False,
        scale_target: float = 16.0,
    ):
        """
        :param static: static or dynamic quantization
        :param group_shape: quantization group shape (PER_TOKEN, PER_TENSOR,
            or arbitrary block size)
        :param num_token_padding: Pad the token dimension of output to this
            size
        :param column_major_scales: For group quantization, output scales in
            column major format
        :param use_dynamic_scale: If True, compute per-token scale from x_max
            to map values into HiF8 sweet spot. Used by KV cache quantization
            and activation quantization with scale_target != 1.0.
            If False, use scale=1.0 (direct cast).
        :param scale_target: When use_dynamic_scale=True, scale = x_max /
            scale_target. Maps activations so their max lands at scale_target,
            placing most values in the HiF8 sweet spot [0.125, scale_target).
        """
        super().__init__()
        self.static = static
        self.group_shape = group_shape
        self.num_token_padding = num_token_padding
        self.column_major_scales = column_major_scales
        self.use_ue8m0 = use_ue8m0
        self.use_dynamic_scale = use_dynamic_scale
        self.scale_target = scale_target

        self.is_group_quant = group_shape.is_per_group()
        if self.is_group_quant:
            assert not static, "Group quantization only supports dynamic mode"
            self.group_size = group_shape.col
        else:
            assert group_shape in {GroupShape.PER_TOKEN, GroupShape.PER_TENSOR}
            assert not static or group_shape == GroupShape.PER_TENSOR, (
                "Only per-tensor scales supported for static quantization."
            )
            self.use_per_token_if_dynamic = group_shape == GroupShape.PER_TOKEN

    def forward_cuda(
        self,
        x: torch.Tensor,
        scale: torch.Tensor | None = None,
        scale_ub: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.forward_native(x, scale, scale_ub)

    def forward_native(
        self,
        x: torch.Tensor,
        scale: torch.Tensor | None = None,
        scale_ub: torch.Tensor | None = None,
    ):
        if self.is_group_quant:
            assert scale is None, "Group quantization is always dynamic"
            return self._quantize_group_native(x)

        assert (scale is not None) == self.static
        assert scale_ub is None or (
            not self.static
            and self.group_shape == GroupShape.PER_TOKEN
            and scale_ub.numel() == 1
        )

        if scale is None:
            if self.group_shape == GroupShape.PER_TOKEN:
                x_max, _ = x.abs().max(dim=-1)
                x_max = x_max.unsqueeze(-1).to(torch.float32)
                if scale_ub is not None:
                    x_max = x_max.clamp(max=scale_ub)
                # x_median, _ = x.median(dim=-1)
                # x_median = x_median.unsqueeze(-1).to(torch.float32)
            else:
                x_max = x.abs().max().unsqueeze(-1).to(torch.float32)
                # x_median = x.median().unsqueeze(-1).to(torch.float32)

            if self.use_dynamic_scale:
                scale = (x_max / self.scale_target).clamp(
                    min=_HIF8_MIN_SCALING_FACTOR)
            else:
                scale = torch.ones(
                    x_max.shape, dtype=torch.float32, device=x_max.device)

        # Even for dynamic per-token scales,
        # reciprocal performs slightly better than division
        out = x.to(torch.float32) * scale.reciprocal()
        out = out.clamp(_HIF8_MIN, _HIF8_MAX)
        out = hif8_cast.fake_quant(out).to(x.dtype)

        # This currently generates an extra Triton kernel in compilation.
        # Fortunately, we don't use padding if compiling.
        # TODO(luka): benchmark torch._scaled_mm to hopefully remove padding
        #  in general.
        if self.num_token_padding is not None:
            padding = max(self.num_token_padding - out.size(0), 0)
            out = F.pad(out, (0, 0, 0, padding), "constant", 0.0)

        return out, scale

    def _quantize_group_native(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        orig_shape = x.shape
        hidden_dim = x.shape[-1]
        num_groups = (hidden_dim + self.group_size - 1) // self.group_size
        padded_dim = num_groups * self.group_size

        if padded_dim != hidden_dim:
            padding = padded_dim - hidden_dim
            x = F.pad(x, (0, padding), mode="constant", value=0.0)

        x_grouped = x.view(-1, num_groups, self.group_size)
        absmax = x_grouped.abs().max(dim=-1, keepdim=True)[0].float()
        scales_raw = absmax / _HIF8_MAX
        if self.use_ue8m0:
            scales_raw = torch.exp2(torch.ceil(torch.log2(scales_raw)))
        scales = (scales_raw).clamp(min=_HIF8_MIN_SCALING_FACTOR)

        x_scaled = x_grouped / scales
        x_quant = x_scaled.clamp(_HIF8_MIN, _HIF8_MAX)
        x_quant = hif8_cast.fake_quant(x_quant).to(x_grouped.dtype)

        x_quant = x_quant.view(-1, padded_dim)
        if padded_dim != hidden_dim:
            x_quant = x_quant[..., :hidden_dim]
        x_quant = x_quant.view(orig_shape)

        scales = scales.squeeze(-1)
        scales = scales.reshape(orig_shape[:-1] + (num_groups,))

        if self.column_major_scales:
            scales = scales.transpose(-2, -1).contiguous().transpose(-1, -2)

        return x_quant, scales

def scaled_hif8_quant(
    input: torch.Tensor,
    scale: torch.Tensor | None = None,
    num_token_padding: int | None = None,
    scale_ub: torch.Tensor | None = None,
    use_per_token_if_dynamic: bool = False,
    output: torch.Tensor | None = None,
    use_wmax: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Fake quantize input tensor to HiF8 and return quantized tensor and scale.

    This function supports both static and dynamic quantization: If you
    provide the scale, it will use static scaling and if you omit it,
    the scale will be determined dynamically. The function also allows
    optional padding of the output tensors for downstream kernels that
    will benefit from padding.

    Args:
        input: The input tensor to be quantized to FP8
        scale: Optional scaling factor for the FP8 quantization
        scale_ub: Optional upper bound for scaling factor in dynamic
            per token case
        num_token_padding: If specified, pad the first dimension
            of the output to at least this value.
        use_per_token_if_dynamic: Whether to do per_tensor or per_token
            in the dynamic quantization case.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: The output tensor in FP8 and
            scaling factor.
    """
    # This code assumes batch_dim and num_tokens are flattened
    assert input.ndim == 2
    shape: tuple[int, int] | torch.Size = input.shape
    # For ROCm on MI300, the output fp8 dtype is torch.float_e3m3fnuz
    out_dtype = input.dtype
    if num_token_padding:
        shape = (max(num_token_padding, input.shape[0]), shape[1])
    if output is None:
        output = torch.empty(shape, device=input.device, dtype=out_dtype)
    else:
        assert num_token_padding is None, "padding not supported if output passed in"
        assert output.dtype == out_dtype

    if scale is None:
        if use_per_token_if_dynamic:
            input_max, _ = input.abs().max(dim=-1)
            # input_max, _ = input.median(dim=-1)
            input_max = input_max.unsqueeze(-1).to(torch.float32)
            if scale_ub is not None:
                input_max = input_max.clamp(max=scale_ub)
        else:
            input_max = input.abs().max().unsqueeze(-1).to(torch.float32)
            # input_max = input.median().unsqueeze(-1).to(torch.float32)

        if use_wmax:
            scale = (input_max / 24).clamp(min=_HIF8_MIN_SCALING_FACTOR)
        else:
            scale = torch.ones(input_max.shape, dtype=torch.float32, device=input_max.device)
        # scale = input_max
    else:
        assert scale.numel() == 1, f"{scale.shape}"

    # Even for dynamic per-token scales,
    # reciprocal performs slightly better than division
    output = input.to(torch.float32) * scale.reciprocal()
    output = output.clamp(_HIF8_MIN, _HIF8_MAX)
    output = hif8_cast.fake_quant(output).to(out_dtype)

    # This currently generates an extra Triton kernel in compilation.
    # Fortunately, we don't use padding if compiling.
    # TODO(luka): benchmark torch._scaled_mm to hopefully remove padding
    #  in general.
    if num_token_padding is not None:
        padding = max(num_token_padding - output.size(0), 0)
        output = F.pad(output, (0, 0, 0, padding), "constant", 0.0)

    return output, scale