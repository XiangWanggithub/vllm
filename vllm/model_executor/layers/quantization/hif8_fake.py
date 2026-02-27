# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Callable
from enum import Enum
from functools import partial
from typing import TYPE_CHECKING, Any, Optional

import torch
import torch.nn.functional as F
from torch.nn import Module
from torch.nn.parameter import Parameter

import vllm.envs as envs
import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm import _custom_ops as ops
from vllm._aiter_ops import rocm_aiter_ops
from vllm.attention.layer import Attention
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.logger import init_logger
from vllm.model_executor.layers.batch_invariant import (
    vllm_is_batch_invariant,
)
from vllm.model_executor.layers.fused_moe import (
    FusedMoE,
    FusedMoEActivationFormat,
    FusedMoEMethodBase,
    FusedMoEPermuteExpertsUnpermute,
    FusedMoEPrepareAndFinalize,
    FusedMoeWeightScaleSupported,
)
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEParallelConfig,
    FusedMoEQuantConfig,
    RoutingMethodType,
    hif8_w8a8_moe_quant_config,
)
from vllm.model_executor.layers.fused_moe.fused_marlin_moe import fused_marlin_moe
from vllm.model_executor.layers.fused_moe.layer import UnquantizedFusedMoEMethod
from vllm.model_executor.layers.linear import (
    LinearBase,
    LinearMethodBase,
    UnquantizedLinearMethod,
)
from vllm.model_executor.layers.quantization import QuantizationMethods
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from vllm.model_executor.layers.quantization.kv_cache import BaseKVCacheMethod
from vllm.model_executor.layers.quantization.utils.flashinfer_utils import (
    FlashinferMoeBackend,
    apply_flashinfer_per_tensor_scale_fp8,
    build_flashinfer_fp8_cutlass_moe_prepare_finalize,
    flashinfer_cutlass_moe_fp8,
    get_flashinfer_moe_backend,
    register_moe_scaling_factors,
    rotate_flashinfer_fp8_moe_weights,
    select_cutlass_fp8_gemm_impl,
    swap_w13_to_w31,
)
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    W8A8BlockFp8LinearOp,
    create_fp8_input_scale,
    create_fp8_scale_parameter,
    create_fp8_weight_parameter,
    deepgemm_post_process_fp8_weight_block,
    maybe_post_process_fp8_weight_block,
    process_fp8_weight_block_strategy,
    process_fp8_weight_tensor_strategy,
    validate_fp8_block_shape,
)
from vllm.model_executor.layers.quantization.utils.marlin_utils import (
    get_marlin_input_dtype,
)
from vllm.model_executor.layers.quantization.utils.marlin_utils_fp8 import (
    apply_fp8_marlin_linear,
    prepare_fp8_layer_for_marlin,
    prepare_moe_fp8_layer_for_marlin,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    GroupShape,
    is_layer_skipped,
)
from vllm.model_executor.layers.quantization.utils.w8a8_utils import (
    Fp8LinearOp,
    all_close_1d,
    cutlass_block_fp8_supported,
    cutlass_fp8_supported,
    maybe_create_device_identity,
    normalize_e4m3fn_to_e4m3fnuz,
    per_tensor_dequantize,
)
from vllm.model_executor.parameter import (
    BlockQuantScaleParameter,
    ModelWeightParameter,
    PerTensorScaleParameter,
    ChannelQuantScaleParameter,
)
from vllm.model_executor.utils import set_weight_attrs
from vllm.platforms import current_platform
from vllm.scalar_type import scalar_types
from vllm.utils.deep_gemm import (
    is_deep_gemm_e8m0_used,
    is_deep_gemm_supported,
)
from vllm.utils.flashinfer import has_flashinfer_moe
from vllm.utils.import_utils import has_deep_gemm

from vllm.model_executor.layers.quantization.input_quant_hif8_fake import QuantFakeHiF8, scaled_hif8_quant

if TYPE_CHECKING:
    from vllm.model_executor.models.utils import WeightsMapper

ACTIVATION_SCHEMES = ["static", "dynamic"]

_HIF8_MAX = 2.0**15*1.25
_HIF8_MIN = -2.0**15*1.25
_HIF8_MIN_SCALING_FACTOR = 1.0 / (_HIF8_MAX * 512.0)

logger = init_logger(__name__)

class Fp8MoeBackend(Enum):
    NONE = 0
    FLASHINFER_TRTLLM = 1
    FLASHINFER_CUTLASS = 2
    DEEPGEMM = 3
    CUTLASS_BLOCK_SCALED_GROUPED_GEMM = 4
    MARLIN = 5
    TRITON = 6

def get_fp8_moe_backend(
    block_quant: bool, moe_parallel_config: FusedMoEParallelConfig
) -> Fp8MoeBackend:
    """
    Select the primary FP8 MoE backend
    Note: Shape-specific fallbacks may still occur at runtime.
    """
    # Prefer FlashInfer backends on supported GPUs; allow SM90 and SM100.
    if (
        current_platform.is_cuda()
        and (
            current_platform.is_device_capability(100)
            or current_platform.is_device_capability(90)
        )
        and envs.VLLM_USE_FLASHINFER_MOE_FP8
        and has_flashinfer_moe()
    ):
        backend = get_flashinfer_moe_backend()
        if backend == FlashinferMoeBackend.TENSORRT_LLM:
            logger.info_once("Using FlashInfer FP8 MoE TRTLLM backend for SM100")
            return Fp8MoeBackend.FLASHINFER_TRTLLM
        else:
            if block_quant and current_platform.is_device_capability(100):
                raise ValueError(
                    "FlashInfer FP8 MoE throughput backend does not "
                    "support block quantization. Please use "
                    "VLLM_FLASHINFER_MOE_BACKEND=latency "
                    "instead."
                )
            logger.info_once("Using FlashInfer FP8 MoE CUTLASS backend for SM90/SM100")
            return Fp8MoeBackend.FLASHINFER_CUTLASS

    # weight-only path for older GPUs without native FP8
    use_marlin = (
        not current_platform.has_device_capability(89)
        or envs.VLLM_TEST_FORCE_FP8_MARLIN
    )
    if current_platform.is_rocm():
        use_marlin = False
    if use_marlin:
        logger.info_once("Using Marlin backend for FP8 MoE")
        return Fp8MoeBackend.MARLIN

    # Determine if we should use DeepGEMM with block-quantized weights:
    # - If explicitly set by user, respect their choice
    # - If not explicitly set (default), disable when TP size is >= 8
    moe_use_deep_gemm = envs.VLLM_MOE_USE_DEEP_GEMM
    if not envs.is_set("VLLM_MOE_USE_DEEP_GEMM") and moe_parallel_config.tp_size >= 8:
        moe_use_deep_gemm = False
        logger.info_once(
            "DeepGEMM MoE is disabled by default when TP size is >= 8. "
            "Set VLLM_MOE_USE_DEEP_GEMM=1 to enable it.",
            scope="local",
        )

    if envs.VLLM_USE_DEEP_GEMM and moe_use_deep_gemm and block_quant:
        if not has_deep_gemm():
            logger.warning_once(
                "DeepGEMM backend requested but not available.", scope="local"
            )
        elif is_deep_gemm_supported():
            logger.info_once("Using DeepGEMM backend for FP8 MoE", scope="local")
            return Fp8MoeBackend.DEEPGEMM

    # CUTLASS BlockScaled GroupedGemm on SM100 with block-quantized weights
    if (
        current_platform.is_cuda()
        and current_platform.is_device_capability(100)
        and block_quant
    ):
        logger.info_once(
            "Using Cutlass BlockScaled GroupedGemm backend for FP8 MoE", scope="local"
        )
        return Fp8MoeBackend.CUTLASS_BLOCK_SCALED_GROUPED_GEMM

    # default to Triton
    logger.info_once("Using Triton backend for FP8 MoE")
    return Fp8MoeBackend.TRITON


class HiF8FakeLinearOp:
    """
    This class executes a HiF8 fake quant linear layer using torch.nn.functional.linear.
    It needs to be a class instead of a method so that config can be read
    in the __init__ method, as reading config is not allowed inside forward.
    """

    def __init__(
        self,
        act_quant_static: bool,
        act_quant_group_shape: GroupShape = GroupShape.PER_TENSOR,
        pad_output: bool | None = None,
    ):
        self.preferred_backend = "torch"

        # Note: we pad the input because torch._scaled_mm is more performant
        # for matrices with batch dimension > 16.
        # This could change in the future.
        # We also don't pad when using torch.compile,
        # as it breaks with dynamic shapes.
        if pad_output is None:
            # config = get_current_vllm_config().compilation_config
            # pad_output = (
            #     config.mode < CompilationMode.VLLM_COMPILE
            #     and self.preferred_backend == "torch"
            # )
            pad_output = False

        self.output_padding = 17 if pad_output else None
        self.act_quant_static = act_quant_static
        self.act_quant_group_shape = act_quant_group_shape
        self.quant_hif8_fake = QuantFakeHiF8(
            static=act_quant_static,
            group_shape=act_quant_group_shape,
            num_token_padding=self.output_padding,
        )

    def apply(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
        out_dtype: torch.dtype | None = None,
        input_scale: torch.Tensor | None = None,
        input_scale_ub: torch.Tensor | None = None,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # ops.scaled_fp8_quant supports both dynamic and static quant.
        #   If dynamic, layer.input_scale is None and x_scale computed from x.
        #   If static, layer.input_scale is scalar and x_scale is input_scale.

        # View input as 2D matrix for fp8 methods
        input_2d = input.view(-1, input.shape[-1])
        output_shape = [*input.shape[:-1], weight.shape[0]]

        if out_dtype is None:
            out_dtype = input.dtype

        # If input not quantized
        if input_scale is None:
            qinput, x_scale = self.quant_hif8_fake(
                input_2d,
                input_scale,
                input_scale_ub,
            )
        else:
            qinput, x_scale = input_2d, input_scale

        # Must have dim() conditions
        # In per-token quant scenario, when the number of token is 1,
        # the scale will only have 1 elements.
        # Without checking the dim(),
        # we cannot distingushes between per-tensor and per-token quant.
        # Example:
        # When the number of token is 1, per-token scale is [[1]]
        # When per-tensor scale is [1] or ().
        per_tensor_weights = weight_scale.numel() == 1
        per_tensor_activations = (x_scale.numel() == 1) and x_scale.dim() < 2

        output = torch.nn.functional.linear(qinput.float(), weight.float())
        if per_tensor_weights:
            output = output * x_scale * weight_scale
        else:
            output = output * x_scale * weight_scale.t()

        if bias is not None:
            output = output + bias

        return output.to(out_dtype).view(*output_shape)


class HiF8FakeConfig(QuantizationConfig):
    """Config class for HiF8 fake quant."""

    def __init__(
        self,
        is_checkpoint_hif8_serialized: bool = False,
        activation_scheme: str = "dynamic",
        ignored_layers: list[str] | None = None,
        weight_block_size: list[int] | None = None,
        per_channel: bool = False,
    ) -> None:
        super().__init__()

        self.is_checkpoint_hif8_serialized = is_checkpoint_hif8_serialized

        if activation_scheme not in ACTIVATION_SCHEMES:
            raise ValueError(f"Unsupported activation scheme {activation_scheme}")
        self.activation_scheme = activation_scheme
        self.ignored_layers = ignored_layers or []
        if weight_block_size is not None:
            if not is_checkpoint_hif8_serialized:
                raise ValueError(
                    "The block-wise quantization only supports fp8-serialized "
                    "checkpoint for now."
                )
            if len(weight_block_size) != 2:
                raise ValueError(
                    "The quantization block size of weight must have 2 "
                    f"dimensions, but got {len(weight_block_size)} dimensions"
                )
            if activation_scheme != "dynamic":
                raise ValueError(
                    "The block-wise quantization only supports "
                    "dynamic activation scheme for now, but got "
                    f"{activation_scheme} activation scheme."
                )
        self.weight_block_size = weight_block_size
        self.per_channel = per_channel

    @classmethod
    def get_name(cls) -> QuantizationMethods:
        return "hif8_fake"

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.half, torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 80

    @classmethod
    def get_config_filenames(cls) -> list[str]:
        return []

    def apply_vllm_mapper(self, hf_to_vllm_mapper: "WeightsMapper"):
        if self.ignored_layers is not None:
            self.ignored_layers = hf_to_vllm_mapper.apply_list(self.ignored_layers)

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "HiF8FakeConfig":
        quant_method = cls.get_from_keys(config, ["quant_method"])
        is_checkpoint_hif8_serialized = "hif8" in quant_method and "fake" not in quant_method
        activation_scheme = cls.get_from_keys(config, ["activation_scheme"])
        ignored_layers = cls.get_from_keys_or(config, ["ignored_layers"], None)
        weight_block_size = cls.get_from_keys_or(config, ["weight_block_size"], None)
        per_channel = cls.get_from_keys_or(config, ["per_channel"], False)
        if not ignored_layers:
            ignored_layers = cls.get_from_keys_or(
                config, ["modules_to_not_convert"], None
            )
        return cls(
            is_checkpoint_hif8_serialized=is_checkpoint_hif8_serialized,
            activation_scheme=activation_scheme,
            ignored_layers=ignored_layers,
            weight_block_size=weight_block_size,
            per_channel=per_channel,
        )

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> Optional["QuantizeMethodBase"]:
        if current_platform.is_xpu():
            raise ValueError("XPU is not supported for HiFP8.")
        if isinstance(layer, LinearBase):
            if is_layer_skipped(
                prefix=prefix,
                ignored_layers=self.ignored_layers,
                fused_mapping=self.packed_modules_mapping,
            ):
                return UnquantizedLinearMethod()
            quant_method = HiF8FakeLinearMethod(self)
            return quant_method
        elif isinstance(layer, FusedMoE):
            if is_layer_skipped(
                prefix=prefix,
                ignored_layers=self.ignored_layers,
                fused_mapping=self.packed_modules_mapping,
            ):
                return UnquantizedFusedMoEMethod(layer.moe_config)
            moe_quant_method = HiF8FakeMoEMethod(self, layer)
            return moe_quant_method
        elif isinstance(layer, Attention):
            return HiF8FakeKVCacheMethod(self)
        return None

    def get_cache_scale(self, name: str) -> str | None:
        """
        Check whether the param name matches the format for k/v cache scales
        in compressed-tensors. If this is the case, return its equivalent
        param name expected by vLLM

        :param name: param name
        :return: matching param name for KV cache scale in vLLM
        """
        if name.endswith(".output_scale") and ".k_proj" in name:
            return name.replace(".k_proj.output_scale", ".attn.k_scale")
        if name.endswith(".output_scale") and ".v_proj" in name:
            return name.replace(".v_proj.output_scale", ".attn.v_scale")
        if name.endswith(".output_scale") and ".q_proj" in name:
            return name.replace(".q_proj.output_scale", ".attn.q_scale")
        if name.endswith("self_attn.prob_output_scale"):
            return name.replace(".prob_output_scale", ".attn.prob_scale")
        # If no matches, return None
        return None


class HiF8FakeLinearMethod(LinearMethodBase):
    """Linear method for HiF8 fake quant.
    Supports loading FP8 checkpoints with static weight scale and
    dynamic/static activation scale.

    Also supports loading quantized FP16/BF16 model checkpoints with dynamic
    activation scaling. The weight scaling factor will be initialized after
    the model weights are loaded.

    Limitations:
    1. Only support per-tensor quantization due to torch._scaled_mm support.
    2. Only support float8_e4m3fn data type due to the limitation of
       torch._scaled_mm (https://github.com/pytorch/pytorch/blob/2e48b39603411a41c5025efbe52f89560b827825/aten/src/ATen/native/cuda/Blas.cpp#L854-L856)

    Args:
        quant_config: The quantization config.
    """

    def __init__(self, quant_config: HiF8FakeConfig):
        self.quant_config = quant_config
        self.out_dtype = torch.get_default_dtype()

        self.use_deep_gemm = is_deep_gemm_supported()

        self.weight_block_size = self.quant_config.weight_block_size
        self.per_channel = self.quant_config.per_channel
        self.block_quant = self.weight_block_size is not None
        self.act_q_static = self.quant_config.activation_scheme == "static"
        if self.weight_block_size:
            self.act_q_group_shape = GroupShape(1, self.weight_block_size[0])
        else:
            # Use per-token quantization for better perf if dynamic and cutlass
            if not self.act_q_static and cutlass_fp8_supported():
                self.act_q_group_shape = GroupShape.PER_TOKEN
            else:
                self.act_q_group_shape = GroupShape.PER_TENSOR

        if self.block_quant:
            assert not self.act_q_static
            assert self.weight_block_size is not None
            self.w8a8_block_fp8_linear = None
            raise NotImplementedError("Weight block quant is not Supported.")
        else:
            self.hif8_fake_linear = HiF8FakeLinearOp(
                act_quant_static=self.act_q_static,
                act_quant_group_shape=self.act_q_group_shape,
            )

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        maybe_create_device_identity()

        output_size_per_partition = sum(output_partition_sizes)
        weight_loader = extra_weight_attrs.get("weight_loader")
        layer.logical_widths = output_partition_sizes
        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = output_size_per_partition
        layer.orig_dtype = params_dtype
        layer.weight_block_size = None

        if self.block_quant:
            assert self.weight_block_size is not None
            layer.weight_block_size = self.weight_block_size
            validate_fp8_block_shape(
                layer,
                input_size,
                output_size,
                input_size_per_partition,
                output_partition_sizes,
                self.weight_block_size,
            )

        # WEIGHT: we use original dtype no matter of whether it is pre-quantized
        weight = ModelWeightParameter(
            data=torch.empty(
                output_size_per_partition,
                input_size_per_partition,
                dtype=params_dtype,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight", weight)

        # If checkpoint is serialized hifp8, load weight scale.
        # Otherwise, wait until process_weights_after_loading.
        if self.quant_config.is_checkpoint_hif8_serialized:
            scale = ChannelQuantScaleParameter(
                data=torch.empty((sum(output_partition_sizes), 1), dtype=torch.float32),
                output_dim=0,
                weight_loader=weight_loader,
            )
            raise NotImplementedError("Loading HiF8 weights is not supported.")

    def process_weights_after_loading(self, layer: Module) -> None:
        size_k_first = True
        input_scale = None
        # TODO(rob): refactor block quant into separate class.
        if self.block_quant:
            raise NotImplementedError("Weight block quant is not supported.")

        # If checkpoint not serialized fp8, quantize the weights.
        elif not self.quant_config.is_checkpoint_hif8_serialized:
            #qweight, weight_scale = ops.scaled_fp8_quant(layer.weight, scale=None)
            weight, weight_scale = scaled_hif8_quant(layer.weight, scale=None, use_per_token_if_dynamic=self.per_channel, use_wmax=False)
            #weight = qweight.t()

        # If checkpoint is fp8 per-tensor, handle that there are N scales for N
        # shards in a fused module
        else:
            weight = layer.weight
            weight_scale = layer.weight_scale

        # Update layer with new values.
        layer.weight = Parameter(weight.data, requires_grad=False)
        layer.weight_scale = Parameter(weight_scale.data, requires_grad=False)
        layer.input_scale = (
            Parameter(input_scale, requires_grad=False)
            if input_scale is not None
            else None
        )

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # if batch invariant mode is enabled, prefer DeepGEMM FP8 path
        # we will use BF16 dequant when DeepGEMM is not supported.
        if vllm_is_batch_invariant():
            if self.block_quant:
                assert self.weight_block_size is not None
                return self.w8a8_block_fp8_linear.apply(
                    input=x,
                    weight=layer.weight,
                    weight_scale=layer.weight_scale,
                    input_scale=layer.input_scale,
                    bias=bias,
                )
            else:
                # per-tensor/channel: dequant to BF16 and run GEMM
                weight_fp8 = layer.weight.to(torch.bfloat16)
                weight_scale = layer.weight_scale.to(torch.bfloat16)
                if weight_scale.numel() == 1:
                    # Per-tensor: simple scalar multiplication
                    weight_bf16 = weight_fp8 * weight_scale
                else:
                    # Multiple scales (fused modules like QKV)
                    # Try to infer correct broadcasting
                    # weight is [K, N], scale could be [num_logical_weights]
                    # Need to figure out how to broadcast - for now just try
                    # direct multiplication
                    if (
                        weight_scale.dim() == 1
                        and weight_scale.shape[0] == weight_fp8.shape[0]
                    ):
                        # Per-row scaling
                        weight_bf16 = weight_fp8 * weight_scale.unsqueeze(1)
                    else:
                        # Fallback
                        weight_bf16 = weight_fp8 * weight_scale
                return torch.nn.functional.linear(x, weight_bf16.t(), bias)

        if self.block_quant:
            assert self.weight_block_size is not None
            raise NotImplementedError("Weight block quant is not supported.")
            return self.w8a8_block_fp8_linear.apply(
                input=x,
                weight=layer.weight,
                weight_scale=layer.weight_scale,
                input_scale=layer.input_scale,
                bias=bias,
            )

        return self.hif8_fake_linear.apply(
            input=x,
            weight=layer.weight,
            weight_scale=layer.weight_scale,
            out_dtype=self.out_dtype,
            input_scale=layer.input_scale,
            bias=bias,
        )


class HiF8FakeMoEMethod(FusedMoEMethodBase):
    """MoE method for FP8.
    Supports loading FP8 checkpoints with static weight scale and
    dynamic/static activation scale.

    Also supports loading quantized FP16/BF16 model checkpoints with dynamic
    activation scaling. The weight scaling factor will be initialized after
    the model weights are loaded.

    Args:
        quant_config: The quantization config.
    """

    def __init__(self, quant_config: HiF8FakeConfig, layer: torch.nn.Module):
        super().__init__(layer.moe_config)
        self.layer = layer
        self.quant_config = quant_config
        self.weight_block_size = self.quant_config.weight_block_size
        self.block_quant: bool = self.weight_block_size is not None
        self.per_channel = self.quant_config.per_channel
        self.act_pertoken = self.quant_config.activation_scheme == "dynamic"

    def create_weights(
        self,
        layer: Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        layer.intermediate_size_per_partition = intermediate_size_per_partition
        layer.hidden_size = hidden_size
        layer.num_experts = num_experts
        layer.orig_dtype = params_dtype
        layer.weight_block_size = None

        if self.quant_config.is_checkpoint_hif8_serialized:
            raise NotImplementedError("Real HiF8 is not supported.")
        if self.block_quant:
            assert self.weight_block_size is not None
            layer.weight_block_size = self.weight_block_size
            tp_size = get_tensor_model_parallel_world_size()
            block_n, block_k = (
                self.weight_block_size[0],
                self.weight_block_size[1],
            )
            # NOTE: To ensure proper alignment of the block-wise quantization
            # scales, the output_size of the weights for both the gate and up
            # layers must be divisible by block_n.
            # Required by column parallel or enabling merged weights
            if intermediate_size_per_partition % block_n != 0:
                raise ValueError(
                    f"The output_size of gate's and up's weight = "
                    f"{intermediate_size_per_partition} is not divisible by "
                    f"weight quantization block_n = {block_n}."
                )
            if tp_size > 1 and intermediate_size_per_partition % block_k != 0:
                # Required by row parallel
                raise ValueError(
                    f"The input_size of down's weight = "
                    f"{intermediate_size_per_partition} is not divisible by "
                    f"weight quantization block_k = {block_k}."
                )

        # WEIGHTS
        w13_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                2 * intermediate_size_per_partition,
                hidden_size,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)

        w13_bias = torch.nn.Parameter(
            torch.zeros(
                num_experts,
                2 * intermediate_size_per_partition,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_bias", w13_bias)
        set_weight_attrs(w13_bias, extra_weight_attrs)

        w2_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                hidden_size,
                intermediate_size_per_partition,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

        w2_bias = torch.nn.Parameter(
            torch.zeros(
                num_experts,
                hidden_size,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_bias", w2_bias)
        set_weight_attrs(w2_bias, extra_weight_attrs)

        # WEIGHT_SCALES
        if not self.block_quant:
            # Per-tensor for weight
            # Allocate 2 scales for w1 and w3 respectively.
            # They will be combined to a single scale after weight loading.
            w13_weight_scale = torch.nn.Parameter(
                torch.ones(num_experts, 2, dtype=torch.float32), requires_grad=False
            )
            w2_weight_scale = torch.nn.Parameter(
                torch.ones(num_experts, dtype=torch.float32), requires_grad=False
            )
            layer.register_parameter("w13_weight_scale", w13_weight_scale)
            layer.register_parameter("w2_weight_scale", w2_weight_scale)
        elif not self.block_quant and self.per_channel:
            w13_weight_scale = torch.nn.Parameter(
                torch.ones(
                    num_experts,
                    2 * intermediate_size_per_partition,
                    dtype=torch.float32,
                ),
                requires_grad=False,
            )
            w2_weight_scale = torch.nn.Parameter(
                torch.ones(
                    num_experts,
                    hidden_size,
                    dtype=torch.float32,
                ),
                requires_grad=False,
            )
            layer.register_parameter("w13_weight_scale", w13_weight_scale)
            layer.register_parameter("w2_weight_scale", w2_weight_scale)
        else:
            w13_weight_scale = torch.nn.Parameter(
                torch.ones(
                    num_experts,
                    2 * ((intermediate_size_per_partition + block_n - 1) // block_n),
                    (hidden_size + block_k - 1) // block_k,
                    dtype=torch.float32,
                ),
                requires_grad=False,
            )
            w2_weight_scale = torch.nn.Parameter(
                torch.ones(
                    num_experts,
                    (hidden_size + block_n - 1) // block_n,
                    (intermediate_size_per_partition + block_k - 1) // block_k,
                    dtype=torch.float32,
                ),
                requires_grad=False,
            )
            layer.register_parameter("w13_weight_scale_inv", w13_weight_scale)
            layer.register_parameter("w2_weight_scale_inv", w2_weight_scale)
            assert self.quant_config.activation_scheme == "dynamic"

        # Add the quantization method used (per tensor/grouped/channel)
        # to ensure the weight scales are loaded in properly
        if self.block_quant:
            extra_weight_attrs.update({"quant_method": FusedMoeWeightScaleSupported.BLOCK.value})
        elif self.per_channel:
            extra_weight_attrs.update({"quant_method": FusedMoeWeightScaleSupported.CHANNEL.value})
        else:
            extra_weight_attrs.update({"quant_method": FusedMoeWeightScaleSupported.TENSOR.value})

        # If loading fp8 checkpoint, pass the weight loaders.
        # If loading an fp16 checkpoint, do not (we will quantize in
        #   process_weights_after_loading()
        if self.quant_config.is_checkpoint_hif8_serialized:
            set_weight_attrs(w13_weight_scale, extra_weight_attrs)
            set_weight_attrs(w2_weight_scale, extra_weight_attrs)

        # INPUT_SCALES
        if self.quant_config.activation_scheme == "static":
            if not self.quant_config.is_checkpoint_hif8_serialized:
                raise ValueError(
                    "Found static activation scheme for checkpoint that "
                    "was not serialized hif8."
                )

            w13_input_scale = torch.nn.Parameter(
                torch.ones(num_experts, dtype=torch.float32), requires_grad=False
            )
            layer.register_parameter("w13_input_scale", w13_input_scale)
            set_weight_attrs(w13_input_scale, extra_weight_attrs)

            w2_input_scale = torch.nn.Parameter(
                torch.ones(num_experts, dtype=torch.float32), requires_grad=False
            )
            layer.register_parameter("w2_input_scale", w2_input_scale)
            set_weight_attrs(w2_input_scale, extra_weight_attrs)

        else:
            layer.w13_input_scale = None
            layer.w2_input_scale = None

        self.rocm_aiter_moe_enabled = False

    def process_weights_after_loading(self, layer: Module) -> None:
        # TODO (rob): refactor block quant into separate class.
        if self.block_quant:
            assert self.quant_config.activation_scheme == "dynamic"
            w13_weight = layer.w13_weight.data
            w13_weight_scale_inv = layer.w13_weight_scale_inv.data
            w2_weight = layer.w2_weight
            w2_weight_scale_inv = layer.w2_weight_scale_inv

            # torch.compile() cannot use Parameter subclasses.
            layer.w13_weight = Parameter(w13_weight, requires_grad=False)
            layer.w13_weight_scale_inv = Parameter(
                w13_weight_scale_inv, requires_grad=False
            )
            layer.w2_weight = Parameter(w2_weight, requires_grad=False)
            layer.w2_weight_scale_inv = Parameter(
                w2_weight_scale_inv, requires_grad=False
            )

        # If checkpoint is fp16, quantize in place.
        elif not self.quant_config.is_checkpoint_hif8_serialized:
            quantized_dtype = layer.w13_weight.data.dtype
            w13_weight = torch.empty_like(layer.w13_weight.data, dtype=quantized_dtype)
            w2_weight = torch.empty_like(layer.w2_weight.data, dtype=quantized_dtype)

            # Re-initialize w13_scale because we directly quantize
            # merged w13 weights and generate a single scaling factor.
            if self.per_channel:
                layer.w13_weight_scale = torch.nn.Parameter(
                    torch.ones(
                        layer.local_num_experts,
                        layer.w13_weight.data.size(1),
                        1,
                        dtype=torch.float32,
                        device=w13_weight.device,
                    ),
                    requires_grad=False,
                )
                layer.w2_weight_scale = torch.nn.Parameter(
                    torch.ones(
                        layer.local_num_experts,
                        layer.w2_weight.data.size(1),
                        1,
                        dtype=torch.float32,
                        device=w2_weight.device,
                    ),
                    requires_grad=False,
                )
            else:
                layer.w13_weight_scale = torch.nn.Parameter(
                    torch.ones(
                        layer.local_num_experts,
                        dtype=torch.float32,
                        device=w13_weight.device,
                    ),
                    requires_grad=False,
                )
                layer.w2_weight_scale = torch.nn.Parameter(
                    torch.ones(
                        layer.local_num_experts,
                        dtype=torch.float32,
                        device=w2_weight.device,
                    ),
                    requires_grad=False,
                )
            for expert in range(layer.local_num_experts):
                w13_weight[expert, :, :], layer.w13_weight_scale[expert] = (
                    scaled_hif8_quant(layer.w13_weight.data[expert, :, :], use_per_token_if_dynamic=self.per_channel, use_wmax=False)
                )
                w2_weight[expert, :, :], layer.w2_weight_scale[expert] = (
                    scaled_hif8_quant(layer.w2_weight.data[expert, :, :], use_per_token_if_dynamic=self.per_channel, use_wmax=False)
                )
            layer.w13_weight = torch.nn.Parameter(w13_weight, requires_grad=False)
            layer.w2_weight = torch.nn.Parameter(w2_weight, requires_grad=False)
            w13_bias = layer.w13_bias.data.to(torch.float32)
            w2_bias = layer.w2_bias.data.to(torch.float32)
            layer.w13_bias = torch.nn.Parameter(w13_bias, requires_grad=False)
            layer.w2_bias = torch.nn.Parameter(w2_bias, requires_grad=False)
        # If checkpoint is fp8, we need to handle that the
        # MoE kernels require single activation scale and single weight
        # scale for w13 per expert.
        else:
            raise NotImplementedError("Loading pre-quantized HiF8 is not supported yet.")

    def maybe_make_prepare_finalize(
        self,
        routing_tables: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    ) -> mk.FusedMoEPrepareAndFinalize | None:
        return super().maybe_make_prepare_finalize(routing_tables)

    def select_gemm_impl(
        self,
        prepare_finalize: FusedMoEPrepareAndFinalize,
        layer: torch.nn.Module,
    ) -> FusedMoEPermuteExpertsUnpermute:
        from vllm.model_executor.layers.fused_moe import (
            BatchedDeepGemmExperts,
            BatchedTritonExperts,
            TritonOrDeepGemmExperts,
        )

        assert self.moe_quant_config is not None

        if (
            prepare_finalize.activation_format
            == FusedMoEActivationFormat.BatchedExperts
        ):
            max_num_tokens_per_rank = prepare_finalize.max_num_tokens_per_rank()
            assert max_num_tokens_per_rank is not None

            experts_impl = (
                BatchedDeepGemmExperts if self.allow_deep_gemm else BatchedTritonExperts
            )
            logger.debug(
                "%s(%s): max_tokens_per_rank=%s, block_size=%s, per_act_token=%s",
                experts_impl.__name__,
                self.__class__.__name__,
                max_num_tokens_per_rank,
                self.weight_block_size,
                False,
            )
            return experts_impl(
                max_num_tokens=max_num_tokens_per_rank,
                num_dispatchers=prepare_finalize.num_dispatchers(),
                quant_config=self.moe_quant_config,
            )
        else:
            logger.debug(
                "TritonOrDeepGemmExperts(%s): block_size=%s, per_act_token=%s",
                self.__class__.__name__,
                self.weight_block_size,
                False,
            )
            return TritonOrDeepGemmExperts(
                quant_config=self.moe_quant_config,
                allow_deep_gemm=False,
            )

    def get_fused_moe_quant_config(
        self, layer: torch.nn.Module
    ) -> FusedMoEQuantConfig | None:
        return hif8_w8a8_moe_quant_config(
            w1_scale=(
                layer.w13_weight_scale_inv
                if self.block_quant
                else layer.w13_weight_scale
            ),
            w2_scale=(
                layer.w2_weight_scale_inv if self.block_quant else layer.w2_weight_scale
            ),
            a1_scale=layer.w13_input_scale,
            a2_scale=layer.w2_input_scale,
            block_shape=self.weight_block_size,
            per_act_token_quant=self.act_pertoken,
            per_out_ch_quant=self.per_channel,
            w1_bias=layer.w13_bias,
            w2_bias=layer.w2_bias,
        )

    @property
    def supports_eplb(self) -> bool:
        return True

    @property
    def allow_inplace(self) -> bool:
        return True

    def apply(
        self,
        layer: FusedMoE,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        top_k: int,
        renormalize: bool,
        use_grouped_topk: bool = False,
        topk_group: int | None = None,
        num_expert_group: int | None = None,
        global_num_experts: int = -1,
        expert_map: torch.Tensor | None = None,
        custom_routing_function: Callable | None = None,
        scoring_func: str = "softmax",
        routed_scaling_factor: float = 1.0,
        e_score_correction_bias: torch.Tensor | None = None,
        apply_router_weight_on_input: bool = False,
        activation: str = "silu",
        enable_eplb: bool = False,
        expert_load_view: torch.Tensor | None = None,
        logical_to_physical_map: torch.Tensor | None = None,
        logical_replica_count: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if enable_eplb:
            assert expert_load_view is not None
            assert logical_to_physical_map is not None
            assert logical_replica_count is not None
            assert isinstance(layer, FusedMoE)

        select_result = layer.select_experts(
            hidden_states=x,
            router_logits=router_logits,
        )

        topk_weights, topk_ids, zero_expert_result = select_result

        from vllm.model_executor.layers.fused_moe import fused_experts

        result = fused_experts(
            hidden_states=x,
            w1=layer.w13_weight,
            w2=layer.w2_weight,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            inplace=True,
            activation=activation,
            global_num_experts=global_num_experts,
            apply_router_weight_on_input=apply_router_weight_on_input,
            expert_map=expert_map,
            quant_config=self.moe_quant_config,
            allow_deep_gemm=False,
            allow_cutlass_block_scaled_grouped_gemm=False,
        )

        if layer.zero_expert_num != 0 and layer.zero_expert_type is not None:
            assert not isinstance(result, tuple), (
                "Shared + zero experts are mutually exclusive not yet supported"
            )
            return result, zero_expert_result
        else:
            return result


class HiF8FakeKVCacheMethod(BaseKVCacheMethod):
    """
    Supports loading kv-cache scaling factors from FP8 checkpoints.
    """

    def __init__(self, quant_config: HiF8FakeConfig):
        super().__init__(quant_config)
        self.quant_hif8_fake = QuantFakeHiF8(
            static=False,
            group_shape=GroupShape.PER_TOKEN,
        )

    def apply(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        scale_ub: torch.Tensor | None = None,
    ):
        assert len(key.shape) == 3, "Key dim needs to be (-1, num_heads, head_dim)"
        assert len(value.shape) == 3, "Value dim needs to be (-1, num_heads, head_dim)"
        key, key_scales = self.quant_hif8_fake(key)
        value, value_scales = self.quant_hif8_fake(value)
        key = (key * key_scales).to(key.dtype)
        value = (value * value_scales).to(value.dtype)

        return key, value
    