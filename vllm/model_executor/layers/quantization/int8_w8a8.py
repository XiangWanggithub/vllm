# SPDX-License-Identifier: Apache-2.0
# INT8 W8A8C8 fake quantization for GPT-OSS-20B.
#
# Simulates INT8 quantization noise while computing in BF16:
#   - Weights:      per-channel symmetric INT8 (scale computed from weight amax)
#   - Activations:  per-token  symmetric INT8 (scale computed dynamically)
#   - KV cache:     per-token-per-head symmetric INT8
#   - SWA layers:   BF16 KV cache passthrough (no quantization)
#   - Sink tokens:  BF16 (position-0 protected from quantization noise)
#
# Usage:
#   vllm serve <model> --quantization int8_w8a8 --kv-cache-dtype int8_w8a8

import re
from collections.abc import Callable
from typing import Any, Optional

import torch
import torch.nn.functional as F
from torch.nn import Module
from torch.nn.parameter import Parameter

from vllm.attention.layer import Attention
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe import (
    FusedMoE,
    FusedMoEMethodBase,
    FusedMoeWeightScaleSupported,
)
from vllm.model_executor.layers.fused_moe.config import (
    biased_moe_quant_config,
)
from vllm.model_executor.layers.fused_moe.layer import UnquantizedFusedMoEMethod
from vllm.model_executor.layers.linear import (
    LinearBase,
    LinearMethodBase,
    UnquantizedLinearMethod,
)
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from vllm.model_executor.layers.quantization.kv_cache import BaseKVCacheMethod
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    is_layer_skipped,
)
from vllm.model_executor.parameter import ModelWeightParameter
from vllm.model_executor.utils import set_weight_attrs

logger = init_logger(__name__)

# ── INT8 constants ───────────────────────────────────────────────────────────
INT8_MAX = 127.0
INT8_MIN = -127.0   # symmetric, avoid -128 for numerical stability
INT8_EPS = 1e-12     # minimum scale to avoid division by zero


# ── Core quantization helpers ────────────────────────────────────────────────

def _int8_fake_quant_per_channel(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-channel (per-row) INT8 fake quantization for 2D weight tensors.

    Args:
        w: Weight tensor [out_features, in_features].

    Returns:
        (w_q, scale) where w_q has INT8-level values in original dtype,
        scale has shape [out_features, 1].
    """
    amax = w.float().abs().amax(dim=-1, keepdim=True)          # [out, 1]
    scale = (amax / INT8_MAX).clamp(min=INT8_EPS)              # [out, 1]
    w_q = (w.float() / scale).round().clamp(INT8_MIN, INT8_MAX)
    return w_q.to(w.dtype), scale


def _int8_fake_quant_per_token(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-token (per-row) INT8 fake quantization for 2D activation tensors.

    Args:
        x: Activation tensor [num_tokens, hidden_dim].

    Returns:
        (x_q, scale) where x_q has INT8-level values in original dtype,
        scale has shape [num_tokens, 1].
    """
    amax = x.float().abs().amax(dim=-1, keepdim=True)          # [T, 1]
    scale = (amax / INT8_MAX).clamp(min=INT8_EPS)
    x_q = (x.float() / scale).round().clamp(INT8_MIN, INT8_MAX)
    return x_q.to(x.dtype), scale


def _int8_fake_quant_3d(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-token-per-head INT8 fake quantization for 3D KV tensors.

    Args:
        x: Tensor [num_tokens, num_heads, head_dim].

    Returns:
        (x_dq, scale) where x_dq is dequantized (noise-added) in original dtype,
        scale has shape [num_tokens, num_heads, 1].
    """
    amax = x.float().abs().amax(dim=-1, keepdim=True)          # [T, H, 1]
    scale = (amax / INT8_MAX).clamp(min=INT8_EPS)
    x_q = (x.float() / scale).round().clamp(INT8_MIN, INT8_MAX)
    x_dq = (x_q * scale).to(x.dtype)
    return x_dq, scale


# ── Config ───────────────────────────────────────────────────────────────────

class Int8W8A8Config(QuantizationConfig):
    """Config class for INT8 W8A8C8 fake quantization."""

    def __init__(
        self,
        activation_scheme: str = "dynamic",
        ignored_layers: list[str] | None = None,
    ) -> None:
        super().__init__()
        if activation_scheme != "dynamic":
            raise ValueError(
                f"INT8 W8A8 only supports dynamic activation, got {activation_scheme}"
            )
        self.activation_scheme = activation_scheme
        self.ignored_layers = ignored_layers or []

    @classmethod
    def get_name(cls) -> str:
        return "int8_w8a8"

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.half, torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 75   # INT8 tensor cores from Turing (SM75)

    @classmethod
    def get_config_filenames(cls) -> list[str]:
        return []

    def apply_vllm_mapper(self, hf_to_vllm_mapper):
        if self.ignored_layers:
            self.ignored_layers = hf_to_vllm_mapper.apply_list(self.ignored_layers)

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "Int8W8A8Config":
        activation_scheme = cls.get_from_keys_or(
            config, ["activation_scheme"], "dynamic"
        )
        ignored_layers = cls.get_from_keys_or(config, ["ignored_layers"], None)
        return cls(activation_scheme=activation_scheme, ignored_layers=ignored_layers)

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> Optional[QuantizeMethodBase]:
        if isinstance(layer, LinearBase):
            if is_layer_skipped(
                prefix=prefix,
                ignored_layers=self.ignored_layers,
                fused_mapping=self.packed_modules_mapping,
            ):
                return UnquantizedLinearMethod()
            return Int8W8A8LinearMethod(self)

        elif isinstance(layer, FusedMoE):
            if is_layer_skipped(
                prefix=prefix,
                ignored_layers=self.ignored_layers,
                fused_mapping=self.packed_modules_mapping,
            ):
                return UnquantizedFusedMoEMethod(layer.moe_config)
            return Int8W8A8MoEMethod(self, layer)

        elif isinstance(layer, Attention):
            # Skip KV quantization for sliding-window attention layers
            if getattr(layer, "sliding_window", None) is not None:
                return None   # SWA → BF16 KV cache passthrough
            return Int8W8A8KVCacheMethod(self, prefix)

        return None


# ── Linear method ────────────────────────────────────────────────────────────

class Int8W8A8LinearMethod(LinearMethodBase):
    """Per-channel weight + per-token activation INT8 fake quantization."""

    def __init__(self, quant_config: Int8W8A8Config):
        self.quant_config = quant_config
        self.out_dtype = torch.get_default_dtype()

    def create_weights(
        self,
        layer: Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        output_size_per_partition = sum(output_partition_sizes)
        weight_loader = extra_weight_attrs.get("weight_loader")
        layer.logical_widths = output_partition_sizes
        layer.orig_dtype = params_dtype

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

    def process_weights_after_loading(self, layer: Module) -> None:
        # Per-channel INT8 fake quantization of weights
        w_q, w_scale = _int8_fake_quant_per_channel(layer.weight.data)
        # w_q: INT8-level values stored as BF16, shape [out, in]
        # w_scale: float32, shape [out, 1]
        layer.weight = Parameter(w_q, requires_grad=False)
        layer.weight_scale = Parameter(w_scale.squeeze(-1), requires_grad=False)

    def apply(
        self,
        layer: Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        input_2d = x.view(-1, x.shape[-1])
        output_shape = [*x.shape[:-1], layer.weight.shape[0]]

        # Per-token activation quantization
        x_q, x_scale = _int8_fake_quant_per_token(input_2d)

        # Matmul in float32 (fake quant: both operands are BF16 INT8-level vals)
        output = F.linear(x_q.float(), layer.weight.float())
        # Dequant: x_scale [T, 1] * weight_scale [out]
        output = output * x_scale * layer.weight_scale.unsqueeze(0)

        if bias is not None:
            output = output + bias.float()

        return output.to(self.out_dtype).view(*output_shape)


# ── MoE method ───────────────────────────────────────────────────────────────

class Int8W8A8MoEMethod(FusedMoEMethodBase):
    """INT8 W8A8 MoE: per-channel weight quant + pre-quantized activations.

    Weights are fake-quantized then dequanted back to BF16 (INT8 noise baked in)
    so the fused MoE kernel runs unquantized BF16 compute.  Activations are
    pre-quantized to INT8 outside the kernel before expert dispatch.
    """

    def __init__(self, quant_config: Int8W8A8Config, layer: Module):
        super().__init__(layer.moe_config)
        self.quant_config = quant_config

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

        # ── weights ──
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
                num_experts, 2 * intermediate_size_per_partition, dtype=params_dtype
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
            torch.zeros(num_experts, hidden_size, dtype=params_dtype),
            requires_grad=False,
        )
        layer.register_parameter("w2_bias", w2_bias)
        set_weight_attrs(w2_bias, extra_weight_attrs)

        # No per-channel scales stored: dequanted weights bake in INT8 noise.
        # No input scales: dynamic per-token quantization.
        layer.w13_input_scale = None
        layer.w2_input_scale = None

    def process_weights_after_loading(self, layer: Module) -> None:
        # Per-expert per-channel INT8 fake quant → dequant back to BF16.
        # The resulting weights have INT8 quantization noise baked in.
        w13 = layer.w13_weight.data.clone()
        w2 = layer.w2_weight.data.clone()

        for e in range(layer.local_num_experts):
            w13_q, w13_s = _int8_fake_quant_per_channel(w13[e])  # [2*inter, hidden]
            w13[e] = (w13_q.float() * w13_s).to(w13.dtype)       # dequant with noise

            w2_q, w2_s = _int8_fake_quant_per_channel(w2[e])     # [hidden, inter]
            w2[e] = (w2_q.float() * w2_s).to(w2.dtype)

        layer.w13_weight = Parameter(w13, requires_grad=False)
        layer.w2_weight = Parameter(w2, requires_grad=False)
        # Cast biases to float32 (required by fused_experts kernel)
        layer.w13_bias = Parameter(
            layer.w13_bias.data.to(torch.float32), requires_grad=False
        )
        layer.w2_bias = Parameter(
            layer.w2_bias.data.to(torch.float32), requires_grad=False
        )

    def get_fused_moe_quant_config(
        self, layer: Module
    ):
        # Unquantized compute with biases (weights already dequanted with noise)
        return biased_moe_quant_config(
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
        # Pre-quantize activation to INT8 (fake quant → dequant)
        x_flat = x.view(-1, x.shape[-1])
        x_q, x_scale = _int8_fake_quant_per_token(x_flat)
        x_dq = (x_q.float() * x_scale).to(x.dtype).view(x.shape)

        # Route using original activation (routing is BF16 in real deployment)
        select_result = layer.select_experts(
            hidden_states=x,
            router_logits=router_logits,
        )
        topk_weights, topk_ids, zero_expert_result = select_result

        from vllm.model_executor.layers.fused_moe import fused_experts

        result = fused_experts(
            hidden_states=x_dq,
            w1=layer.w13_weight,
            w2=layer.w2_weight,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            inplace=True,
            activation=activation,
            global_num_experts=global_num_experts,
            apply_router_weight_on_input=apply_router_weight_on_input,
            expert_map=expert_map,
            quant_config=self.get_fused_moe_quant_config(layer),
        )

        if layer.zero_expert_num != 0 and layer.zero_expert_type is not None:
            assert not isinstance(result, tuple)
            return result, zero_expert_result
        return result


# ── KV cache method ──────────────────────────────────────────────────────────

class Int8W8A8KVCacheMethod(BaseKVCacheMethod):
    """Per-token-per-head INT8 fake quantization on KV cache.

    Only applied to full-attention layers (SWA layers are skipped via
    get_quant_method returning None).  Sink tokens at position 0 are
    preserved in BF16 to avoid amplified quantization error on the
    always-attended initial token.
    """

    N_SINK_TOKENS = 1

    def __init__(self, quant_config: Int8W8A8Config, prefix: str = ""):
        super().__init__(quant_config)

    def apply(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        scale_ub: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
    ):
        assert key.ndim == 3, f"Key must be [T, H, D], got {key.shape}"
        assert value.ndim == 3, f"Value must be [T, H, D], got {value.shape}"

        # Protect sink tokens
        has_sinks = positions is not None and self.N_SINK_TOKENS > 0
        if has_sinks:
            original_key = key.clone()
            original_value = value.clone()

        # Per-token-per-head INT8 fake quantization
        key, _ = _int8_fake_quant_3d(key)
        value, _ = _int8_fake_quant_3d(value)

        # Restore sink tokens to original BF16
        if has_sinks:
            sink_mask = (positions < self.N_SINK_TOKENS).unsqueeze(-1).unsqueeze(-1)
            key = torch.where(sink_mask, original_key, key)
            value = torch.where(sink_mask, original_value, value)

        return key, value
