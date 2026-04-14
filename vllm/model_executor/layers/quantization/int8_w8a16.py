# SPDX-License-Identifier: Apache-2.0
"""
INT8 W8A16 weight-only quantization.

Two modes, selected automatically:

  from_int8_checkpoint=False (default, --quantization int8_w8a16 on BF16 model):
    Loads BF16 weights, quantizes per-channel to INT8 in process_weights_after_loading,
    dequantizes back to BF16 before each matmul.  MoE delegates to ExpertsInt8MoEMethod.

  from_int8_checkpoint=True (config.json quant_method=int8_w8a16 in INT8 checkpoint):
    Registers weight as int8 + weight_scale as float32, loading directly from the
    INT8 checkpoint (82 GB vs 162 GB from disk).  MoE loads per-expert int8 weights
    and stacks scales via FusedMoE's _load_per_channel_weight_scale.
    Dequantizes to BF16 in process_weights_after_loading; inference runs in BF16.

All GEMMs run in BF16 — no INT8 GEMM kernels required, so this works on SM100
(B200/Blackwell) where cutlass_scaled_mm INT8 is not implemented.

Usage
-----
  # BF16 model (quantize in-memory):
  --model_args "pretrained=/path/to/bf16,quantization=int8_w8a16"

  # INT8 checkpoint (load directly):
  --model_args "pretrained=/path/to/int8_checkpoint"
  (quant_method=int8_w8a16 in config.json triggers from_int8_checkpoint=True)
"""

from collections.abc import Callable
from typing import Any, Optional

import torch
import torch.nn.functional as F
from torch.nn import Module
from torch.nn.parameter import Parameter

from vllm.model_executor.layers.fused_moe import FusedMoE, FusedMoEMethodBase
from vllm.model_executor.layers.fused_moe.config import FUSED_MOE_UNQUANTIZED_CONFIG
from vllm.model_executor.layers.linear import LinearBase, LinearMethodBase
from vllm.model_executor.layers.quantization import QuantizationMethods
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from vllm.model_executor.layers.quantization.experts_int8 import ExpertsInt8MoEMethod
from vllm.model_executor.parameter import ModelWeightParameter
from vllm.model_executor.utils import set_weight_attrs

_INT8_MAX = 127.0
_INT8_EPS = 1e-8


_BF16_SKIP_SUBSTRINGS = [
    "mlp.gate",
    "shared_expert_gate",
]


class Int8W8A16Config(QuantizationConfig):
    """W8A16: INT8 weights dequantized to BF16 at runtime, BF16 activations."""

    def __init__(self, from_int8_checkpoint: bool = False) -> None:
        super().__init__()
        self.from_int8_checkpoint = from_int8_checkpoint

    @classmethod
    def get_name(cls) -> QuantizationMethods:
        return "int8_w8a16"

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.bfloat16, torch.float16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 80  # Ampere+; Blackwell SM100 satisfies this

    @classmethod
    def get_config_filenames(cls) -> list[str]:
        return []

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "Int8W8A16Config":
        return cls(from_int8_checkpoint=True)

    @staticmethod
    def _should_skip(prefix: str) -> bool:
        return any(sub in prefix for sub in _BF16_SKIP_SUBSTRINGS)

    def get_quant_method(
        self, layer: Module, prefix: str
    ) -> Optional[QuantizeMethodBase]:
        if isinstance(layer, LinearBase):
            if self._should_skip(prefix):
                return None
            if self.from_int8_checkpoint:
                return Int8W8A16LinearFromCkptMethod(self)
            return Int8W8A16LinearMethod(self)
        if isinstance(layer, FusedMoE):
            if self.from_int8_checkpoint:
                return Int8W8A16MoEFromCkptMethod(self, layer.moe_config)
            return ExpertsInt8MoEMethod(self, layer.moe_config)
        return None


# ── Mode 1: BF16 checkpoint → quantize in process_weights_after_loading ─────

class Int8W8A16LinearMethod(LinearMethodBase):
    """Load BF16, quantize to INT8 in memory, dequantize in apply."""

    def __init__(self, quant_config: Int8W8A16Config):
        self.quant_config = quant_config

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
        weight = ModelWeightParameter(
            data=torch.empty(
                sum(output_partition_sizes),
                input_size_per_partition,
                dtype=params_dtype,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=extra_weight_attrs.get("weight_loader"),
        )
        layer.register_parameter("weight", weight)

    def process_weights_after_loading(self, layer: Module) -> None:
        w = layer.weight.data.float()
        amax = w.abs().amax(dim=1, keepdim=True).clamp(min=_INT8_EPS)
        scale = amax / _INT8_MAX
        w_int8 = (w / scale).round().clamp(-128, 127).to(torch.int8)
        layer.weight = Parameter(w_int8, requires_grad=False)
        layer.weight_scale = Parameter(scale.to(torch.float32), requires_grad=False)

    def apply(self, layer: Module, x: torch.Tensor,
              bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        w = (layer.weight.float() * layer.weight_scale).to(x.dtype)
        return F.linear(x, w, bias)


# ── Mode 2: INT8 checkpoint → load directly, dequantize after loading ────────

class Int8W8A16LinearFromCkptMethod(LinearMethodBase):
    """Load int8 weight + float32 scale directly from checkpoint."""

    def __init__(self, quant_config: Int8W8A16Config):
        self.quant_config = quant_config

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
        weight_loader = extra_weight_attrs.get("weight_loader")

        # Register weight as int8 — the checkpoint contains int8 tensors.
        weight = ModelWeightParameter(
            data=torch.empty(
                sum(output_partition_sizes),
                input_size_per_partition,
                dtype=torch.int8,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight", weight)

        # Scale: [out, 1] float32
        weight_scale = ModelWeightParameter(
            data=torch.ones(
                sum(output_partition_sizes),
                1,
                dtype=torch.float32,
            ),
            input_dim=None,
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight_scale", weight_scale)

    def process_weights_after_loading(self, layer: Module) -> None:
        w = (layer.weight.data.float() * layer.weight_scale.data).to(torch.bfloat16)
        layer.weight = Parameter(w, requires_grad=False)
        del layer.weight_scale

    def apply(self, layer: Module, x: torch.Tensor,
              bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        return F.linear(x, layer.weight.to(x.dtype), bias)


class Int8W8A16MoEFromCkptMethod(FusedMoEMethodBase):
    """Load per-expert int8 weights + float32 scales from INT8 checkpoint.

    expert_params_mapping stacks per-expert tensors into w13_weight [E,2*i,in]
    and w2_weight [E,h,i]. Scales are stacked via FusedMoE's
    _load_per_channel_weight_scale (triggered by quant_method="channel" on
    the scale parameters). Dequantizes to BF16 in process_weights_after_loading.
    """

    def __init__(self, quant_config: Int8W8A16Config, moe_config):
        super().__init__(moe_config)
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
        layer._params_dtype = params_dtype
        weight_loader = extra_weight_attrs["weight_loader"]

        # ── int8 weight tensors ──────────────────────────────────────────────
        w13_weight = Parameter(
            torch.empty(num_experts, 2 * intermediate_size_per_partition,
                        hidden_size, dtype=torch.int8),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)

        w2_weight = Parameter(
            torch.empty(num_experts, hidden_size,
                        intermediate_size_per_partition, dtype=torch.int8),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

        # ── float32 scale tensors ────────────────────────────────────────────
        # quant_method="channel" tells FusedMoE.weight_loader to use
        # _load_per_channel_weight_scale, which correctly stacks per-expert
        # [inter, 1] scales into [E, 2*inter, 1] via _load_w13.
        w13_weight_scale = Parameter(
            torch.ones(num_experts, 2 * intermediate_size_per_partition, 1,
                       dtype=torch.float32),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight_scale", w13_weight_scale)
        set_weight_attrs(w13_weight_scale, {
            "weight_loader": weight_loader,
            "quant_method": "channel",   # triggers _load_per_channel_weight_scale
        })

        w2_weight_scale = Parameter(
            torch.ones(num_experts, hidden_size, 1, dtype=torch.float32),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight_scale", w2_weight_scale)
        set_weight_attrs(w2_weight_scale, {
            "weight_loader": weight_loader,
            "quant_method": "channel",
        })

    def process_weights_after_loading(self, layer: Module) -> None:
        w13 = (layer.w13_weight.data.float()
               * layer.w13_weight_scale.data).to(layer._params_dtype)
        w2 = (layer.w2_weight.data.float()
              * layer.w2_weight_scale.data).to(layer._params_dtype)
        layer.w13_weight = Parameter(w13, requires_grad=False)
        layer.w2_weight = Parameter(w2, requires_grad=False)
        del layer.w13_weight_scale, layer.w2_weight_scale

    def get_fused_moe_quant_config(self, layer: Module):
        return FUSED_MOE_UNQUANTIZED_CONFIG

    def apply(
        self,
        layer: FusedMoE,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        top_k: int,
        renormalize: bool,
        use_grouped_topk: bool = False,
        topk_group: Optional[int] = None,
        num_expert_group: Optional[int] = None,
        global_num_experts: int = -1,
        expert_map: Optional[torch.Tensor] = None,
        custom_routing_function: Optional[Callable] = None,
        scoring_func: str = "softmax",
        routed_scaling_factor: float = 1.0,
        e_score_correction_bias: Optional[torch.Tensor] = None,
        apply_router_weight_on_input: bool = False,
        activation: str = "silu",
        enable_eplb: bool = False,
        expert_load_view: Optional[torch.Tensor] = None,
        logical_to_physical_map: Optional[torch.Tensor] = None,
        logical_replica_count: Optional[torch.Tensor] = None,
    ):
        from vllm.model_executor.layers.fused_moe import fused_experts

        topk_weights, topk_ids, zero_expert_result = layer.select_experts(
            hidden_states=x,
            router_logits=router_logits,
        )
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
        )
        if layer.zero_expert_num != 0 and layer.zero_expert_type is not None:
            return result, zero_expert_result
        return result
