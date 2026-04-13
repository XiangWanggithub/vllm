# SPDX-License-Identifier: Apache-2.0
"""
INT8 W8A16 weight-only quantization.

Loads BF16 weights, quantizes per-channel to INT8 during
process_weights_after_loading, then dequantizes back to BF16 before
each matmul.  All GEMMs run in BF16 — no INT8 GEMM kernels required,
so this works on SM100 (B200/Blackwell) where cutlass_scaled_mm INT8
is not implemented.

For MoE experts, delegates to ExpertsInt8MoEMethod which uses the
triton-based fused_experts kernel with INT8 weights + BF16 activations.

Usage:
    lm_eval --model vllm \
        --model_args "pretrained=...,quantization=int8_w8a16" ...
"""

from typing import Any, Optional

import torch
import torch.nn.functional as F
from torch.nn import Module
from torch.nn.parameter import Parameter

from vllm.model_executor.layers.fused_moe import FusedMoE
from vllm.model_executor.layers.linear import LinearBase, LinearMethodBase
from vllm.model_executor.layers.quantization import QuantizationMethods
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from vllm.model_executor.layers.quantization.experts_int8 import ExpertsInt8MoEMethod
from vllm.model_executor.parameter import ModelWeightParameter

_INT8_MAX = 127.0
_INT8_EPS = 1e-8


class Int8W8A16Config(QuantizationConfig):
    """W8A16: INT8 weights (dequantized to BF16 at runtime), BF16 activations."""

    @classmethod
    def get_name(cls) -> QuantizationMethods:
        return "int8_w8a16"

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.bfloat16, torch.float16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 80  # Ampere+; Blackwell (SM100) is 100 → satisfied

    @classmethod
    def get_config_filenames(cls) -> list[str]:
        return []

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "Int8W8A16Config":
        return cls()

    def get_quant_method(
        self, layer: Module, prefix: str
    ) -> Optional[QuantizeMethodBase]:
        if isinstance(layer, LinearBase):
            return Int8W8A16LinearMethod(self)
        if isinstance(layer, FusedMoE):
            return ExpertsInt8MoEMethod(self, layer.moe_config)
        return None


class Int8W8A16LinearMethod(LinearMethodBase):
    """Per-channel INT8 weight quantization for linear layers.

    Weights are quantized in-memory after loading and dequantized to BF16
    on every forward pass.  No INT8 GEMM kernel is used.
    """

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
        output_size_per_partition = sum(output_partition_sizes)
        layer._params_dtype = params_dtype

        weight = ModelWeightParameter(
            data=torch.empty(
                output_size_per_partition,
                input_size_per_partition,
                dtype=params_dtype,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=extra_weight_attrs.get("weight_loader"),
        )
        layer.register_parameter("weight", weight)

    def process_weights_after_loading(self, layer: Module) -> None:
        w = layer.weight.data.float()                          # [out, in]
        amax = w.abs().amax(dim=1, keepdim=True).clamp(min=_INT8_EPS)
        scale = amax / _INT8_MAX                              # [out, 1]
        w_int8 = (w / scale).round().clamp(-128, 127).to(torch.int8)

        # Store int8 weight + float32 scale to halve memory footprint.
        layer.weight = Parameter(w_int8, requires_grad=False)
        layer.weight_scale = Parameter(scale.to(torch.float32), requires_grad=False)

    def apply(
        self,
        layer: Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Dequantize: [out, in] int8 → BF16
        w = (layer.weight.float() * layer.weight_scale).to(x.dtype)
        return F.linear(x, w, bias)
