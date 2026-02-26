# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Standalone Hadamard rotation module for MoE layers.

Applies group-wise Hadamard transforms to redistribute activation values
more uniformly before quantization, reducing quantization error. The
rotation is orthogonal (H^T H = I), so rotating both weights and
activations cancels out: (x @ H) @ (W @ H)^T = x @ W^T.

This module has no quantization dependencies and can be reused for
FP8, MXFP4, INT8, etc.
"""
from dataclasses import dataclass

import torch

from vllm import _custom_ops as ops
from vllm.logger import init_logger

logger = init_logger(__name__)


def _largest_pow2_divisor(n: int, cap: int = 64) -> int:
    """Return the largest power-of-2 that divides n, capped at cap."""
    d = n & -n  # isolate lowest set bit
    return min(d, cap)


@dataclass
class HadamardRotationConfig:
    """Configuration for Hadamard rotation in MoE layers."""
    enabled: bool = False
    group_size: int = 64  # must be power-of-2
    rotate_w2: bool = False  # rotate second MoE matmul too
    w2_group_size: int = 0  # auto-set from w2 dim if 0


def hadamard_rotate(x: torch.Tensor, group_size: int) -> torch.Tensor:
    """Group-wise Hadamard rotation on the last dimension.

    Reshapes (..., D) -> (..., D//group_size, group_size),
    applies normalized Walsh-Hadamard transform per group, reshapes back.

    The hadacore kernel computes the normalized (orthogonal, self-inverse)
    WHT, so H^2 = I and norms are preserved.

    Note: hadacore_transform only supports fp16/bf16. The input must be
    in one of these dtypes.

    Args:
        x: Input tensor of shape (..., D) where D is divisible by
           group_size. Must be fp16 or bf16.
        group_size: Size of each group (must be power-of-2).

    Returns:
        Rotated tensor of the same shape and dtype.
    """
    D = x.shape[-1]
    assert D % group_size == 0, (
        f"Dimension {D} not divisible by group_size {group_size}"
    )
    original_shape = x.shape
    # Reshape to (..., D//group_size, group_size)
    x = x.unflatten(-1, (-1, group_size)).contiguous()
    # hadacore_transform returns the transformed tensor. Use inplace=True
    # for the correct kernel path (inplace=False returns garbage).
    # The kernel does NOT actually modify x in-place despite the flag name,
    # so we must capture the return value.
    x = ops.hadacore_transform(x, inplace=True)
    # Reshape back to original shape
    x = x.flatten(-2, -1)
    assert x.shape == original_shape
    return x


def rotate_moe_weights(
    w13: torch.Tensor,
    w2: torch.Tensor,
    config: HadamardRotationConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Offline weight rotation. Called once at model load time.

    For the first matmul: y1 = x @ W13^T
    We want: (x @ H) @ (W13 @ H)^T = x @ H @ H^T @ W13^T = x @ W13^T
    So we rotate W13 along its last dim (the hidden/input dim).

    w13 shape: (E, 2*intermediate, hidden) — rotate along hidden dim (dim=-1)
    w2 shape:  (E, hidden, intermediate) — optionally rotate along
               intermediate dim (dim=-1)

    Args:
        w13: First MoE weight tensor of shape (E, 2*inter, hidden).
        w2: Second MoE weight tensor of shape (E, hidden, inter).
        config: Hadamard rotation configuration.

    Returns:
        Tuple of rotated (w13, w2) tensors.
    """
    group_size = config.group_size
    assert group_size > 0 and (group_size & (group_size - 1)) == 0, (
        f"group_size must be power-of-2, got {group_size}"
    )

    # hadacore only supports fp16/bf16, so use bf16 for weight rotation
    orig_dtype_w13 = w13.dtype
    hidden_dim = w13.shape[-1]
    assert hidden_dim % group_size == 0, (
        f"w13 hidden dim {hidden_dim} not divisible by group_size {group_size}"
    )
    logger.info(
        "Applying Hadamard rotation to w13 weights "
        "(shape=%s, group_size=%d)", w13.shape, group_size
    )
    w13 = hadamard_rotate(w13.to(torch.bfloat16), group_size).to(orig_dtype_w13)

    if config.rotate_w2:
        orig_dtype_w2 = w2.dtype
        inter_dim = w2.shape[-1]
        w2_gs = config.w2_group_size if config.w2_group_size > 0 \
            else _largest_pow2_divisor(inter_dim, group_size)
        assert inter_dim % w2_gs == 0, (
            f"w2 intermediate dim {inter_dim} not divisible by "
            f"w2_group_size {w2_gs}"
        )
        config.w2_group_size = w2_gs
        logger.info(
            "Applying Hadamard rotation to w2 weights "
            "(shape=%s, group_size=%d)", w2.shape, w2_gs
        )
        w2 = hadamard_rotate(w2.to(torch.bfloat16), w2_gs).to(orig_dtype_w2)

    return w13, w2


def get_hadamard_config_from_env() -> HadamardRotationConfig:
    """Read Hadamard rotation config from environment variables."""
    import vllm.envs as envs
    return HadamardRotationConfig(
        enabled=envs.VLLM_FP8_HADAMARD_ROTATION,
        group_size=envs.VLLM_FP8_HADAMARD_GROUP_SIZE,
        rotate_w2=envs.VLLM_FP8_HADAMARD_ROTATE_W2,
    )
