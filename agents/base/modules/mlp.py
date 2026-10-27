# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026-2027 zh
"""
内容：
    提供前馈策略和价值网络使用的多层感知机。
"""

from __future__ import annotations

import math

import torch
from torch import nn


def _activation(name: str) -> nn.Module:
    """将来源实现使用的激活名称映射为独立模块。"""
    activations = {
        "elu": nn.ELU, "selu": nn.SELU, "relu": nn.ReLU, "crelu": nn.CELU,
        "lrelu": nn.LeakyReLU, "tanh": nn.Tanh, "sigmoid": nn.Sigmoid,
        "softplus": nn.Softplus, "gelu": nn.GELU, "swish": nn.SiLU,
        "mish": nn.Mish, "identity": nn.Identity,
    }
    if name not in activations:
        raise ValueError(f"Unknown activation: {name}.")
    return activations[name]()


class MLP(nn.Sequential):
    """由线性层和激活构成的网络，末层默认不加激活。"""

    def __init__(
        self,
        input_dim: int,
        output_dim: int | tuple[int, ...] | list[int],
        hidden_dims: tuple[int, ...] | list[int],
        activation: str = "elu",
        last_activation: str | None = None,
    ) -> None:
        if input_dim <= 0:
            raise ValueError("input_dim must be positive.")
        hidden_dims = [input_dim if dim == -1 else dim for dim in hidden_dims]
        output_shape = None if isinstance(output_dim, int) else tuple(output_dim)
        final_dim = output_dim if output_shape is None else math.prod(output_shape)
        if any(dim <= 0 for dim in hidden_dims) or final_dim <= 0:
            raise ValueError("MLP dimensions must be positive.")
        dims = [input_dim, *hidden_dims, final_dim]
        layers = []
        for index in range(len(dims) - 1):
            layers.append(nn.Linear(dims[index], dims[index + 1]))
            if index < len(dims) - 2:
                layers.append(_activation(activation))
        if output_shape is not None:
            layers.append(nn.Unflatten(dim=-1, unflattened_size=output_shape))
        if last_activation is not None:
            layers.append(_activation(last_activation))
        super().__init__(*layers)

    def init_weights(self, scales: float | tuple[float, ...]) -> None:
        """按线性层次序应用正交初始化，偏置设为零。"""
        linear_index = 0
        for module in self:
            if isinstance(module, nn.Linear):
                gain = scales if isinstance(scales, (float, int)) else scales[linear_index]
                nn.init.orthogonal_(module.weight, gain=gain)
                nn.init.zeros_(module.bias)
                linear_index += 1
