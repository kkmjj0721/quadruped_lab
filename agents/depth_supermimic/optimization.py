# Copyright (c) 2026-2027 zh
"""内容：冻结参数而保留输入导数，统一梯度裁剪及 Adam／Muon 复合优化器的有限性更新。"""

from contextlib import contextmanager

from torch import nn

from .optim import MuonAdam


@contextmanager
def frozen(*modules):
    parameters = [parameter for module in modules for parameter in module.parameters()]
    previous = [parameter.requires_grad for parameter in parameters]
    try:
        for parameter in parameters:
            parameter.requires_grad_(False)
        yield
    finally:
        for parameter, enabled in zip(parameters, previous):
            parameter.requires_grad_(enabled)


def clip_gradients(parameters, norm: float, element: float | None = None):
    parameters = list(parameters)
    value = nn.utils.clip_grad_norm_(parameters, norm)
    if element is not None:
        for parameter in parameters:
            if parameter.grad is not None:
                parameter.grad.clamp_(-element, element)
    return value


def step_if_finite(optimizer, finite):
    """复合优化器统一门控，融合 Adam 使用设备标志，普通 Adam 在更新边界判断。"""
    if isinstance(optimizer, MuonAdam):
        optimizer.step(ok=finite)
    elif optimizer.param_groups[0]["fused"]:
        optimizer.found_inf = (~finite).float()
        try:
            optimizer.step()
        finally:
            optimizer.found_inf = None
    elif bool(finite):
        optimizer.step()
