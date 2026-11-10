# Copyright (c) 2026-2027 zh
"""内容：在设备上累计有限指标，并在日志边界集中读取标量与有效掩码。"""

import math

import torch


class ModelMetrics:
    """保留主线逐更新过滤、按总更新次数平均及全无效键缺省的语义。"""

    def __init__(self):
        self.totals = {}
        self.valid = {}

    def add(self, metrics):
        for name, value in metrics.items():
            if isinstance(value, torch.Tensor):
                value = value.detach()
                finite = torch.isfinite(value).all()
                contribution = torch.where(finite, value, 0)
            else:
                finite = math.isfinite(value)
                contribution = value if finite else 0
            self.totals[name] = self.totals.get(name, 0) + contribution
            self.valid[name] = self.valid.get(name, False) | finite

    def mean(self, updates):
        return {name: value / updates for name, value in self.totals.items()}, self.valid


def read_metrics(metrics, device, *, optional=(), validity=None):
    """一次回传张量；仅过滤学习器原本可省略的键，保留其他指标的异常。"""
    validity = validity if validity is not None else {}
    tensor_names = [name for name, value in metrics.items() if isinstance(value, torch.Tensor)]
    mask_names = list(validity)
    packed = [metrics[name].detach().to(device=device, dtype=torch.float64).reshape(()) for name in tensor_names]
    packed.extend(torch.as_tensor(validity[name], device=device, dtype=torch.float64).reshape(()) for name in mask_names)
    values = torch.stack(packed).cpu().tolist() if packed else []
    result = {name: float(value) for name, value in metrics.items() if name not in tensor_names}
    result.update(zip(tensor_names, values[:len(tensor_names)]))
    for name, valid in zip(mask_names, values[len(tensor_names):]):
        if not valid:
            result.pop(name, None)
    for name in optional:
        if name in result and not math.isfinite(result[name]):
            del result[name]
    return result
