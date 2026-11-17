# Copyright (c) 2026-2027 zh
"""内容：统一完整监督、价值损失与返回递推的编译边界及随机数语义。"""

import torch


def compile_kernel(function, backend: str):
    """编译固定形状的完整函数，保持显式舍入及原生随机数消费。"""
    if backend not in ("inductor", "eager"):
        raise ValueError(f"Unsupported full-loss compile backend: {backend}")
    arguments = {"backend": backend, "fullgraph": True, "dynamic": False}
    if backend == "inductor":
        arguments["options"] = {"emulate_precision_casts": True, "fallback_random": True,
                                "triton.cudagraphs": False}
    return torch.compile(function, **arguments)
