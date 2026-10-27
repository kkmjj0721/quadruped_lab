# Copyright (c) 2026-2027 zh
"""内容：提供 Go2 单 MLP SuperMimic 训练器装配入口。"""

from .single_runner import make_single_actor_runner


def make_go2_supermimic_runner(cfg=None, *, device="cpu", log_dir=None, execution=None):
    return make_single_actor_runner(cfg, device=device, log_dir=log_dir, execution=execution)
