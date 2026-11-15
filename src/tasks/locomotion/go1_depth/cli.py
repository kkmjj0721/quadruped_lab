# Copyright (c) 2026-2027 zh
"""
内容：
    解析独立 Go1 Depth 任务的训练入口，不加载或覆盖 Go2 训练分支。

用法：
    python scripts/train.py Unitree-Go1-Depth-SuperMimic-Rough --max-iterations 5000 --save-interval 5000
"""

import sys

import mjlab
import tyro

from src.config.go1_depth.go1_depth_config import Go1DepthTrainConfig


TASK_MODES = {"Unitree-Go1-Depth-SuperMimic-Rough": "rough", "Unitree-Go1-Depth-SuperMimic-Flat": "flat"}


def train_main(task_id: str, arguments: list[str]) -> None:
    """由统一入口传入明确任务及参数，实际装配留在任务训练模块。"""
    from .rl.training import launch_training

    if task_id not in TASK_MODES:
        raise ValueError(f"Unknown Go1 Depth SuperMimic task: {task_id}")
    args = tyro.cli(Go1DepthTrainConfig, args=arguments, prog=f"{sys.argv[0]} {task_id}", config=mjlab.TYRO_FLAGS)
    launch_training(task_id, args)
