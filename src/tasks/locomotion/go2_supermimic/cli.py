# Copyright (c) 2026-2027 zh
"""
内容：
    把 Go2 SuperMimic 训练参数解析从统一入口收回本任务，入口只做任务识别。

用法：
    python scripts/train.py Unitree-Go2-SuperMimic-Flat --device cuda:0 --num-envs 4096
"""

from __future__ import annotations

import sys

import mjlab
import tyro

from .config.train_cfg import SuperMimicTrainConfig
from .rl import training


def train_main(task_id: str, arguments: list[str]) -> None:
    """按本任务的扁平参数解析，随后校验并启动单设备训练。"""
    args = tyro.cli(
        SuperMimicTrainConfig, args=arguments, prog=f"{sys.argv[0]} {task_id}", config=mjlab.TYRO_FLAGS,
    )
    training.launch_training(task_id, args)
