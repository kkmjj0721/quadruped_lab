# Copyright (c) 2026-2027 zh
"""内容：解析 Go1 深度训练与播放参数，入口只负责装配和分派。"""

import sys

import mjlab
import tyro

from .training import DepthTrainArguments, launch_training


def train_main(task_id, arguments):
    args = tyro.cli(DepthTrainArguments, args=arguments, prog=f"{sys.argv[0]} {task_id}", config=mjlab.TYRO_FLAGS)
    launch_training(task_id, args)
