# Copyright (c) 2026-2027 zh
"""
内容：
    启动Go1单阶段MGDP感知与SuperMimic训练，想象固定32步。

用法：
    .venv/bin/python scripts/train_go1_depth.py --output logs/supermimic/go1_depth/local --num-envs 16 --verify
"""

from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.runtime import configure_dependency_warnings

configure_dependency_warnings()

from src.tasks.locomotion.go1_depth.cli import train_main


if __name__ == "__main__":
    train_main()
