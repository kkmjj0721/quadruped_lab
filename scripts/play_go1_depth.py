# Copyright (c) 2026-2027 zh
"""
内容：
    在真实Go1仿真中重放检查点的32步想象动作，保存轨迹和逐步误差。

用法：
    .venv/bin/python scripts/play_go1_depth.py --checkpoint logs/supermimic/go1_depth/local/model_final.pt --output logs/supermimic/go1_depth/evaluation
"""

from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.runtime import configure_dependency_warnings

configure_dependency_warnings()

from src.tasks.locomotion.go1_depth.cli import play_main


if __name__ == "__main__":
    play_main()
