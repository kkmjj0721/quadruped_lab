# Copyright (c) 2026-2027 zh
"""
内容：
    从SuperMimic检查点显式导出部署策略，不启动仿真或继续训练。

用法：
    uv run --no-sync python -B tools/export_supermimic_policy.py '<实验目录>/model_1000.pt' --output-dir .artifacts/deploy/go2
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

# 脚本从tools/启动时显式加入仓库根目录，使agents与src按项目内源码解析。
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.tasks.locomotion.go2_supermimic.rl.deployment import export_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path, help="SuperMimic model checkpoint such as model_1000.pt")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory receiving policy_<N>.{pt,onnx,json}")
    arguments = parser.parse_args()
    paths = export_checkpoint(arguments.checkpoint, arguments.output_dir)
    print(f"[INFO] Exported deployment policy: {Path(paths['onnx']).parent}", flush=True)


if __name__ == "__main__":
    main()
