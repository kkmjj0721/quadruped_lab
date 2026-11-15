# Copyright (c) 2026-2027 zh
"""
内容：
    从 Go1 Depth 检查点导出编码器与策略，以及对应的历史、相机和动作元数据。

用法：
    python scripts/export.py --checkpoint logs/depth_supermimic/go1_depth/运行目录/model_5000.pt --output outputs/go1_depth/policy.pt --onnx
"""

import argparse
import json
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agents.depth_supermimic.export import export_policy, load_actor


def main():
    parser = argparse.ArgumentParser(description="Export a Go1 depth encoder and policy")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--onnx", action="store_true")
    args = parser.parse_args()
    actor, spec, _, iteration = load_actor(args.checkpoint)
    snapshot = json.loads((args.checkpoint.parent / "task_config.json").read_text(encoding="utf-8"))
    export_policy(actor, args.output, spec=spec, iteration=iteration, metadata=snapshot, onnx=args.onnx)
    print(f"[INFO] Policy exported to {args.output}", flush=True)


if __name__ == "__main__":
    main()
