# Copyright (c) 2026-2027 zh
"""
内容：
    按显式预算运行 Go1 Depth 的真实速度、失败和视觉输入诊断。

用法：
    python scripts/evaluate.py --checkpoint logs/depth_supermimic/go1_depth/运行目录/model_5000.pt --output outputs/go1_depth_eval.json
"""

import argparse
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.tasks.locomotion.go1_depth.rl.evaluation import evaluate_checkpoint


def main():
    parser = argparse.ArgumentParser(description="Evaluate a Go1 depth policy in real simulation")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--depth-mode", choices=("normal", "zero", "reverse"), default="normal")
    args = parser.parse_args()
    evaluate_checkpoint(args.checkpoint, output=args.output, device=args.device, num_envs=args.num_envs,
                        steps=args.steps, seed=args.seed, depth_mode=args.depth_mode)
    print(f"[INFO] Evaluation saved to {args.output}", flush=True)


if __name__ == "__main__":
    main()
