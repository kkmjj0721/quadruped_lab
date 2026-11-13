# Copyright (c) 2026-2027 zh
"""内容：解析Go1单阶段视觉训练、恢复及固定32步轨迹播放参数。"""

import argparse
import json
from pathlib import Path

import torch

from agents.perceptive_supermimic import ExecutionConfig, PerceptiveConfig, PerceptiveRunner
from src.config.go1_depth.go1_depth_locomotion_config import Go1LocomotionConfig
from src.config.go1_depth.go1_depth_supermimic_config import make_algorithm_config, make_execution_config

from .rl.adapter import Go1DepthEnv
from .rl.evaluation import compare_trajectories
from .rl.training import train


def load_configuration(checkpoint):
    """检查点与同次运行的配置文件共同定义恢复任务，不猜测其他目录。"""
    path = Path(checkpoint).resolve()
    data = json.loads((path.parent / "configuration.json").read_text(encoding="utf-8"))
    if "moe_expert_num" in data["algorithm"]:
        raise ValueError("MoE v2 configurations cannot be loaded by the MLP v3 actor; start a new training run")
    return Go1LocomotionConfig(**data["environment"]), PerceptiveConfig(**data["algorithm"]), ExecutionConfig(**data["execution"])


def train_main():
    parser = argparse.ArgumentParser(description="Train Go1 depth SuperMimic with fixed 32-step imagination")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--iterations", type=int, default=10000)
    parser.add_argument("--num-envs", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--terrain", choices=("flat", "vision"))
    parser.add_argument("--baseline", action="store_true", help="Disable height alignment and use the existing depth L1 objective")
    parser.add_argument("--resume")
    parser.add_argument("--verify", action="store_true", help="Run held-out 32-step comparisons, export and resume checks")
    parser.add_argument("--minimum-training-seconds", type=float, default=0.0)
    parser.add_argument("--timeout-seconds", type=float, default=86400.0)
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.resume:
        if args.num_envs is not None or args.seed is not None or args.terrain is not None or args.baseline:
            parser.error("Resume uses its saved task and algorithm; omit task overrides")
        environment, algorithm, execution = load_configuration(args.resume)
    else:
        environment = Go1LocomotionConfig(**{key: getattr(args, key) for key in ("num_envs", "seed", "terrain") if getattr(args, key) is not None})
        algorithm, execution = make_algorithm_config(perception=not args.baseline), make_execution_config()
    train(environment, algorithm, execution, device=args.device, output=args.output, iterations=args.iterations,
          minimum_training_seconds=args.minimum_training_seconds, timeout_seconds=args.timeout_seconds,
          checkpoint=args.resume, verify=args.verify)


def play_main():
    parser = argparse.ArgumentParser(description="Replay Go1 checkpoint in real simulation with a fixed 32-step model comparison")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batches", type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(1)
    environment, algorithm, execution = load_configuration(args.checkpoint)
    if algorithm.horizon != 32:
        parser.error("Go1 evaluation requires a 32-step checkpoint")
    torch.manual_seed(environment.seed)
    env = Go1DepthEnv(environment, device=args.device, latent_dim=algorithm.latent_dim)
    try:
        runner = PerceptiveRunner(env, algorithm, execution=execution)
        runner.load(args.checkpoint)
        result = compare_trajectories(env, runner.actor, runner.model, args.output, batches=args.batches)
        print(json.dumps(result, allow_nan=False), flush=True)
    finally:
        env.close()
