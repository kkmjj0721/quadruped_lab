# Copyright (c) 2026-2027 zh
"""内容：解析 Go1 MGDP 两阶段训练、同阶段恢复与固定32步播放参数。"""

import argparse
from dataclasses import replace
import json
from pathlib import Path

import torch

from agents.perceptive_supermimic import ExecutionConfig, PerceptiveConfig, PerceptiveRunner
from src.config.go1_depth.go1_depth_config import (
    Go1LocomotionConfig, Go1MGDPConfig, make_algorithm_config, make_mgdp_algorithm_config, make_execution_config,
)

from .rl.adapter import Go1DepthEnv
from .rl.evaluation import compare_trajectories
from .rl.training import train


def load_configuration(checkpoint):
    """检查点与同次运行的配置共同定义任务；旧算法格式由加载边界明确拒绝。"""
    path = Path(checkpoint).resolve()
    data = json.loads((path.parent / "configuration.json").read_text(encoding="utf-8"))
    if "moe_expert_num" in data["algorithm"]:
        raise ValueError("MoE configurations are incompatible with the visual-clock v4 actor")
    if data["algorithm"].get("perception_model") == "mgdp":
        environment = Go1MGDPConfig(**data["environment"])
    else:
        environment = Go1LocomotionConfig(**data["environment"])
    return environment, PerceptiveConfig(**data["algorithm"]), ExecutionConfig(**data["execution"])


def train_main():
    parser = argparse.ArgumentParser(description="Train Go1 MGDP stages with fixed 32-step SuperMimic imagination")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--iterations", type=int, default=10000)
    parser.add_argument("--num-envs", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--stage", type=int, choices=(1, 2))
    parser.add_argument("--stage1-run", help="Stage-1 run containing model_final.pt and perception_best.pt")
    parser.add_argument("--legacy", action="store_true", help="Use the previous local single-frame recipe")
    parser.add_argument("--terrain", choices=("flat", "vision"), help="Legacy recipe only")
    parser.add_argument("--baseline", action="store_true", help="Legacy depth-L1 recipe without height alignment")
    parser.add_argument("--resume")
    parser.add_argument("--verify", action="store_true", help="Run held-out 32-step comparisons, export and resume checks")
    parser.add_argument("--minimum-training-seconds", type=float, default=0.0)
    parser.add_argument("--timeout-seconds", type=float, default=86400.0)
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.resume:
        if any(value is not None for value in (args.num_envs, args.seed, args.terrain, args.stage, args.stage1_run)) or args.baseline or args.legacy:
            parser.error("Resume uses its saved stage, task and algorithm; omit task overrides")
        environment, algorithm, execution = load_configuration(args.resume)
    elif args.legacy:
        if args.stage is not None or args.stage1_run:
            parser.error("Legacy training has no MGDP stages")
        environment = Go1LocomotionConfig(**{key: getattr(args, key) for key in ("num_envs", "seed", "terrain") if getattr(args, key) is not None})
        algorithm, execution = make_algorithm_config(perception=not args.baseline), make_execution_config()
    else:
        if args.terrain is not None or args.baseline:
            parser.error("MGDP stages define terrain and perception objectives; terrain/baseline require --legacy")
        if args.stage1_run:
            if args.stage != 2 or args.num_envs is not None or args.seed is not None:
                parser.error("Stage transfer requires --stage 2 and inherits environment settings from stage 1")
            environment, algorithm, execution = load_configuration(Path(args.stage1_run) / "model_final.pt")
            if algorithm.perception_model != "mgdp" or environment.stage != 1:
                parser.error("Stage transfer source must be an MGDP stage-1 run")
            environment = replace(environment, stage=2)
        else:
            if args.stage == 2:
                parser.error("Stage 2 requires --stage1-run or a same-stage --resume")
            environment = Go1MGDPConfig(**{key: getattr(args, key) for key in ("num_envs", "seed", "stage") if getattr(args, key) is not None})
            algorithm, execution = make_mgdp_algorithm_config(), make_execution_config()
    train(environment, algorithm, execution, device=args.device, output=args.output, iterations=args.iterations,
          minimum_training_seconds=args.minimum_training_seconds, timeout_seconds=args.timeout_seconds,
          checkpoint=args.resume, stage1_run=args.stage1_run, verify=args.verify)


def play_main():
    parser = argparse.ArgumentParser(description="Replay Go1 checkpoint with a fixed 32-step model comparison")
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
    if algorithm.perception_model == "mgdp":
        from .rl.mgdp_adapter import MGDPGo1DepthEnv
        env = MGDPGo1DepthEnv(environment, device=args.device, latent_dim=algorithm.latent_dim)
    else:
        env = Go1DepthEnv(environment, device=args.device, latent_dim=algorithm.latent_dim)
    try:
        runner = PerceptiveRunner(env, algorithm, execution=execution)
        runner.load(args.checkpoint)
        result = compare_trajectories(env, runner.actor, runner.model, args.output, batches=args.batches)
        print(json.dumps(result, allow_nan=False), flush=True)
    finally:
        env.close()
