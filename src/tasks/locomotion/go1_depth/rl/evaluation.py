# Copyright (c) 2026-2027 zh
"""内容：在固定种子和预算下测量 Go1 真实速度误差、失败和深度输入对照，不推断收敛或通行成功率。"""

import json
from pathlib import Path

import torch

from agents.depth_supermimic.export import load_actor
from agents.supermimic.interfaces import append_history
from src.config.go1_depth.go1_depth_config import Go1DepthTrainConfig

from .sim_adapter import Go1DepthAdapter
from .training import resolve_config


@torch.no_grad()
def evaluate_checkpoint(checkpoint, *, output, device="cuda:0", num_envs=16, steps=1000, seed=1000, depth_mode="normal"):
    """输出真实转移统计；置零和时间倒序深度仅作为显式诊断，不修改训练奖励。"""
    if num_envs < 1 or steps < 1 or seed < 0 or depth_mode not in ("normal", "zero", "reverse"):
        raise ValueError("Invalid evaluation budget, seed or depth mode")
    checkpoint, output = Path(checkpoint).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"Evaluation output already exists: {output}")
    source = json.loads((checkpoint.parent / "task_config.json").read_text(encoding="utf-8"))
    cfg, _, _ = resolve_config(source["task_id"], Go1DepthTrainConfig(resume=checkpoint, num_envs=num_envs, seed=seed), training=False)
    cfg.noise.add_noise = cfg.terrain.curriculum = False
    cfg.terrain.seed = seed
    torch.manual_seed(seed)
    actor, spec, _, iteration = load_actor(checkpoint, device)
    env = Go1DepthAdapter(cfg, device=device)
    try:
        if spec != env.task.spec:
            raise ValueError("Evaluation task dimensions differ from the checkpoint")
        state = env.reset()
        history = env.proprio_frame()[:, None].repeat(1, spec.history_length, 1)
        hidden = torch.zeros(num_envs, spec.latent_dim, device=env.device)
        linear = torch.zeros(2, device=env.device)
        angular = torch.zeros((), device=env.device)
        counts = torch.zeros(2, device=env.device, dtype=torch.long)
        reward_sums = {name: torch.zeros((), device=env.device) for name in cfg.to_dict()["reward"]["scales"]}
        for _ in range(steps):
            depth = env.depth_stack()
            if depth_mode == "zero":
                depth = torch.zeros_like(depth)
            elif depth_mode == "reverse":
                depth = depth.flip(1)
            action, hidden = actor(history, depth, hidden)
            result = env.step(action)
            velocity = env.task.tracking_velocity(result.final_state)
            linear.add_((velocity[:, :2] - state.command[:, :2]).square().sum(0))
            angular.add_((velocity[:, 2] - state.command[:, 2]).square().sum())
            counts.add_(torch.stack((result.terminated.sum(), result.truncated.sum())))
            for name in reward_sums:
                reward_sums[name].add_(env.last_metrics[f"reward/{name}"])
            frame = result.next_observation
            history = append_history(history, frame)
            done = result.terminated | result.truncated
            history = torch.where(done[:, None, None], frame[:, None].expand_as(history), history)
            hidden.masked_fill_(done[:, None], 0)
            state = result.next_state
        result = {"checkpoint": str(checkpoint), "iteration": iteration, "seed": seed, "num_envs": num_envs,
                  "steps": steps, "depth_mode": depth_mode, "velocity_rmse_xy": (linear / (steps * num_envs)).sqrt().cpu().tolist(),
                  "yaw_rate_rmse": float((angular / (steps * num_envs)).sqrt()), "terminated_episodes": int(counts[0]),
                  "truncated_episodes": int(counts[1]), "reward_per_step": {name: float(value / steps) for name, value in reward_sums.items()},
                  "scope": "Fixed-step real-simulation diagnostic; no route-success or multi-seed convergence claim"}
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        return result
    finally:
        env.close()
