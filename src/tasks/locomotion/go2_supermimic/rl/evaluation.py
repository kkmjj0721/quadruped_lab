# Copyright (c) 2026-2027 zh
"""
内容：
    分命令评估学生、教师与零动作，记录真实终态跟踪、首次回合存活和部署对拍。
"""

from __future__ import annotations

import math
from pathlib import Path
import re

import torch

from agents.supermimic.interfaces import PolicyInput

from ..mdp.state_spec import ANGULAR_VELOCITY, LINEAR_VELOCITY, QUATERNION, quat_apply_inverse
from .checkpoints import build_playback_policy, load_playback_source
from .playback import make_playback
from ..config.loading import is_single_actor_config


def _finite_metrics(value):
    """检查嵌套结果中的有限数值，允许没有部署文件时的空对拍结果。"""
    if isinstance(value, dict):
        return all(_finite_metrics(item) for item in value.values())
    if isinstance(value, list):
        return all(_finite_metrics(item) for item in value)
    return math.isfinite(value) if isinstance(value, (int, float)) else True


@torch.no_grad()
def evaluate_policy(task_id: str, checkpoint: str | Path, *, branch="student", device="cpu", num_envs=32,
                    seconds=20.0) -> dict:
    """固定四组命令；三分支均使用检查点配置、相同初始种子及播放历史维护。"""
    if branch not in ("actor", "student", "teacher", "zero"):
        raise ValueError("Evaluation branch must be actor, student, teacher or zero")
    if isinstance(num_envs, bool) or not isinstance(num_envs, int) or num_envs < 4:
        raise ValueError("Evaluation requires at least four environments for four command groups")
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("Evaluation duration must be positive and finite")
    checkpoint = Path(checkpoint).expanduser().resolve()
    env, student = make_playback(task_id, checkpoint_file=checkpoint, num_envs=num_envs, device=device)
    try:
        if is_single_actor_config(env.adapter.cfg):
            if branch == "teacher":
                raise ValueError("Single-actor checkpoints do not contain a teacher policy")
            if branch == "student":
                branch = "actor"
        teacher = (build_playback_policy(load_playback_source(task_id, checkpoint), env.adapter)
                   if branch == "teacher" else None)
        device = env.adapter.device
        commands = torch.tensor(((.5, 0., 0.), (-.5, 0., 0.), (0., .5, 0.), (0., 0., .5)), device=device)
        command_names = ("forward", "backward", "lateral", "yaw")
        groups = torch.arange(num_envs, device=device) % 4
        fixed_commands = commands[groups]
        env.adapter.set_play_commands(fixed_commands)
        # 教师构造也会消耗随机数；在所有组件就绪后统一重置物理初态。
        torch.manual_seed(env.adapter.cfg.env.seed)
        observations = env.reset()
        count = math.ceil(seconds / env.adapter.step_dt)
        version = re.fullmatch(r"model_(\d+)\.pt", checkpoint.name)
        exported = checkpoint.parent / f"policy_{int(version.group(1))}.pt" if version is not None else None
        parity = None
        if exported is not None and exported.is_file():
            deployed = torch.jit.load(str(exported), map_location="cpu")
            expected = student(observations).cpu()
            actual = deployed(observations["history"].flatten(1).cpu())
            torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4)
            parity = float((actual - expected).abs().max())
        first_steps = torch.full((num_envs,), count, device=device, dtype=torch.long)
        first_done = torch.zeros(num_envs, device=device, dtype=torch.bool)
        first_failed = first_done.clone()
        velocity_sum = torch.zeros(num_envs, 3, device=device, dtype=torch.float64)
        linear_square, yaw_square, action_square, height_sum, reward_sum = (
            torch.zeros(num_envs, device=device, dtype=torch.float64) for _ in range(5))
        failure_counts = torch.zeros(num_envs, device=device, dtype=torch.long)
        timeout_counts = failure_counts.clone()
        action_max = 0.0
        for step in range(count):
            if teacher is not None:
                state = env.adapter.get_state()
                data = PolicyInput(state, observations["observation"], observations["history"],
                                   env.adapter.task.privileged(state), torch.ones(num_envs, device=device, dtype=torch.bool))
                actions = teacher.act(data)
            elif branch == "zero":
                actions = torch.zeros(num_envs, env.adapter.task.spec.action_dim, device=device)
            else:
                actions = student(observations)
            if not torch.isfinite(actions).all():
                raise FloatingPointError("Evaluation policy produced non-finite actions")
            observations, reward, terminated, truncated, info = env.step(actions)
            final = info["final_state"]
            velocity = quat_apply_inverse(final.physical[:, QUATERNION], final.physical[:, LINEAR_VELOCITY])
            measured = torch.cat((velocity[:, :2], final.physical[:, ANGULAR_VELOCITY][:, 2:3]), dim=-1)
            velocity_sum += measured.double()
            linear_square += (measured[:, :2] - fixed_commands[:, :2]).double().square().sum(-1)
            yaw_square += (measured[:, 2] - fixed_commands[:, 2]).double().square()
            action_square += actions.double().square().mean(-1)
            height_sum += env.adapter.task.physical_features(final)[:, 0].double()
            reward_sum += reward.double()
            action_max = max(action_max, float(actions.abs().max()))
            done = terminated | truncated
            ended = done & ~first_done
            first_steps[ended] = step + 1
            first_failed[ended] = terminated[ended]
            first_done |= done
            failure_counts += terminated.long()
            timeout_counts += (truncated & ~terminated).long()

        def summarize(rows):
            mean_velocity = velocity_sum[rows].mean(0) / count
            return {
                "num_envs": int(rows.sum()), "mean_vx_mps": float(mean_velocity[0]),
                "mean_vy_mps": float(mean_velocity[1]), "mean_yaw_rate_radps": float(mean_velocity[2]),
                "linear_velocity_rmse_mps": math.sqrt(float(linear_square[rows].mean()) / count),
                "yaw_rate_rmse_radps": math.sqrt(float(yaw_square[rows].mean()) / count),
                "first_episode_mean_seconds": float(first_steps[rows].double().mean()) * env.adapter.step_dt,
                "first_episode_failure_fraction": float(first_failed[rows].double().mean()),
                "first_episode_censored_fraction": float((~first_done[rows]).double().mean()),
                "failures": int(failure_counts[rows].sum()), "timeouts": int(timeout_counts[rows].sum()),
                "mean_reward_per_step": float(reward_sum[rows].mean()) / count,
                "mean_return_per_environment": float(reward_sum[rows].mean()),
                "mean_base_height_m": float(height_sum[rows].mean()) / count,
                "action_rms": math.sqrt(float(action_square[rows].mean()) / count),
            }

        result = {
            "checkpoint": str(checkpoint), "task_id": task_id, "branch": branch, "device": str(device),
            "seed": env.adapter.cfg.env.seed, "control_steps": count, "commands_body": commands.tolist(),
            "duration_seconds": count * env.adapter.step_dt, "zero_action": branch == "zero",
            **summarize(torch.ones(num_envs, device=device, dtype=torch.bool)),
            "per_command": {name: {"command_body": commands[index].tolist(), **summarize(groups == index)}
                            for index, name in enumerate(command_names)},
            "falls_per_environment_minute": int(failure_counts.sum()) / (num_envs * count * env.adapter.step_dt / 60),
            "action_abs_max": action_max, "command_rms": float(fixed_commands.square().sum(-1).mean().sqrt()),
            "torchscript_max_abs_error": parity,
        }
        if not _finite_metrics(result):
            raise FloatingPointError("Evaluation produced non-finite metrics")
        return result
    finally:
        env.close()


def evaluate_student(task_id: str, checkpoint: str | Path, *, device="cpu", num_envs=32, seconds=20.0,
                     zero_action=False, branch="student") -> dict:
    """兼容原学生评估入口；新调用方通过branch明确选择学生、教师或零动作。"""
    if zero_action:
        if branch != "student":
            raise ValueError("zero_action cannot be combined with an explicit nonstudent branch")
        branch = "zero"
    return evaluate_policy(task_id, checkpoint, branch=branch, device=device, num_envs=num_envs, seconds=seconds)
