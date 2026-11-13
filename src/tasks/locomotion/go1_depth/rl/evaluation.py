# Copyright (c) 2026-2027 zh
"""内容：从同一真实起点生成固定32步想象动作，再用真实Go1重放并保存逐步误差。"""

import json
from pathlib import Path

import numpy as np
import torch

from agents.perceptive_supermimic.interfaces import append_history
from agents.perceptive_supermimic.interfaces import TaskState
from src.config.go1_depth.go1_depth_supermimic_config import IMAGINATION_HORIZON


@torch.no_grad()
def compare_trajectories(env, actor, model, directory, *, batches: int = 4) -> dict:
    """模型轨迹全程不读取未来真值；终止当步有效，之后不拼接新回合。"""
    if isinstance(batches, bool) or not isinstance(batches, int) or batches < 1:
        raise ValueError("Evaluation batches must be a positive integer")
    output = Path(directory)
    output.mkdir(parents=True, exist_ok=True)
    task, horizon = env.task, IMAGINATION_HORIZON
    records = {key: [] for key in ("real_states", "model_states", "actions", "valid", "terminated", "truncated",
                                   "position_error", "velocity_error", "joint_error", "rotation_error", "latent_error",
                                   "tracking_error", "rewards")}
    initial_data = {key: [] for key in ("initial_memory", "initial_command", "initial_context", "initial_depth", "target_depths")}
    for _ in range(batches):
        decision = env.reset()
        state = decision.state.detached_clone()
        history = decision.observation.proprio[:, None].repeat(1, task.spec.history_length, 1)
        latent = actor.encoder(decision.observation.depth)
        for name in ("memory", "command", "context"):
            initial_data[f"initial_{name}"].append(getattr(state, name).cpu().numpy().copy())
        initial_data["initial_depth"].append(decision.observation.depth.cpu().numpy().copy())
        imagined_states = [state.physical.clone()]
        imagined_latents, actions = [], []
        for _ in range(horizon):
            action = actor.act_latent(history, latent)
            prediction = model(state, latent, action)
            state, latent = prediction.state, prediction.latent
            history = append_history(history, task.observe(state, noisy=True))
            imagined_states.append(state.physical.clone())
            imagined_latents.append(latent.clone())
            actions.append(action.clone())
        active = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
        real_states = [decision.state.physical.clone()]
        values = {key: [] for key in records if key not in ("real_states", "model_states", "actions")}
        target_depths = []
        for step, action in enumerate(actions):
            result = env.step(action)
            actual = result.final.state
            # 同一TaskState布局，仅替换物理量用于误差读取；指令和上下文沿用采集初态。
            predicted_state = decision.state.detached_clone()
            predicted_state.physical = imagined_states[step + 1]
            actual_latent = actor.encoder(result.final.observation.depth)
            target_depths.append(result.final.observation.depth.cpu().numpy().copy())
            differences = {
                "position_error": (task.world_position(predicted_state) - task.world_position(actual)).norm(dim=-1),
                "velocity_error": (task.body_linear_velocity(predicted_state) - task.body_linear_velocity(actual)).square().mean(-1).sqrt(),
                "joint_error": (task.joint_position(predicted_state) - task.joint_position(actual)).square().mean(-1).sqrt(),
                "rotation_error": task.rotation_angle_error(predicted_state, actual),
                "latent_error": ((imagined_latents[step] - actual_latent) / model.z_std).square().mean(-1).sqrt(),
                "tracking_error": (task.body_linear_velocity(actual)[:, :2] - decision.state.command[:, :2]).square().mean(-1).sqrt(),
                "rewards": result.reward,
            }
            real_states.append(actual.physical.clone())
            values["valid"].append(active.clone())
            values["terminated"].append(result.terminated & active)
            values["truncated"].append(result.truncated & active)
            for name, value in differences.items():
                if not bool(torch.isfinite(value[active]).all()):
                    raise FloatingPointError(f"Non-finite trajectory metric: {name}")
                values[name].append(value.clone())
            active &= ~(result.terminated | result.truncated)
        records["real_states"].append(torch.stack(real_states, 1).cpu().numpy())
        records["model_states"].append(torch.stack(imagined_states, 1).cpu().numpy())
        records["actions"].append(torch.stack(actions, 1).cpu().numpy())
        for name, value in values.items():
            records[name].append(torch.stack(value, 1).cpu().numpy())
        initial_data["target_depths"].append(np.stack(target_depths, axis=1))
    arrays = {name: np.concatenate(value, axis=0) for name, value in records.items()}
    arrays.update({name: np.concatenate(value, axis=0) for name, value in initial_data.items()})
    valid = arrays["valid"]
    counts = valid.sum(0)
    metric_names = ("position_error", "velocity_error", "joint_error", "rotation_error", "latent_error", "tracking_error", "rewards")
    curves = {name: [float(arrays[name][valid[:, step], step].mean()) if counts[step] else None
                     for step in range(horizon)] for name in metric_names}
    summary = {
        "horizon": horizon, "control_dt": task.dt, "trajectories": int(valid.shape[0]),
        "valid_samples_per_step": counts.tolist(),
        "complete_32_step_trajectories": int((valid[:, -1] & ~arrays["terminated"][:, -1] & ~arrays["truncated"][:, -1]).sum()),
        "terminal_step_included": True, "action_source": "closed_loop_imagined_policy_then_identical_real_action_replay",
        "future_real_states_used_by_model": False,
        "failure_fraction": float(arrays["terminated"].any(1).mean()),
        "timeout_fraction": float(arrays["truncated"].any(1).mean()),
        "curves": curves,
        "final_step": {name: value[-1] for name, value in curves.items()},
        "mean_tracking_rmse": float(arrays["tracking_error"][valid].mean()),
        "mean_real_reward": float(arrays["rewards"][valid].mean()),
    }
    np.savez_compressed(output / "trajectories.npz", **arrays)
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return summary


@torch.no_grad()
def replay_heldout_trajectories(task, actor, model, source, directory) -> dict:
    """用训练前保留且未放入训练回放的同一组真实轨迹复核模型，固定32步和记录动作。"""
    output = Path(directory)
    output.mkdir(parents=True, exist_ok=True)
    with np.load(source) as loaded:
        arrays = {name: loaded[name] for name in loaded.files}
    device = next(actor.parameters()).device
    tensor = lambda name: torch.as_tensor(arrays[name], device=device)
    state = TaskState(tensor("real_states")[:, 0], tensor("initial_memory"), tensor("initial_command"), tensor("initial_context"))
    latent = actor.encoder(tensor("initial_depth"))
    actions, real, depths = tensor("actions"), tensor("real_states"), tensor("target_depths")
    valid = arrays["valid"]
    counts = valid.sum(0)
    metrics = {name: [] for name in ("position_error", "velocity_error", "joint_error", "rotation_error", "latent_error")}
    model_states = [state.physical.clone()]
    for step in range(IMAGINATION_HORIZON):
        predicted = model(state, latent, actions[:, step])
        state, latent = predicted.state, predicted.latent
        model_states.append(state.physical.clone())
        actual = TaskState(real[:, step + 1], actions[:, step], tensor("initial_command"), tensor("initial_context"))
        target_latent = actor.encoder(depths[:, step])
        errors = {
            "position_error": (task.world_position(state) - task.world_position(actual)).norm(dim=-1),
            "velocity_error": (task.body_linear_velocity(state) - task.body_linear_velocity(actual)).square().mean(-1).sqrt(),
            "joint_error": (task.joint_position(state) - task.joint_position(actual)).square().mean(-1).sqrt(),
            "rotation_error": task.rotation_angle_error(state, actual),
            "latent_error": ((latent - target_latent) / model.z_std).square().mean(-1).sqrt(),
        }
        for name, value in errors.items():
            selected = value[torch.as_tensor(valid[:, step], device=device)]
            if not bool(torch.isfinite(selected).all()):
                raise FloatingPointError(f"Non-finite held-out trajectory metric: {name}")
            metrics[name].append(float(selected.mean()) if selected.numel() else None)
    summary = {
        "horizon": IMAGINATION_HORIZON, "control_dt": task.dt, "trajectories": int(valid.shape[0]),
        "valid_samples_per_step": counts.tolist(),
        "complete_32_step_trajectories": int((valid[:, -1] & ~arrays["terminated"][:, -1] & ~arrays["truncated"][:, -1]).sum()),
        "identical_recorded_initial_states_and_actions": True, "training_replay_contains_this_dataset": False,
        "future_real_states_used_by_model": False, "curves": metrics,
        "final_step": {name: values[-1] for name, values in metrics.items()},
    }
    np.savez_compressed(output / "model_states.npz", model_states=torch.stack(model_states, 1).cpu().numpy(), valid=valid)
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return summary


def plot_trajectory_comparison(before: dict, after: dict, path) -> None:
    """只画训练结束后已有32步评估产物，不读取运行中的训练日志。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    font = font_manager.FontProperties(family="Noto Sans CJK JP")
    plt.rcParams["axes.unicode_minus"] = False
    figure, axes = plt.subplots(2, 3, figsize=(13, 7), constrained_layout=True)
    keys = ("position_error", "velocity_error", "joint_error", "rotation_error", "latent_error")
    labels = ("位置误差（米）", "机身速度均方根误差（米／秒）", "关节角均方根误差（弧度）", "姿态角误差（弧度）", "归一化视觉特征误差")
    steps = np.arange(1, IMAGINATION_HORIZON + 1)
    for axis, key, label in zip(axes.flat, keys, labels):
        axis.plot(steps, before["curves"][key], label="训练前")
        axis.plot(steps, after["curves"][key], label="训练后")
        axis.set_title(label, fontproperties=font)
        axis.set_xlabel("想象步数", fontproperties=font)
        axis.grid(alpha=0.25)
        axis.legend(prop=font)
    axis = axes.flat[-1]
    axis.plot(steps, before["valid_samples_per_step"], label="训练前")
    axis.plot(steps, after["valid_samples_per_step"], label="训练后")
    axis.set_title("每一步有效轨迹数", fontproperties=font)
    axis.set_xlabel("想象步数", fontproperties=font)
    axis.legend(prop=font)
    axis.grid(alpha=0.25)
    figure.suptitle("Go1留出轨迹32步对照：相同真实起点与动作，终止后排除", fontproperties=font)
    figure.savefig(path, dpi=150)
    plt.close(figure)
