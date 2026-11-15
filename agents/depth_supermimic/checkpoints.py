# Copyright (c) 2026-2027 zh
"""内容：保存视觉模型、优化器、统计与训练计数；恢复后按参考重新填充真实回放。"""

from dataclasses import asdict
from pathlib import Path

import torch

from agents.supermimic.optim import MuonAdam

from .config import DepthSuperMimicConfig


REFERENCE = "InstinctMJ@03563ee02665f327425106372264070315673748;MGDP@f43139d1d97849819a0dce609dc0b59faf577b3f"
FORMAT = "depth_supermimic_mgdp_gru_v3"


def save_checkpoint(runner, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {"format": FORMAT, "reference": REFERENCE, "spec": asdict(runner.spec), "config": asdict(runner.config),
            "task_configuration": runner.task.semantic_configuration(), "num_envs": runner.env.num_envs,
            "execution": asdict(runner.execution), "iteration": runner.iteration, "real_steps": runner.real_steps,
            "prefill_real_steps": runner.prefill_real_steps, "elapsed": runner.elapsed,
            "training_seconds": runner.training_seconds,
            "nonfinite_steps": runner.nonfinite_steps, "curriculum": runner.env.curriculum_state_dict(),
            "networks": {name: getattr(runner, name).state_dict() for name in ("actor", "model", "decoder", "critic", "failure")},
            "optimizers": {name: getattr(runner.learner, "opt_" + name).state_dict() for name in ("model", "actor", "critic", "failure")},
            "updates": {name: int(getattr(runner.learner, name)) for name in ("model_updates", "actor_updates", "critic_updates", "failure_updates", "actor_skipped")}}
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(data, temporary)
    temporary.replace(path)


def load_checkpoint(runner, path):
    data = torch.load(path, map_location=runner.device, weights_only=True)
    if data["format"] != FORMAT or data["spec"] != asdict(runner.spec):
        raise ValueError("Checkpoint format or task specification differs from the active depth task")
    if "num_envs" not in data:
        raise ValueError("Checkpoint is missing its original environment count")
    if data["num_envs"] != runner.env.num_envs:
        raise ValueError("Checkpoint resumption requires the original environment count")
    if asdict(DepthSuperMimicConfig.from_checkpoint(data["config"])) != asdict(runner.config):
        raise ValueError("Checkpoint algorithm configuration differs from the active depth task")
    if "task_configuration" not in data:
        raise ValueError("Checkpoint is missing the task semantic configuration")
    if data["task_configuration"] != runner.task.semantic_configuration():
        raise ValueError("Checkpoint task semantic configuration differs from the active depth task")
    optimizer_names = ("model", "actor", "critic", "failure")
    if set(data["optimizers"]) != set(optimizer_names):
        raise ValueError("Checkpoint optimizer set differs from the active depth task")
    # 先核对全部优化器结构，禁止类型不兼容的快照已经覆盖部分网络。
    for name in optimizer_names:
        optimizer = getattr(runner.learner, "opt_" + name)
        state = data["optimizers"][name]
        composite = isinstance(optimizer, MuonAdam)
        expected = {"muon", "adam"} if composite else {"state", "param_groups"}
        if not isinstance(state, dict) or set(state) != expected:
            raise ValueError(f"Checkpoint optimizer type differs for {name}; optimizer types cannot be converted")
        components = state.values() if composite else (state,)
        if any(not isinstance(part, dict) or set(part) != {"state", "param_groups"} for part in components):
            raise ValueError(f"Checkpoint optimizer state is invalid for {name}")
    for name, state in data["networks"].items():
        getattr(runner, name).load_state_dict(state)
    for name, state in data["optimizers"].items():
        optimizer = getattr(runner.learner, "opt_" + name)
        if isinstance(optimizer, MuonAdam):
            optimizer.load_state_dict(state)
            continue
        # 当前执行方式决定 fused/capturable；不从旧机器恢复设备实现开关。
        groups, learning_rates = [], []
        for current, saved in zip(optimizer.param_groups, state["param_groups"], strict=True):
            learning_rates.append(current["lr"])
            groups.append({**saved, "fused": current.get("fused"), "capturable": current.get("capturable", False),
                           "lr": saved["lr"]})
        optimizer.load_state_dict({**state, "param_groups": groups})
        for group, reference in zip(optimizer.param_groups, learning_rates, strict=True):
            # 图读取长期存活的LR张量；恢复其值，不替换地址。
            if isinstance(reference, torch.Tensor):
                reference.copy_(torch.as_tensor(group["lr"], device=reference.device))
                group["lr"] = reference
            else:
                group["lr"] = float(group["lr"])
            if not group.get("capturable", False) and not group.get("fused", False):
                for parameter in group["params"]:
                    state = optimizer.state.get(parameter, {})
                    if "step" in state:
                        state["step"] = state["step"].cpu()
    for name, value in data["updates"].items():
        setattr(runner.learner, name, value)
    runner.iteration, runner.real_steps = data["iteration"], data["real_steps"]
    runner.prefill_real_steps, runner.elapsed = data["prefill_real_steps"], data["elapsed"]
    runner.training_seconds = data["training_seconds"]
    runner.nonfinite_steps = data["nonfinite_steps"]
    runner.env.load_curriculum_state_dict(data["curriculum"])
    runner.learner.reset_graphs()
    runner.reset_collection()
    return runner.iteration
