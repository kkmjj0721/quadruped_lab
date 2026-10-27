# Copyright (c) 2026-2027 zh
"""
内容：
    将本体归一化、MoE学生编码器和公共动作网络导出为独立的部署策略。
"""

from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory

import torch
from torch import Tensor, nn

from .models.actor_critic import ActorCritic


class StudentPolicy(nn.Module):
    """接收旧帧到当前帧排列的扁平历史，返回未经关节缩放的确定性动作。"""

    def __init__(self, policy: ActorCritic):
        super().__init__()
        self.observation_dim = policy.spec.observation_dim
        self.history_length = policy.spec.history_length
        self.action_limit = getattr(policy, "action_limit", None)
        self.normalizer = deepcopy(policy.observation_normalizer) if policy.normalize_observations else nn.Identity()
        self.student_encoder = deepcopy(policy.student_encoder)
        self.actor = deepcopy(policy.actor)
        self.cpu().eval().requires_grad_(False)

    def forward(self, obs_history: Tensor) -> Tensor:
        if obs_history.ndim != 2 or obs_history.shape[1] != self.history_length * self.observation_dim:
            raise ValueError("Expected flattened observation history [batch, history_length * observation_dim]")
        history = self.normalizer(obs_history.reshape(-1, self.history_length, self.observation_dim))
        latent, _ = self.student_encoder(history)
        action = self.actor(torch.cat((history[:, -1], latent), dim=-1))
        limit = self.action_limit
        return action if limit is None else limit * torch.tanh(action / limit)


def export_student_policy(policy: ActorCritic, directory: str | Path, *, iteration: int,
                          metadata: dict | None = None) -> dict[str, Path]:
    """先完整生成并检查产物，再替换版本文件和最新文件；不改变训练网络。"""
    if isinstance(iteration, bool) or not isinstance(iteration, int) or iteration < 0:
        raise ValueError("Export iteration must be a nonnegative integer")
    module = StudentPolicy(policy)
    for name, value in module.state_dict().items():
        if not torch.isfinite(value).all():
            raise ValueError(f"Cannot export non-finite student tensor: {name}")
    if policy.normalize_observations:
        if (module.normalizer.variance < 0).any() or module.normalizer.count < 0:
            raise ValueError("Cannot export invalid observation normalization statistics")
    information = {
        **(metadata or {}),
        "format": "supermimic_student_policy_v1", "iteration": iteration,
        "input_name": "obs_history", "output_name": "actions", "dtype": "float32",
        "input_shape": [None, policy.spec.history_length * policy.spec.observation_dim],
        "output_shape": [None, policy.spec.action_dim],
        "observation_dim": policy.spec.observation_dim, "history_length": policy.spec.history_length,
        "history_order": "frame_major_oldest_first", "current_frame_index": policy.spec.history_length - 1,
        "history_reset": "repeat_initial_frame", "num_experts": policy.student_encoder.num_experts,
        "normalization": {"enabled": policy.normalize_observations,
                          "epsilon": policy.observation_normalizer.epsilon,
                          "formula": "(x - mean) / sqrt(variance + epsilon)",
                          "statistics_in_graph": policy.normalize_observations},
        "action_semantics": "unscaled_joint_position_offsets", "action_limit": module.action_limit,
        "action_transform": "identity" if module.action_limit is None else "L * tanh(raw / L)",
        # 保留旧字段：它只表示确定性硬裁剪，平滑输出限制由action_limit描述。
        "action_clip": None, "action_clip_semantics": "deterministic_output_hard_clip",
    }
    encoded = json.dumps(information, ensure_ascii=False, allow_nan=False, indent=2) + "\n"
    import onnx

    destination = Path(directory)
    destination.mkdir(parents=True, exist_ok=True)
    version = f"policy_{iteration}"
    artifacts = {"pt": destination / f"{version}.pt", "onnx": destination / f"{version}.onnx",
                 "metadata": destination / f"{version}.json"}
    with TemporaryDirectory(prefix=".student_export_", dir=destination) as temporary:
        temporary = Path(temporary)
        torch.jit.script(module).save(str(temporary / "policy.pt"))
        torch.onnx.export(
            module, (torch.zeros(1, policy.spec.history_length * policy.spec.observation_dim),),
            str(temporary / "policy.onnx"), export_params=True, opset_version=18,
            input_names=["obs_history"], output_names=["actions"],
            dynamic_axes={"obs_history": {0: "batch"}, "actions": {0: "batch"}},
            dynamo=False, external_data=False,
        )
        onnx_model = onnx.load(temporary / "policy.onnx")
        for key, value in information.items():
            entry = onnx_model.metadata_props.add()
            entry.key, entry.value = key, json.dumps(value, ensure_ascii=False, allow_nan=False)
        onnx.checker.check_model(onnx_model)
        onnx.save(onnx_model, temporary / "policy.onnx")
        (temporary / "policy.json").write_text(encoded, encoding="utf-8")
        for kind, target in artifacts.items():
            filename = "policy.json" if kind == "metadata" else f"policy.{kind}"
            shutil.copyfile(temporary / filename, temporary / target.name)
            os.replace(temporary / target.name, target)
            os.replace(temporary / filename, destination / filename)
    return artifacts
