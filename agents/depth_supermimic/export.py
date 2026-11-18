# Copyright (c) 2026-2027 zh
"""内容：导出只含深度编码、落足查询与单路动作MLP的部署策略。"""

from copy import deepcopy
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from torch import nn

from .policy import Mlp


class DepthDeployment(nn.Module):
    """显式输入四幅已编码的深度帧和五帧本体，调用者负责米制预处理与帧时序。"""

    def __init__(self, actor, perception):
        super().__init__()
        self.actor = deepcopy(actor)
        self.encoder = deepcopy(perception.encoder)
        self.fpm = deepcopy(perception.fpm)

    def forward(self, history, depth_frames):
        batch = history.shape[0]
        tokens = self.encoder(depth_frames.reshape(-1, 1, 64, 64)).reshape(batch, 64, 16)
        points = self.fpm(self.actor.normalized_history(history)).reshape(batch, 4, 2)
        return self.actor(history, tokens, points)


def export_policy(actor, perception, directory, *, iteration, metadata=None):
    """先生成并校验全部部署产物，再原子替换当前版本。"""
    import onnx

    if isinstance(iteration, bool) or not isinstance(iteration, int) or iteration < 0:
        raise ValueError("Export iteration must be a nonnegative integer")
    module = DepthDeployment(actor, perception).cpu().eval().requires_grad_(False)
    for layer in module.modules():
        if isinstance(layer, Mlp):
            layer.use_bf16 = False
    if any(not torch.isfinite(value).all() for value in module.state_dict().values()):
        raise ValueError("Cannot export non-finite deployment parameters")
    if module.actor.normalizer.count <= 0 or (module.actor.normalizer.var < 0).any():
        raise ValueError("Invalid policy normalization statistics")
    information = {
        **(metadata or {}), "format": "supermimic_depth_query_policy_v2", "iteration": iteration,
        "inputs": {"history": [None, 5, 47], "depth_frames": [None, 4, 1, 64, 64]},
        "outputs": {"actions": [None, 12]}, "dtype": "float32", "actor_hidden": [512, 256, 128],
        "history_order": "oldest_to_newest", "depth_order": "oldest_to_newest_camera_refresh",
        "reset": "repeat_initial_proprioception_and_depth_frame",
        "depth_processing": "optical_z_meters_finite_gt_0_clamp_0_2_divide_2_minus_0_5_else_plus_0_5",
        "control_hz": 50, "camera_hz": 10, "camera_period_control_steps": 5,
        "action_semantics": "unscaled_joint_position_offsets", "action_transform": "identity",
        "training_only": ["world_model", "visual_prior", "critic", "decoder"],
    }
    destination = Path(directory)
    destination.mkdir(parents=True, exist_ok=True)
    example = (torch.zeros(1, 5, 47), torch.zeros(1, 4, 1, 64, 64))
    with TemporaryDirectory(prefix=".depth_export_", dir=destination) as temporary:
        temporary = Path(temporary)
        traced = torch.jit.trace(module, example)
        traced.save(str(temporary / "policy.pt"))
        torch.onnx.export(module, example, str(temporary / "policy.onnx"), export_params=True,
                          opset_version=18, dynamo=False, input_names=["history", "depth_frames"],
                          output_names=["actions"], external_data=False,
                          dynamic_axes={"history": {0: "batch"}, "depth_frames": {0: "batch"},
                                        "actions": {0: "batch"}})
        model = onnx.load(temporary / "policy.onnx")
        for key, value in information.items():
            item = model.metadata_props.add()
            item.key, item.value = key, json.dumps(value, ensure_ascii=False, allow_nan=False)
        onnx.checker.check_model(model)
        onnx.save(model, temporary / "policy.onnx")
        (temporary / "policy.json").write_text(json.dumps(information, ensure_ascii=False, allow_nan=False,
                                                         indent=2) + "\n", encoding="utf-8")
        for suffix in ("pt", "onnx", "json"):
            os.replace(temporary / f"policy.{suffix}", destination / f"policy.{suffix}")
    return {name: destination / f"policy.{name}" for name in ("pt", "onnx", "json")}
