# Copyright (c) 2026-2027 zh
"""内容：导出含混合专家历史编码器、双路观测归一化及动作网络的独立单策略。"""

from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory

import torch


def export_policy(actor, directory, *, iteration: int, metadata=None):
    """每个产物原子替换；部署图含历史与当前帧两套归一化，不含世界模型、critic 或任务。"""
    if isinstance(iteration, bool) or not isinstance(iteration, int) or iteration < 0:
        raise ValueError("Export iteration must be a nonnegative integer")
    module = deepcopy(actor).cpu().eval().requires_grad_(False)
    if any(not torch.isfinite(value).all() for value in module.state_dict().values()):
        raise ValueError("Cannot export non-finite policy tensors")
    for normalizer in module.policy_normalizers():
        if (normalizer.var < 0).any() or normalizer.count <= 0:
            raise ValueError("Invalid policy normalization statistics")
    information = {
        **(metadata or {}), "format": "supermimic_moe_policy_v1",
        "iteration": iteration,
        "input_name": "obs_history", "output_name": "actions", "dtype": "float32",
        "input_shape": [None, module.history_length * module.observation_dim],
        "output_shape": [None, module.action_dim], "observation_dim": module.observation_dim,
        "history_length": module.history_length, "history_order": module.history_order,
        "history_reset": "repeat_initial_frame",
        "normalization": {"statistics_in_graph": True, "split": "history_and_current_frame",
                          "formula": "clamp((x-mean)/(sqrt(var)+1e-6),-10,10)"},
        "action_semantics": "unscaled_joint_position_offsets", "action_transform": "identity",
        "actor_input_order": module.input_order,
        "encoder": {"type": "moe", "experts": module.encoder.moe.experts.expert_num,
                    "latent_dim": module.encoder.moe.experts.output_dim,
                    "gate_input": "normalized_history"},
    }
    import onnx

    destination = Path(directory)
    destination.mkdir(parents=True, exist_ok=True)
    paths = {"pt": destination / f"policy_{iteration}.pt", "onnx": destination / f"policy_{iteration}.onnx",
             "metadata": destination / f"policy_{iteration}.json"}
    with TemporaryDirectory(prefix=".single_policy_", dir=destination) as temporary:
        temporary = Path(temporary)
        torch.jit.script(module).save(str(temporary / "policy.pt"))
        torch.onnx.export(module, (torch.zeros(1, module.history_length * module.observation_dim),),
                          str(temporary / "policy.onnx"), export_params=True, opset_version=18, dynamo=False,
                          input_names=["obs_history"], output_names=["actions"], external_data=False,
                          dynamic_axes={"obs_history": {0: "batch"}, "actions": {0: "batch"}})
        model = onnx.load(temporary / "policy.onnx")
        for name, value in information.items():
            field = model.metadata_props.add()
            field.key, field.value = name, json.dumps(value, ensure_ascii=False, allow_nan=False)
        onnx.checker.check_model(model)
        onnx.save(model, temporary / "policy.onnx")
        (temporary / "policy.json").write_text(json.dumps(information, ensure_ascii=False, allow_nan=False, indent=2) + "\n", encoding="utf-8")
        for kind, path in paths.items():
            filename = "policy.json" if kind == "metadata" else f"policy.{kind}"
            shutil.copyfile(temporary / filename, temporary / path.name)
            os.replace(temporary / path.name, path)
            os.replace(temporary / filename, destination / filename)
    return paths

