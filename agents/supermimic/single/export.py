# Copyright (c) 2026-2027 zh
"""内容：导出带历史排列、观测归一化及动作网络的独立单策略。"""

from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory

import torch


def export_policy(actor, directory, *, iteration: int, metadata=None):
    """每个产物原子替换；部署图不包含世界模型、critic、任务或教师。"""
    if isinstance(iteration, bool) or not isinstance(iteration, int) or iteration < 0:
        raise ValueError("Export iteration must be a nonnegative integer")
    module = deepcopy(actor).cpu().eval().requires_grad_(False)
    if any(not torch.isfinite(value).all() for value in module.state_dict().values()):
        raise ValueError("Cannot export non-finite policy tensors")
    if (module.normalizer.var < 0).any() or module.normalizer.count <= 0:
        raise ValueError("Invalid policy normalization statistics")
    information = {
        **(metadata or {}), "format": "supermimic_single_mlp_policy_v1",
        "iteration": iteration,
        "input_name": "obs_history", "output_name": "actions", "dtype": "float32",
        "input_shape": [None, module.history_length * module.observation_dim],
        "output_shape": [None, module.action_dim], "observation_dim": module.observation_dim,
        "history_length": module.history_length, "history_order": "frame_major_oldest_first",
        "history_reset": "repeat_initial_frame",
        "normalization": {"statistics_in_graph": True, "formula": "clamp((x-mean)/(sqrt(var)+1e-6),-10,10)"},
        "action_semantics": "unscaled_joint_position_offsets", "action_transform": "identity",
        "actor_input_order": "term_major_oldest_first", "history_permutation": module.history_indices.tolist(),
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

