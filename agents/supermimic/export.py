# Copyright (c) 2026-2027 zh
"""内容：导出含混合专家历史编码器、双路观测归一化及动作网络的独立单策略。"""

from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory

import torch


def _prepare_policy(actor, iteration, metadata):
    """冻结部署副本并生成两种导出入口共用的元数据。"""
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
    return module, information


def _write_onnx(module, path, information):
    """把完整策略与元数据写入尚未发布的ONNX文件。"""
    import onnx

    torch.onnx.export(module, (torch.zeros(1, module.history_length * module.observation_dim),),
                      str(path), export_params=True, opset_version=18, dynamo=False,
                      input_names=["obs_history"], output_names=["actions"], external_data=False,
                      dynamic_axes={"obs_history": {0: "batch"}, "actions": {0: "batch"}})
    model = onnx.load(path)
    for name, value in information.items():
        field = model.metadata_props.add()
        field.key, field.value = name, json.dumps(value, ensure_ascii=False, allow_nan=False)
    onnx.checker.check_model(model)
    onnx.save(model, path)


def export_policy_onnx(actor, path, *, iteration: int, metadata=None):
    """检查点保存时仅更新当前ONNX，包含MoE和归一化，不生成逐轮部署副本。"""
    module, information = _prepare_policy(actor, iteration, metadata)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".policy_export_", dir=destination.parent) as temporary:
        pending = Path(temporary) / "policy.onnx"
        _write_onnx(module, pending, information)
        os.replace(pending, destination)
    return destination


def export_policy(actor, directory, *, iteration: int, metadata=None):
    """显式部署导出保留PT／ONNX／JSON及轮次副本，各产物原子替换。"""
    module, information = _prepare_policy(actor, iteration, metadata)

    destination = Path(directory)
    destination.mkdir(parents=True, exist_ok=True)
    paths = {"pt": destination / f"policy_{iteration}.pt", "onnx": destination / f"policy_{iteration}.onnx",
             "metadata": destination / f"policy_{iteration}.json"}
    with TemporaryDirectory(prefix=".policy_export_", dir=destination) as temporary:
        temporary = Path(temporary)
        torch.jit.script(module).save(str(temporary / "policy.pt"))
        _write_onnx(module, temporary / "policy.onnx", information)
        (temporary / "policy.json").write_text(json.dumps(information, ensure_ascii=False, allow_nan=False, indent=2) + "\n", encoding="utf-8")
        for kind, path in paths.items():
            filename = "policy.json" if kind == "metadata" else f"policy.{kind}"
            shutil.copyfile(temporary / filename, temporary / path.name)
            os.replace(temporary / path.name, path)
            os.replace(temporary / filename, destination / filename)
    return paths

