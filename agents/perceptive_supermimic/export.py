# Copyright (c) 2026-2027 zh
"""内容：导出只包含视觉感知与动作策略的TorchScript、ONNX及语义元数据。"""

from copy import deepcopy
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory

import torch


def _prepare_policy(actor, iteration: int, metadata):
    """在导出边界校验参数与元数据，部署副本固定在CPU且不影响训练对象。"""
    if isinstance(iteration, bool) or not isinstance(iteration, int) or iteration < 0:
        raise ValueError("Export iteration must be a nonnegative integer")
    if any(not torch.isfinite(tensor).all() for tensor in actor.state_dict().values()):
        raise ValueError("Cannot export non-finite policy tensors")
    if (actor.normalizer.var < 0).any() or actor.normalizer.count <= 0:
        raise ValueError("Invalid policy normalization statistics")
    information = {
        "format": "perceptive_supermimic_visual_clock_policy_v4",
        "model_id": actor.model_id,
        "iteration": iteration,
        "spec": json.loads(actor.spec_json),
        "spec_hash": actor.spec_hash,
        "semantic_id": actor.semantic_id,
        "depth_semantic_id": actor.depth_semantic_id,
        "input_names": ["obs_history", "depth"],
        "output_names": ["actions"],
        "input_shapes": {"obs_history": [None, actor.history_length, actor.observation_dim],
                         "depth": [None, *actor.depth_shape]},
        "output_shape": [None, actor.action_dim],
        "dtype": "float32",
        "history_order": "frame_major_oldest_first",
        "history_reset": "repeat_initial_frame",
        "depth_input": "task_processed_values_in_0_1",
        "visual_clock": {"interval_control_steps": json.loads(actor.spec_json)["visual_interval"],
                         "stack_order": "oldest_first", "reset": "repeat_new_frame",
                         "hold": "reuse_processed_stack_until_next_acquisition"},
        "normalization": {"statistics_in_graph": True, "history": "exclude_command_then_flatten",
                          "command": "task_observation_scaling_without_running_normalization",
                          "formula": "clamp((x-mean)/(sqrt(var)+1e-6),-10,10)"},
        "command_indices": actor.command_indices.cpu().tolist(),
        "proprioceptive_indices": actor.proprio_indices.cpu().tolist(),
        "action_transform": "identity_after_symmetric_clip",
        "action_clip": actor.action_clip,
        "action_mapping": "defined_by_task_semantic_id",
        "latent_dim": actor.latent_dim,
        "history_encoder": {"type": "mlp", "hidden_dims": list(actor.history_hidden), "activation": "elu",
                            "latent_dim": actor.proprio_latent_dim, "velocity_dim": actor.velocity_dim,
                            "latent_normalization": "none"},
        "visual_projector": {"input_dim": actor.latent_dim, "hidden_dims": list(actor.visual_hidden),
                             "output_dim": actor.visual_policy_dim, "activation": "elu",
                             "gradient_boundary": "detach_visual_input_then_train_projection"},
        "velocity_estimate": {"frame": "body", "units": "m/s", "components": ["vx", "vy", "vz"],
                              "supervision": "same_time_real_state_mse"},
        "actor_input_order": actor.input_order,
        "actor_input_dim": actor.proprio_latent_dim + actor.velocity_dim + actor.command_dim + actor.visual_policy_dim,
        "actor_hidden_dims": list(actor.actor_hidden),
    }
    if metadata is not None:
        overlap = set(information).intersection(metadata)
        if overlap:
            raise ValueError(f"Metadata cannot override policy semantics: {sorted(overlap)}")
        information.update(metadata)
    json.dumps(information, ensure_ascii=False, allow_nan=False)
    return deepcopy(actor).cpu().eval().requires_grad_(False), information


def _write_onnx(module, path: Path, information):
    """以动态批次导出双输入网络，任务语义随ONNX保留。"""
    import onnx

    history = torch.zeros(1, module.history_length, module.observation_dim)
    depth = torch.zeros(1, *module.depth_shape)
    torch.onnx.export(module, (history, depth), str(path), export_params=True,
                      opset_version=18, dynamo=False, external_data=False,
                      input_names=["obs_history", "depth"], output_names=["actions"],
                      dynamic_axes={"obs_history": {0: "batch"}, "depth": {0: "batch"}, "actions": {0: "batch"}})
    model = onnx.load(path)
    for name, value in information.items():
        field = model.metadata_props.add()
        field.key, field.value = name, json.dumps(value, ensure_ascii=False, allow_nan=False)
    onnx.checker.check_model(model)
    onnx.save(model, path)


def export_policy_onnx(actor, path, *, iteration: int, metadata=None):
    """写入指定ONNX路径，成功生成并校验后再替换目标文件。"""
    module, information = _prepare_policy(actor, iteration, metadata)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".perceptive_export_", dir=destination.parent) as temporary:
        pending = Path(temporary) / "policy.onnx"
        _write_onnx(module, pending, information)
        os.replace(pending, destination)
    return destination


def export_policy(actor, directory, *, iteration: int, metadata=None):
    """显式导出部署产物；图像处理按depth_semantic_id由部署任务实现。"""
    module, information = _prepare_policy(actor, iteration, metadata)
    destination = Path(directory)
    destination.mkdir(parents=True, exist_ok=True)
    paths = {"pt": destination / f"policy_{iteration}.pt",
             "onnx": destination / f"policy_{iteration}.onnx",
             "metadata": destination / f"policy_{iteration}.json"}
    description = json.dumps(information, ensure_ascii=False, allow_nan=False, indent=2) + "\n"
    with TemporaryDirectory(prefix=".perceptive_export_", dir=destination) as temporary:
        temporary = Path(temporary)
        torch.jit.script(module).save(str(temporary / "policy.pt"), _extra_files={"metadata.json": description})
        _write_onnx(module, temporary / "policy.onnx", information)
        (temporary / "policy.json").write_text(description, encoding="utf-8")
        for name, path in paths.items():
            suffix = "json" if name == "metadata" else name
            os.replace(temporary / f"policy.{suffix}", path)
    return paths
