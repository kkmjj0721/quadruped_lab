# Copyright (c) 2026-2027 zh
"""内容：导出双帧编码器、固定 GRU 和 actor，显式传递部署时的视觉记忆。"""

from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path

import torch

from .checkpoints import FORMAT
from .config import DepthSuperMimicConfig
from .interfaces import DepthSpec
from .networks import DepthActor


def load_actor(checkpoint_path, device="cpu"):
    """只读取策略网络，播放与导出不分配回放、世界模型或优化器。"""
    saved = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if saved["format"] != FORMAT:
        raise ValueError("Unsupported depth policy checkpoint format")
    spec = DepthSpec(**saved["spec"])
    config = DepthSuperMimicConfig.from_checkpoint(saved["config"])
    execution, cuda = saved.get("execution", {}), torch.device(device).type == "cuda"
    actor_bf16, encoder_bf16 = execution.get("actor_bf16", False), execution.get("encoder_bf16", False)
    actor = DepthActor(spec, config, bf16=actor_bf16 and cuda, encoder_bf16=encoder_bf16 and cuda).to(device).eval()
    actor.load_state_dict(saved["networks"]["actor"])
    # CPU播放仍记录训练来源精度，后续导出不误报为FP32训练。
    actor.source_network_precision = {"actor": "bf16" if actor_bf16 else "fp32", "encoder": "bf16" if encoder_bf16 else "fp32"}
    return actor, spec, config, saved["iteration"]


def export_policy(actor, path, *, spec, iteration, metadata=None, onnx=False):
    """显式输入本体历史、双帧深度和 GRU 状态，同时输出动作和下一视觉状态。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    source_precision = getattr(actor, "source_network_precision", {"actor": "bf16" if actor.bf16 else "fp32",
                                                               "encoder": "bf16" if actor.encoder.bf16 else "fp32"})
    policy = deepcopy(actor).cpu().eval().requires_grad_(False)
    # 部署副本统一FP32，保留训练实例的局部autocast配置。
    policy.bf16 = policy.encoder.bf16 = False
    example = (torch.zeros(1, spec.history_length, spec.observation_dim), torch.zeros(1, *spec.depth_shape),
               torch.zeros(1, spec.latent_dim))
    traced = torch.jit.trace(policy, example)
    with torch.no_grad():
        expected, exported = policy(*example), traced(*example)
    torch.testing.assert_close(exported, expected)
    traced.save(str(path))
    if onnx:
        torch.onnx.export(policy, example, str(path.with_suffix(".onnx")), input_names=["proprio_history", "depth_stack", "visual_hidden"],
                          output_names=["actions", "next_visual_hidden"],
                          dynamic_axes={name: {0: "batch"} for name in ("proprio_history", "depth_stack", "visual_hidden", "actions", "next_visual_hidden")},
                          opset_version=17, dynamo=False)
    information = {"format": "depth_supermimic_policy_v3", "iteration": iteration, "spec": asdict(spec),
                   "inputs": {"proprio_history": ["batch", spec.history_length, spec.observation_dim],
                              "depth_stack": ["batch", *spec.depth_shape],
                              "visual_hidden": ["batch", spec.latent_dim]},
                   "outputs": {"actions": ["batch", spec.action_dim], "next_visual_hidden": ["batch", spec.latent_dim]},
                   "history_order": "oldest_first", "depth_range": [0.0, 1.0],
                   "depth_invalid_value": 0.0, "visual_hidden_initial_value": 0.0,
                   "visual_hidden_reset": "Zero completed or explicitly reset environments before the next policy call",
                   "visual_hidden_feedback": "Pass next_visual_hidden back as visual_hidden on the next control step",
                   "visual_memory_training": "Frozen GRU weights; reconstruction trains the preceding CNN only",
                   "source_network_precision": source_precision, "export_network_precision": "fp32", "metadata": metadata or {}}
    path.with_suffix(".json").write_text(json.dumps(information, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path
