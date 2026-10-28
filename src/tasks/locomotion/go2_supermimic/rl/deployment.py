# Copyright (c) 2026-2027 zh
"""
内容：
    由检查点装配部署策略，并生成与任务配置一致的部署语义元数据。
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from agents.supermimic.config import SuperMimicConfig
from agents.supermimic.export import export_policy
from agents.supermimic.interfaces import AlgorithmSpec
from agents.supermimic.policy import make_actor

from ..config.loading import apply_settings, configuration_for_settings, read_settings
from ..mdp.reference_task import Go2ReferenceTaskModel
from ..mdp.state_spec import JOINT_NAMES


def deployment_metadata(cfg, *, spec_hash: str, algorithm) -> dict:
    """只从任务配置与算法配置派生，保证离线导出与部署检查使用同一份语义。"""
    control = cfg.control
    return {
        "spec_hash": spec_hash, "algorithm": "moe_history_encoder",
        "observation_layout": [
            {"name": "base_angular_velocity_body", "width": 3},
            {"name": "projected_gravity_body", "width": 3},
            {"name": "joint_position_offset_with_encoder_bias", "width": 12},
            {"name": "joint_velocity", "width": 12},
            {"name": "previous_unscaled_action", "width": 12},
            {"name": "velocity_command_heading_xy_world_yaw", "width": 3},
        ],
        "joint_names": list(JOINT_NAMES), "action_joint_names": list(JOINT_NAMES),
        "default_joint_positions": [float(cfg.init_state.default_joint[name]) for name in JOINT_NAMES],
        "action_scales": [control.action_scales[name] for _ in range(4) for name in ("hip", "thigh", "calf")],
        "joint_target_formula": "default_joint_positions + actions * action_scales",
        "joint_target_coordinate": "encoder",
        "simulation_physical_target_formula": "default_joint_positions + actions * action_scales - encoder_bias",
        "control_dt_seconds": float(cfg.sim.timestep * cfg.control.decimation),
        "observation_semantics": "blind_proprioception_and_heading_velocity_command",
        "history_encoder": {"type": "mixture_of_experts", "input": "five_frame_blind_history",
                            "latent_dim": int(algorithm.moe_latent_dim),
                            "experts": int(algorithm.moe_expert_num),
                            "gate": "softmax_over_experts",
                            "current_frame_fused_with_latent": True},
        "velocity_input": "not_observed_estimated_from_history",
        "reference_commit": "d9eee9b3f248e5300e636c433255db2c1c3e086a",
    }


def checkpoint_actor(checkpoint: dict):
    """只从检查点装配actor；部署不分任务策略之外的世界模型与critic。"""
    if not isinstance(checkpoint.get("config"), dict) or not isinstance(checkpoint.get("spec"), dict):
        raise ValueError("Checkpoint is missing the algorithm configuration or task specification")
    if any(not torch.isfinite(value).all() for value in checkpoint["actor"].values()):
        raise ValueError("Checkpoint contains non-finite policy tensors")
    spec = AlgorithmSpec(**checkpoint["spec"])
    algorithm = SuperMimicConfig(**checkpoint["config"])
    # 观测分段宽度是任务侧常量，装配策略时不需要创建仿真器。
    actor = make_actor(spec, algorithm, Go2ReferenceTaskModel.observation_term_dims)
    actor.load_state_dict(checkpoint["actor"], strict=True)
    return actor.eval().requires_grad_(False), algorithm


def export_checkpoint(checkpoint_path: str | Path, directory: str | Path, *, iteration: int | None = None):
    """读取检查点与同目录任务快照，导出PT/ONNX/JSON三件套。"""
    path = Path(checkpoint_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint file not found: {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or checkpoint.get("format") != "supermimic_moe_v1":
        raise ValueError(f"Not a SuperMimic checkpoint: {path}")
    # 任务快照与播放共用同一份可信来源和同一套覆盖规则。
    snapshot_path = path.parent / "task_config.json"
    settings, wrapped = read_settings(snapshot_path)
    if not wrapped:
        raise ValueError(f"Export requires a complete task snapshot: {snapshot_path}")
    cfg = configuration_for_settings(settings)
    apply_settings(cfg, settings, replace_mappings=True)
    if json.loads(snapshot_path.read_text(encoding="utf-8")).get("spec_hash") != checkpoint.get("spec_hash"):
        raise ValueError("Checkpoint and task snapshot metadata mismatch")
    actor, algorithm = checkpoint_actor(checkpoint)
    iteration = checkpoint["iteration"] if iteration is None else iteration
    if isinstance(iteration, bool) or not isinstance(iteration, int) or iteration < 0:
        raise ValueError("Export iteration must be a nonnegative integer")
    return export_policy(actor, Path(directory).expanduser().resolve(), iteration=iteration,
                         metadata=deployment_metadata(cfg, spec_hash=checkpoint["spec_hash"], algorithm=algorithm))
