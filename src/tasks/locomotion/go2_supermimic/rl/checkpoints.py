# Copyright (c) 2026-2027 zh
"""
内容：
    恢复SuperMimic训练配置并校验播放检查点，只装配策略和归一化统计。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import torch

from agents.supermimic.single.models import MoeActor, make_actor
from src.config.go2.go2_reference_supermimic_config import Go2ReferenceCfg

from ..config.train_cfg import TASK_MODES
from ..go_supermimic_env_cfg import validate_config
from ..mdp.reference_task import Go2ReferenceTaskModel
from .single_runner import algorithm_config
from ..config.loading import (
    apply_settings, read_settings, configuration_for_settings,
)


@dataclass(frozen=True)
class PlaybackSource:
    """原训练配置与CPU检查点；播放覆盖应施加到独立的配置副本。"""

    cfg: Go2ReferenceCfg
    checkpoint: dict
    path: Path
    snapshot: dict


def load_playback_source(task_id: str, checkpoint_file: str | Path) -> PlaybackSource:
    """读取可信本项目检查点及同目录的完整任务快照，不创建仿真。"""
    if task_id not in TASK_MODES:
        raise ValueError(f"Unknown Go2 SuperMimic task: {task_id}")
    path = Path(checkpoint_file).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint file not found: {path}")
    snapshot_path = path.parent / "task_config.json"
    settings, wrapped = read_settings(snapshot_path)
    if not wrapped:
        raise ValueError("Playback requires a complete task_config.json snapshot with metadata")
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    if any(not isinstance(snapshot.get(name), str) or not snapshot[name] for name in ("spec_hash", "geometry_hash")):
        raise ValueError("Task snapshot is missing spec_hash or geometry_hash metadata")
    cfg = configuration_for_settings(settings)
    sections = set(cfg.to_dict()) - {"init_member_classes", "to_dict"}
    if not sections.issubset(settings):
        raise ValueError("Playback requires all task configuration sections")
    apply_settings(cfg, settings, replace_mappings=True)
    validate_config(cfg)
    if cfg.terrain.mode != TASK_MODES[task_id]:
        raise ValueError("Playback task terrain mode differs from the saved configuration")
    # 与训练保持相同的可信文件约定；只把所需策略装配到目标设备。
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    expected_format = "supermimic_moe_v1"
    if not isinstance(checkpoint, dict) or checkpoint.get("format") != expected_format:
        raise ValueError("Incompatible SuperMimic checkpoint format")
    task_metadata = checkpoint.get("go2_task")
    if (not isinstance(task_metadata, dict) or not isinstance(task_metadata.get("curriculum"), dict)
            or task_metadata.get("spec_hash") != snapshot["spec_hash"]
            or checkpoint.get("spec_hash") != snapshot["spec_hash"]):
        raise ValueError("Checkpoint and task snapshot metadata mismatch")
    if not isinstance(checkpoint.get("spec"), dict) or not isinstance(checkpoint.get("actor"), dict):
        raise ValueError("Checkpoint is missing the algorithm spec or policy state")
    return PlaybackSource(cfg, checkpoint, path, snapshot)


def build_playback_policy(source: PlaybackSource, adapter) -> MoeActor:
    """用原训练语义验证权重，同时允许单环境、无噪声和固定课程播放。"""
    expected, actual = source.cfg.to_dict(), adapter.cfg.to_dict()
    for section, name in (("env", "num_envs"), ("noise", "add_noise"), ("terrain", "curriculum")):
        expected[section][name] = actual[section][name]
    expected["domain_randomization"]["enabled"] = actual["domain_randomization"]["enabled"]
    if json.dumps(expected, sort_keys=True) != json.dumps(actual, sort_keys=True):
        raise ValueError("Playback configuration changed unsupported task semantics")
    if source.snapshot["geometry_hash"] != adapter.surface.geometry_hash:
        raise ValueError("Playback terrain geometry differs from the saved task")
    original_task = Go2ReferenceTaskModel(source.cfg, adapter.task.default_joint_pos, adapter.task.joint_limits,
                                         adapter.surface, device=adapter.device)
    checkpoint = source.checkpoint
    if checkpoint["spec"] != asdict(original_task.spec) or checkpoint["spec_hash"] != original_task.spec.hash:
        raise ValueError("Checkpoint task spec does not match the saved configuration and current robot")
    policy_config = algorithm_config(source.cfg)
    if checkpoint.get("config") != asdict(policy_config):
        raise ValueError("Single-actor checkpoint configuration mismatch")
    policy = make_actor(original_task.spec, policy_config,
                        observation_term_dims=original_task.observation_term_dims).to(adapter.device)
    weights = checkpoint["actor"]
    for name, tensor in weights.items():
        if not isinstance(tensor, torch.Tensor) or not torch.isfinite(tensor).all():
            raise ValueError(f"Invalid checkpoint policy tensor: {name}")
        if name.endswith(("normalizer.variance", "normalizer.var", "normalizer.count")) and (tensor < 0).any():
            raise ValueError(f"Invalid checkpoint normalization statistics: {name}")
    policy.load_state_dict(weights, strict=True)
    policy.eval()
    policy.requires_grad_(False)
    return policy
