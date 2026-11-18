# Copyright (c) 2026-2027 zh
"""内容：校验本任务检查点与配置，并装配不含训练组件的播放策略。"""

from copy import deepcopy
from dataclasses import asdict, dataclass
import json
from pathlib import Path

import torch

from agents.depth_supermimic.config import DepthSuperMimicConfig
from agents.depth_supermimic.export import DepthDeployment
from agents.depth_supermimic.interfaces import AlgorithmSpec, DEPTH_QUERY_MODEL_ID
from agents.depth_supermimic.perception import Perception
from agents.depth_supermimic.policy import DepthActor

from .config_loading import apply_settings, configuration_for_settings, read_settings
from .train_cfg import TASK_MODES
from .env_cfg import validate_config
from .mdp.reference_task import Go1DepthTaskModel
from .rl.runner import algorithm_config


@dataclass
class PlaybackSource:
    cfg: object
    checkpoint: dict
    snapshot: dict


def load_playback_source(task_id, checkpoint_file):
    if task_id not in TASK_MODES:
        raise ValueError(f"Unknown Go1 depth task: {task_id}")
    if checkpoint_file is None:
        raise ValueError("Trained depth playback requires a checkpoint")
    path = Path(checkpoint_file).expanduser().resolve()
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("format") != DEPTH_QUERY_MODEL_ID:
        raise ValueError("Incompatible Go1 depth checkpoint format")
    settings, wrapped = read_settings(path.parent / "task_config.json")
    if not wrapped:
        raise ValueError("Depth playback requires a complete task snapshot")
    snapshot = json.loads((path.parent / "task_config.json").read_text(encoding="utf-8"))
    if snapshot.get("spec_hash") != checkpoint.get("spec_hash"):
        raise ValueError("Checkpoint and task configuration differ")
    cfg = configuration_for_settings(settings)
    apply_settings(cfg, settings, replace_mappings=True)
    validate_config(cfg)
    if cfg.terrain.mode != TASK_MODES[task_id]:
        raise ValueError("Playback terrain differs from the training checkpoint")
    AlgorithmSpec(**checkpoint["spec"])
    return PlaybackSource(cfg, checkpoint, snapshot)


def build_playback_policy(source, adapter):
    checkpoint = source.checkpoint
    expected, actual = source.cfg.to_dict(), adapter.cfg.to_dict()
    for section, key in (("env", "num_envs"), ("noise", "add_noise"), ("terrain", "curriculum"),
                         ("domain_randomization", "enabled")):
        expected[section][key] = actual[section][key]
    if json.dumps(expected, sort_keys=True) != json.dumps(actual, sort_keys=True):
        raise ValueError("Playback configuration changed unsupported task semantics")
    if source.snapshot["geometry_hash"] != adapter.surface.geometry_hash:
        raise ValueError("Playback terrain geometry differs from the checkpoint")
    original = Go1DepthTaskModel(deepcopy(source.cfg), adapter.task.default_joint_pos,
                                  adapter.task.joint_limits, adapter.surface, device=adapter.device)
    if checkpoint["spec"] != asdict(original.spec) or checkpoint["spec_hash"] != original.spec.hash:
        raise ValueError("Playback robot or saved task specification differs from the checkpoint")
    config = DepthSuperMimicConfig(**checkpoint["config"])
    if asdict(config) != asdict(algorithm_config(source.cfg)):
        raise ValueError("Checkpoint algorithm differs from the task configuration")
    actor, perception = DepthActor(config.actor_hidden), Perception()
    actor.load_state_dict(checkpoint["actor"], strict=True)
    perception.load_state_dict(checkpoint["perception"], strict=True)
    module = DepthDeployment(actor, perception).to(adapter.device).eval().requires_grad_(False)
    if any(not torch.isfinite(value).all() for value in module.state_dict().values()):
        raise ValueError("Playback checkpoint contains non-finite deployment tensors")
    return module
