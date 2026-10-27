# Copyright (c) 2026-2027 zh
"""内容：把Go2集中参数转换为SuperMimic网络、拟合与更新调度配置。"""

from dataclasses import dataclass, fields
import math

from agents.supermimic import LearnerConfig, RunnerConfig
from src.config.go2.go2_supermimic_config import Go2SuperMimicCfg


@dataclass(frozen=True)
class Go2SuperMimicRunnerCfg:
    """仅保存配置数据，不构造网络、环境或优化器。"""

    seed: int
    policy: dict
    model_hidden_dims: tuple[int, ...]
    world_model: dict
    learner: LearnerConfig
    runner: RunnerConfig


def unitree_go2_supermimic_runner_cfg(go2_cfg: Go2SuperMimicCfg | None = None) -> Go2SuperMimicRunnerCfg:
    """创建相互独立的网络参数及通用算法配置，供rl/runner.py装配。"""
    cfg = go2_cfg if go2_cfg is not None else Go2SuperMimicCfg()
    dimensions = ("actor_hidden_dims", "critic_hidden_dims", "teacher_hidden_dims", "student_hidden_dims")
    policy = {name: tuple(getattr(cfg.policy, name)) for name in dimensions}
    policy["normalize_observations"] = cfg.policy.normalize_observations
    if cfg.policy.action_limit is not None:
        value = cfg.policy.action_limit
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError("Action limit must be positive and finite, or None for legacy policies")
        policy["action_limit"] = cfg.policy.action_limit
    return Go2SuperMimicRunnerCfg(
        seed=cfg.env.seed, policy=policy, model_hidden_dims=tuple(cfg.model.hidden_dims),
        world_model={name: getattr(cfg.model, name) for name in ("learning_rate", "max_grad_norm", "termination_weight")},
        learner=LearnerConfig(**{field.name: getattr(cfg.learner, field.name) for field in fields(LearnerConfig)}),
        runner=RunnerConfig(**{field.name: getattr(cfg.runner, field.name) for field in fields(RunnerConfig)}),
    )
