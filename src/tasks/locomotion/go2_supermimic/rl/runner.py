# Copyright (c) 2026-2027 zh
"""
内容：
    装配Go2任务与SuperMimic算法，并在参数续训时保存／恢复真实课程进度。
"""

import hashlib
import json
from pathlib import Path

import torch

from agents.supermimic import (
    ActorCritic, SuperMimicLearner, SuperMimicRunner, WorldModelTrainer,
)
from agents.supermimic.export import export_student_policy
from src.config.go2.go2_supermimic_config import Go2SuperMimicCfg

from ..config import unitree_go2_supermimic_runner_cfg
from ..mdp.state_spec import JOINT_NAMES
from .transition_adapter import make_go2_supermimic_env
from .world_model import Go2DynamicsModel


class Go2SuperMimicRunner(SuperMimicRunner):
    """扩展任务侧课程持久化；继续保持参数续训而非仿真状态恢复语义。"""

    def _collection_step_metrics(self):
        """分项按真实采样步平均，实际贡献包含权重和控制dt。"""
        metrics = {}
        for name, value in self.env.last_metrics.items():
            if name.startswith("reward/"):
                term = name.removeprefix("reward/")
                metrics[f"task/reward_raw/{term}"] = value
                metrics[f"task/reward/{term}"] = self.env.step_dt * getattr(self.env.cfg.reward.scales, term) * value
            else:
                metrics[f"task/{name}"] = value
        return metrics

    def collect(self, steps):
        # 学习后的闲置Torch缓存会占用Warp图启动需要的显存，只在采样阶段边界归还。
        # 不在每个物理步清缓存，也不释放仍被模型、优化器或回放使用的张量。
        if self.device.type == "cuda":
            idle = torch.cuda.memory_reserved(self.device) - torch.cuda.memory_allocated(self.device)
            if idle >= 256 * 1024 ** 2:
                with torch.cuda.device(self.device):
                    torch.cuda.empty_cache()
        metrics = super().collect(steps)
        metrics.update({f"task_last/{name}": value for name, value in self.env.last_metrics.items()})
        return metrics

    def save(self, path):
        super().save(path)
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        checkpoint["go2_task"] = {"spec_hash": self.spec.hash, "curriculum": self.env.curriculum_state_dict()}
        torch.save(checkpoint, Path(path))
        export_student_policy(self.policy, Path(path).parent, iteration=self.iteration, metadata=self._deployment_metadata())

    def _deployment_metadata(self):
        """物理接口随部署策略保存，调用方负责历史更新和关节目标换算。"""
        control = self.env.cfg.control
        limit = self.policy.action_limit
        return {
            "spec_hash": self.spec.hash,
            "observation_layout": [
                {"name": "base_angular_velocity_body", "width": 3},
                {"name": "projected_gravity_body", "width": 3},
                {"name": "velocity_command_body", "width": 3},
                {"name": "gait_phase_sin_cos", "width": 2},
                {"name": "joint_position_offset", "width": 12},
                {"name": "joint_velocity", "width": 12},
                {"name": "previous_unscaled_action", "width": 12},
            ],
            "joint_names": list(JOINT_NAMES), "action_joint_names": list(JOINT_NAMES),
            "default_joint_positions": self.env.task.default_joint_pos.detach().cpu().tolist(),
            "action_scales": [control.action_scale * (control.hip_reduction if "_hip_" in name else 1.0)
                              for name in JOINT_NAMES],
            "joint_target_formula": "default_joint_positions + actions * action_scales",
            "control_dt_seconds": float(self.env.step_dt),
            "gait_period_seconds": float(self.env.cfg.reward.gait_period),
            "gait_command_threshold": float(self.env.cfg.reward.command_threshold),
            "real_exploration": {
                "action_noise_std": float(self.config.action_noise_std), "hard_clip_limit": limit,
                "formula": "policy_action + noise" if limit is None else "clamp(policy_action + noise, -L, L)",
                "replay_and_history": "executed_action",
            },
        }

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        task_state = checkpoint.get("go2_task")
        if task_state is None or task_state.get("spec_hash") != self.spec.hash:
            raise ValueError("Go2 task checkpoint metadata mismatch")
        result = super().load(path)
        self.env.load_curriculum_state_dict(task_state["curriculum"])
        return result


def make_go2_supermimic_runner(cfg=None, *, device="cpu", log_dir=None, control_gate=None, execution=None):
    """只做组件装配；默认模型预热，显式启用后才执行想象控制更新。"""
    from ..config.loading import is_single_actor_config
    from src.config.go2.go2_single_supermimic_config import Go2SingleActorCfg

    cfg = cfg if cfg is not None else Go2SingleActorCfg()
    if is_single_actor_config(cfg):
        if control_gate is not None:
            raise ValueError("Single-actor training uses explicit warmup; control_gate belongs to legacy CTS")
        from .single_runner import make_single_actor_runner
        return make_single_actor_runner(cfg, device=device, log_dir=log_dir, execution=execution)
    if execution is not None:
        raise ValueError("Execution microbatches are only supported by the single MoE runner")
    rl_cfg = unitree_go2_supermimic_runner_cfg(cfg)
    torch.manual_seed(rl_cfg.seed)
    env = make_go2_supermimic_env(cfg, device=device)
    try:
        policy = ActorCritic(env.task.spec, **rl_cfg.policy).to(env.device)
        model = Go2DynamicsModel(env.task, rl_cfg.model_hidden_dims).to(env.device)
        learner = SuperMimicLearner(policy, model, rl_cfg.learner)
        trainer = WorldModelTrainer(model, **rl_cfg.world_model)
        runner = Go2SuperMimicRunner(env, learner, trainer, rl_cfg.runner, log_dir, control_gate=control_gate,
                                    replay_privileged_fn=env.task.privileged)
        if runner.log_dir is not None:
            root = Path(__file__).resolve().parents[1]
            hashes = {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in root.rglob("*.py")}
            snapshot = {"config": env.cfg.to_dict(), "spec_hash": env.task.spec.hash,
                        "geometry_hash": env.surface.geometry_hash, "source_hashes": hashes}
            (runner.log_dir / "task_config.json").write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
        return runner
    except BaseException:
        env.close()
        raise

