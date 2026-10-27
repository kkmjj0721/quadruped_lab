# Copyright (c) 2026-2027 zh
"""
内容：
    装配Go2盲策略播放，维护本体历史、课程快照和查看器的完整／部分重置。
"""

from __future__ import annotations

from copy import deepcopy

import torch
from mjlab.utils.wrappers import VideoRecorder

from agents.supermimic.interfaces import append_history
from src.config.go2.go2_reference_supermimic_config import Go2ReferenceCfg

from ..config.train_cfg import TASK_MODES
from .checkpoints import build_playback_policy, load_playback_source
from .transition_adapter import make_go2_supermimic_env


class SuperMimicVideoRecorder(VideoRecorder):
    """显式转发盲策略观测，避免继承的原生manager读取绕过历史适配。"""

    def get_observations(self):
        return self._wrapped_env.get_observations()


class SuperMimicPlayEnv:
    """将任务转移转换为查看器协议；原生属性委托，重置始终经过任务adapter。"""

    def __init__(self, adapter):
        self.adapter = adapter
        self._observation = None
        self._history = None

    def __getattr__(self, name):
        return getattr(self.adapter.raw_env, name)

    @property
    def unwrapped(self):
        # Viser的局部重置直接调用unwrapped.reset，不能暴露会绕过任务状态的raw_env。
        return self

    @torch.no_grad()
    def reset(self, *, env_ids=None):
        state = self.adapter.reset(env_ids=env_ids)
        frame = self.adapter.task.observe(state, noisy=False)
        length = self.adapter.task.spec.history_length
        if env_ids is None or self._history is None:
            self._observation = frame
            self._history = frame.unsqueeze(1).repeat(1, length, 1)
        else:
            self._observation[env_ids] = frame[env_ids]
            self._history[env_ids] = frame[env_ids].unsqueeze(1).expand(-1, length, -1)
        return self.get_observations()

    def get_observations(self):
        if self._history is None:
            raise RuntimeError("Reset the SuperMimic playback environment before observing")
        return {"observation": self._observation, "history": self._history}

    @torch.no_grad()
    def step(self, actions):
        result = self.adapter.step(actions)
        frame = self.adapter.task.observe(result.next_state, noisy=False)
        history = append_history(self._history, frame)
        done = result.terminated | result.truncated
        if done.any():
            history[done] = frame[done].unsqueeze(1).expand(-1, history.shape[1], -1)
        self._observation, self._history = frame, history
        return self.get_observations(), result.reward, result.terminated, result.truncated, {"final_state": result.final_state}

    def close(self):
        self.adapter.close()


def _restore_curriculum(adapter, checkpoint):
    """按保存的环境行循环映射地形，保留训练时命令阶段并冻结后续课程。"""
    saved = checkpoint["go2_task"]["curriculum"]
    terrain = saved["terrain"]
    families, levels = terrain["family_ids"], terrain["levels"]
    if (not isinstance(families, torch.Tensor) or not isinstance(levels, torch.Tensor)
            or families.ndim != 1 or families.numel() == 0 or levels.shape != families.shape
            or families.dtype != torch.long or levels.dtype != torch.long):
        raise ValueError("Invalid saved terrain curriculum tensors")
    counts = torch.tensor(adapter.map.level_counts, dtype=torch.long, device=families.device)
    if (families < 0).any() or (families >= counts.numel()).any():
        raise ValueError("Invalid saved terrain family IDs")
    if (levels < 0).any() or (levels >= counts[families]).any():
        raise ValueError("Invalid saved terrain levels")
    rows = torch.arange(adapter.num_envs, device=families.device) % families.numel()
    adapter.terrain_curriculum.family_ids.copy_(families[rows].to(adapter.device))
    adapter.load_curriculum_state_dict({
        "command": saved["command"],
        "terrain": {"family_ids": families[rows], "levels": levels[rows]},
    })


def make_playback(task_id, *, checkpoint_file=None, agent="trained", num_envs=None, device="cpu",
                  render_mode=None, video_height=None, video_width=None, no_terminations=False):
    """返回已重置的播放环境与确定性盲策略策略，不创建训练组件。"""
    if task_id not in TASK_MODES:
        raise ValueError(f"Unknown Go2 SuperMimic task: {task_id}")
    if agent not in ("trained", "zero", "random"):
        raise ValueError(f"Unsupported agent mode: {agent}")
    source = load_playback_source(task_id, checkpoint_file) if agent == "trained" else None
    cfg = deepcopy(source.cfg) if source is not None else Go2ReferenceCfg()
    cfg.terrain.mode = TASK_MODES[task_id]
    cfg.env.num_envs = num_envs if num_envs is not None else 1
    torch.manual_seed(cfg.env.seed)
    adapter = make_go2_supermimic_env(
        cfg, device=device, play=True, render_mode=render_mode, video_height=video_height,
        video_width=video_width, no_terminations=no_terminations,
    )
    try:
        model = build_playback_policy(source, adapter) if source is not None else None
        if source is not None:
            _restore_curriculum(adapter, source.checkpoint)
        env = SuperMimicPlayEnv(adapter)
        env.reset()

        @torch.no_grad()
        def policy(observations):
            if model is not None:
                return model(observations["history"])
            shape = (adapter.num_envs, adapter.task.spec.action_dim)
            if agent == "zero":
                return torch.zeros(shape, device=adapter.device)
            return 2 * torch.rand(shape, device=adapter.device) - 1

        return env, policy
    except BaseException:
        adapter.close()
        raise
