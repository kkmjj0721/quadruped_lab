# Copyright (c) 2026-2027 zh
"""
内容：
    装配Go2盲策略播放，维护本体历史、课程快照和查看器的完整／部分重置。
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import torch
from mjlab.managers.command_manager import NullCommandManager
from mjlab.utils.wrappers import VideoRecorder

from agents.supermimic.interfaces import append_history
from src.config.go2.go2_reference_supermimic_config import Go2ReferenceCfg

from ..config.train_cfg import TASK_MODES
from .checkpoints import build_playback_policy, load_playback_source
from .transition_adapter import make_go2_supermimic_env


class _PlaybackCommandManager(NullCommandManager):
    """在播放侧提供Viser速度面板，通过已有覆盖接口控制全部播放环境。"""

    def __init__(self, adapter):
        super().__init__()
        self.active_terms = ["twist"]
        self.adapter = adapter
        self._enabled = None
        self._sliders = []
        self._command = None

    def create_gui(self, server, get_env_idx, on_change=None, request_action=None):
        """沿用参考面板的三轴速度、可调范围与归零按钮，回调只修改界面状态。"""
        with server.gui.add_folder("速度控制"):
            self._enabled = server.gui.add_checkbox("启用手动速度（全部环境）", initial_value=False)
            axes = ("前进速度 vx (m/s)", "侧移速度 vy (m/s)", "转向速度 wz (rad/s)")
            for label, bounds in zip(axes, self.adapter.cfg.commands.initial_ranges, strict=True):
                limit = max(abs(bounds[0]), abs(bounds[1]))
                maximum = server.gui.add_slider(f"调节范围：{label}", min=0.1, max=10.0,
                                               step=0.1, initial_value=limit)
                slider = server.gui.add_slider(label, min=-limit, max=limit, step=0.05, initial_value=0.0)

                @maximum.on_update
                def _(_event, target=slider, source=maximum):
                    target.min, target.max = -source.value, source.value

                self._sliders.append(slider)
            zero = server.gui.add_button("速度归零")

            @zero.on_click
            def _(_event):
                for slider in self._sliders:
                    slider.value = 0.0

    def update(self):
        """在仿真线程应用界面输入；关闭手动控制后恢复任务原有命令逻辑。"""
        command = (tuple(slider.value for slider in self._sliders)
                   if self._enabled is not None and self._enabled.value else None)
        if command == self._command:
            return False
        self.adapter.set_play_commands(command)
        self._command = command
        return True

    def get_active_iterable_terms(self, env_idx):
        return [("twist", self.adapter.get_state().command[env_idx].cpu().tolist())]


class SuperMimicVideoRecorder(VideoRecorder):
    """显式转发盲策略观测，避免继承的原生manager读取绕过历史适配。"""

    def get_observations(self):
        return self._wrapped_env.get_observations()


class SuperMimicPlayEnv:
    """将任务转移转换为查看器协议；原生属性委托，重置始终经过任务adapter。"""

    def __init__(self, adapter):
        self.adapter = adapter
        self.command_manager = _PlaybackCommandManager(adapter)
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
        self._update_commands()
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

    def _update_commands(self):
        """同步当前命令与最新观测，局部重置前也更新未重置环境的决策帧。"""
        if self.command_manager.update() and self._history is not None:
            # 更新当前决策帧中的命令，保留旧历史帧，不额外推进历史时间。
            self._observation = self.adapter.task.observe(self.adapter.get_state(), noisy=False)
            self._history[:, -1] = self._observation

    def get_observations(self):
        if self._history is None:
            raise RuntimeError("Reset the SuperMimic playback environment before observing")
        self._update_commands()
        return {"observation": self._observation, "history": self._history}

    @torch.no_grad()
    def step(self, actions):
        self._update_commands()
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
        "terrain": {**terrain, "family_ids": families[rows], "levels": levels[rows]},
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


def make_playback_env(task_id, *, checkpoint_file=None, agent="trained", num_envs=None, device="cpu",
                      video=False, video_length=200, video_height=None, video_width=None,
                      no_terminations=False):
    """在播放环境外按需装配录像；录像器不可用时关闭已建环境并继续抛出。"""
    checkpoint = Path(checkpoint_file).expanduser().resolve() if checkpoint_file is not None else None
    record_video = checkpoint is not None and video
    if video and checkpoint is None:
        print("[WARN] Video recording with dummy agents is disabled (no checkpoint/log_dir).")
    env, policy = make_playback(
        task_id, checkpoint_file=checkpoint, agent=agent, num_envs=num_envs, device=device,
        render_mode="rgb_array" if record_video else None, video_height=video_height,
        video_width=video_width, no_terminations=no_terminations,
    )
    if not record_video:
        return env, policy
    try:
        recorder = SuperMimicVideoRecorder(env, video_folder=checkpoint.parent / "videos" / "play",
                                           step_trigger=lambda step: step == 0,
                                           video_length=video_length, disable_logger=True)
    except BaseException:
        env.close()
        raise
    print("[INFO] Recording videos during play.")
    return recorder, policy
