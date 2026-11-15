# Copyright (c) 2026-2027 zh
"""内容：装配 Go1 深度策略播放，维护历史、视觉记忆、出生地形、速度控制与奖励显示。"""

from pathlib import Path

import torch
from mjlab.managers.command_manager import NullCommandManager

from agents.depth_supermimic.export import load_actor
from agents.supermimic.interfaces import append_history
from src.config.go1_depth.go1_depth_config import Go1DepthTrainConfig

from .sim_adapter import Go1DepthAdapter
from .training import resolve_config


class _PlaybackAdapter(Go1DepthAdapter):
    """只在播放重置时重采样地形，沿用适配器的状态与相机同步。"""

    def _reset_rows(self, ids):
        if not self.map.plane:
            self.terrain_curriculum.family_ids[ids] = torch.randint(self.map.cols, (ids.numel(),), device=self.device)
            self.terrain_curriculum.levels[ids] = torch.randint(self.map.rows, (ids.numel(),), device=self.device)
        super()._reset_rows(ids)


class _PlaybackRewardManager:
    """沿用 Go2 的显示口径，提供所选环境的各项加权奖励率。"""

    def __init__(self, adapter):
        self.adapter = adapter
        self.rates = {}

    def reset(self, state, env_ids):
        if not self.rates:
            actions = torch.zeros((self.adapter.num_envs, self.adapter.task.spec.action_dim), device=self.adapter.device)
            terms = self.adapter.task.reward_terms(state, actions, state, actions[:, 0])
            self.rates = {name: torch.zeros_like(value) for name, value in terms.items()}
        else:
            ids = slice(None) if env_ids is None else env_ids
            for values in self.rates.values():
                values[ids] = 0

    def update(self, state, result):
        # 使用实际执行动作和自动重置前的终态，与真实单步奖励保持一致。
        terms = self.adapter.task.reward_terms(state, result.executed_action, result.final_state, result.terminated.float())
        self.rates = {name: getattr(self.adapter.cfg.reward.scales, name) * value for name, value in terms.items()}

    def get_active_iterable_terms(self, env_idx):
        return [(name, [values[env_idx].item()]) for name, values in self.rates.items()]

    def get_visualizable_terms(self):
        return []


class _PlaybackCommandManager(NullCommandManager):
    """沿用 Go2 的 Viser 速度面板，通过已有覆盖接口控制全部播放环境。"""

    def __init__(self, adapter):
        super().__init__()
        self.active_terms = ["twist"]
        self.adapter = adapter
        self._enabled = None
        self._sliders = []
        self._command = None

    def create_gui(self, server, get_env_idx, on_change=None, request_action=None):
        """界面回调只修改滑条状态，命令由仿真线程统一应用。"""
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
        """关闭手动控制后恢复任务自身的命令采样和 heading 反馈。"""
        command = (tuple(slider.value for slider in self._sliders)
                   if self._enabled is not None and self._enabled.value else None)
        if command == self._command:
            return False
        self.adapter.set_play_commands(command)
        self._command = command
        return True

    def get_active_iterable_terms(self, env_idx):
        return [("twist", self.adapter.get_state().command[env_idx].cpu().tolist())]


class Go1DepthPlayEnv:
    """查看器使用标准仿真属性，重置和步进统一经过 Go1 转移适配器。"""

    def __init__(self, adapter):
        self.adapter = adapter
        self.command_manager = _PlaybackCommandManager(adapter)
        self.reward_manager = _PlaybackRewardManager(adapter)
        self.history = None
        self.hidden = torch.zeros(adapter.num_envs, adapter.task.spec.latent_dim, device=adapter.device)

    def __getattr__(self, name):
        return getattr(self.adapter.raw_env, name)

    @property
    def unwrapped(self):
        return self

    @torch.no_grad()
    def reset(self, *, env_ids=None):
        self._update_commands()
        state = self.adapter.reset(env_ids=env_ids)
        self.reward_manager.reset(state, env_ids)
        self.cfg.viewer.reward_bar_max_terms = len(self.reward_manager.rates)
        frame = self.adapter.proprio_frame()
        if env_ids is None or self.history is None:
            self.history = frame[:, None].repeat(1, self.adapter.task.spec.history_length, 1)
            self.hidden.zero_()
        else:
            self.history[env_ids] = frame[env_ids, None].expand(-1, self.history.shape[1], -1)
            self.hidden[env_ids] = 0
        return self.get_observations()

    def _update_commands(self):
        """仅刷新当前决策帧，保留旧历史、已采集的本体测量和视觉记忆。"""
        if self.command_manager.update() and self.history is not None:
            self.history[:, -1] = self.adapter.proprio_frame()

    def get_observations(self):
        self._update_commands()
        return {"history": self.history, "depth": self.adapter.depth_stack(), "visual_hidden": self.hidden}

    @torch.no_grad()
    def step(self, actions):
        self._update_commands()
        before = self.adapter.get_state()
        result = self.adapter.step(actions)
        self.reward_manager.update(before, result)
        frame = result.next_observation
        history = append_history(self.history, frame)
        done = result.terminated | result.truncated
        self.history = torch.where(done[:, None, None], frame[:, None].expand_as(history), history)
        self.hidden.masked_fill_(done[:, None], 0)
        return self.get_observations(), result.reward, result.terminated, result.truncated, {"final_state": result.final_state}

    def close(self):
        self.adapter.close()


def make_playback(task_id, *, checkpoint_file=None, agent="trained", num_envs=1, device="cpu", render_mode=None,
                  video_height=None, video_width=None, no_terminations=False):
    """从同目录任务快照恢复感知和控制规格；播放不分配世界模型、优化器或回放。"""
    if agent not in ("trained", "zero", "random"):
        raise ValueError(f"Unsupported Go1 depth playback agent: {agent}")
    checkpoint = Path(checkpoint_file).resolve() if checkpoint_file is not None else None
    if agent == "trained" and checkpoint is None:
        raise ValueError("Trained Go1 depth playback requires a checkpoint")
    args = Go1DepthTrainConfig(device=device, num_envs=num_envs, resume=checkpoint if agent == "trained" else None)
    cfg, _, _ = resolve_config(task_id, args, training=False)
    torch.manual_seed(cfg.env.seed)
    actor, spec = (None, None)
    if agent == "trained":
        actor, spec, _, _ = load_actor(checkpoint, device)
    adapter = _PlaybackAdapter(cfg, device=device, play=True, render_mode=render_mode,
                               video_height=video_height, video_width=video_width, no_terminations=no_terminations)
    try:
        if spec is not None and spec != adapter.task.spec:
            raise ValueError("Playback task dimensions differ from the checkpoint")
        env = Go1DepthPlayEnv(adapter)
        env.reset()

        @torch.no_grad()
        def policy(observations):
            if actor is not None:
                actions, hidden = actor(observations["history"], observations["depth"], observations["visual_hidden"])
                env.hidden.copy_(hidden)
                return actions
            shape = (adapter.num_envs, adapter.task.spec.action_dim)
            return torch.zeros(shape, device=adapter.device) if agent == "zero" else 2 * torch.rand(shape, device=adapter.device) - 1

        return env, policy
    except BaseException:
        adapter.close()
        raise
