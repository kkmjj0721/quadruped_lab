# Copyright (c) 2026-2027 zh
"""内容：维护 Go1 查看器本体历史、命令面板、奖励展示与课程恢复。"""

import torch
from mjlab.managers.command_manager import NullCommandManager
from mjlab.utils.wrappers import VideoRecorder

from agents.depth_supermimic.tensor_state import append_history


class _PlaybackRewardManager:
    """向查看器提供所选环境的加权奖励率，不乘控制周期。"""

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

    def update(self, state, actions, result):
        # 用真实转移的终态重算同一组纯张量评分，避免读取自动重置后的状态。
        terms = self.adapter.task.reward_terms(state, actions, result.final_state, result.terminated.float())
        self.rates = {name: getattr(self.adapter.cfg.reward.scales, name) * value for name, value in terms.items()}

    def get_active_iterable_terms(self, env_idx):
        return [(name, [values[env_idx].item()]) for name, values in self.rates.items()]

    def get_visualizable_terms(self):
        return []


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
    """显式转发策略观测，避免继承的原生manager读取绕过历史适配。"""

    def get_observations(self):
        return self._wrapped_env.get_observations()


class SuperMimicPlayEnv:
    """将任务转移转换为查看器协议；原生属性委托，重置始终经过任务adapter。"""

    def __init__(self, adapter):
        self.adapter = adapter
        self.command_manager = _PlaybackCommandManager(adapter)
        self.reward_manager = _PlaybackRewardManager(adapter)
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
        self.reward_manager.reset(state, env_ids)
        self.cfg.viewer.reward_bar_max_terms = len(self.reward_manager.rates)
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
        before = self.adapter.get_state()
        result = self.adapter.step(actions)
        self.reward_manager.update(before, actions, result)
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
    saved = checkpoint["curriculum"]
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

