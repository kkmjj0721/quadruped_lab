# Copyright (c) 2026-2027 zh
"""内容：保存真实转移及带噪历史，支持单帧存储重建并构造终止掩码。"""

from __future__ import annotations

from dataclasses import dataclass, fields

import torch
from torch import Tensor

from .tensor_state import TaskState


@dataclass
class Seed:
    state: TaskState
    history: Tensor

    def index(self, index):
        """分批保留种子协议；带外感受状态的种子覆写此方法。"""
        return Seed(self.state.index(index), self.history[index])


@dataclass
class Window:
    """批次优先；目标取下一slot，终止当步的reset后状态仅供参考稀有事件判定。"""

    initial: TaskState
    actions: Tensor
    targets: TaskState
    valid: Tensor
    terminated: Tensor

    def index(self, index):
        return Window(self.initial.index(index), self.actions[index], self.targets.index(index),
                      self.valid[index], self.terminated[index])

    @staticmethod
    def concatenate(windows):
        def states(name):
            return TaskState(**{field.name: torch.cat([getattr(getattr(window, name), field.name) for window in windows])
                                for field in fields(TaskState)})
        return Window(states("initial"), torch.cat([window.actions for window in windows]), states("targets"),
                      torch.cat([window.valid for window in windows]), torch.cat([window.terminated for window in windows]))


class Replay:
    """以半精度保存历史，所有已采集时刻均可独立作为想象起点。

    frames模式要求逐控制步连续插入：新历史移动一帧，结束后重复新回合首帧。
    首次插入和clear后的首次插入均保留调用方传入的完整历史前缀。
    设备索引用于已就绪回放的图捕获；clear后调用方须重新填充再重放采样图。
    """

    def __init__(self, spec, num_envs: int, capacity: int, device, history_storage="full", *, device_sampling=False):
        if capacity < 1:
            raise ValueError("Replay capacity must be positive")
        if history_storage not in ("full", "frames"):
            raise ValueError("Replay history storage must be 'full' or 'frames'")
        self.spec, self.num_envs, self.capacity = spec, num_envs, capacity
        self.device = torch.device(device)
        self.history_storage = history_storage
        self.device_sampling = device_sampling
        self.size = self.clock = 0
        if device_sampling:
            # 图重放读取稳定地址的当前边界；主机计数仍供写入和就绪检查使用。
            self.size_device = torch.zeros((), dtype=torch.long, device=device)
            self.clock_device = torch.zeros_like(self.size_device)
        self.states = self._state_storage((capacity, num_envs))
        self.actions = torch.zeros(capacity, num_envs, spec.action_dim, device=device)
        if history_storage == "full":
            self.histories = torch.zeros(capacity, num_envs, spec.history_length, spec.observation_dim,
                                         device=device, dtype=torch.float16)
        else:
            self.frame_capacity = capacity + spec.history_length - 1
            self.observation_frames = torch.zeros(self.frame_capacity, num_envs, spec.observation_dim,
                                                  device=device, dtype=torch.float16)
            self.history_depth = torch.zeros(capacity, num_envs, device=device, dtype=torch.int32)
            self.next_history_depth = torch.full((num_envs,), spec.history_length, device=device, dtype=torch.int32)
            self.history_offsets = torch.arange(spec.history_length - 1, -1, -1, device=device)
        self.done = torch.zeros(capacity, num_envs, device=device, dtype=torch.bool)
        self.terminated = torch.zeros_like(self.done)

    def _state_storage(self, shape):
        widths = (self.spec.state_dim, self.spec.memory_dim, self.spec.command_dim, self.spec.context_dim)
        return TaskState(*(torch.zeros(*shape, width, device=self.device) for width in widths))

    def clear(self):
        self.size = self.clock = 0
        if self.device_sampling:
            self.size_device.zero_()
            self.clock_device.zero_()
        if self.history_storage == "frames":
            self.history_depth.zero_()
            self.next_history_depth.fill_(self.spec.history_length)

    @torch.no_grad()
    def add(self, state: TaskState, history: Tensor, action: Tensor, result):
        slot = self.clock % self.capacity
        for field in fields(TaskState):
            getattr(self.states, field.name)[slot].copy_(getattr(state, field.name))
        self.actions[slot].copy_(action)
        self.done[slot].copy_(result.terminated | result.truncated)
        self.terminated[slot].copy_(result.terminated)
        if self.history_storage == "full":
            self.histories[slot].copy_(history)
        else:
            if self.clock == 0:
                # 虚拟时刻-(H-1)..-1恰好对应观测环末尾的H-1个槽。
                self.observation_frames[self.capacity:].copy_(history[:, :-1].transpose(0, 1))
            self.observation_frames[self.clock % self.frame_capacity].copy_(history[:, -1])
            self.history_depth[slot].copy_(self.next_history_depth)
            # 结束标志属于本次动作，只限制下一决策态的历史回退深度。
            self.next_history_depth.copy_(torch.where(
                self.done[slot], 1, (self.next_history_depth + 1).clamp_max(self.spec.history_length)))
        self.clock += 1
        self.size = min(self.size + 1, self.capacity)
        if self.device_sampling:
            self.clock_device.add_(1)
            self.size_device.add_(1).clamp_max_(self.capacity)

    def ready(self, horizon: int) -> bool:
        return self.size > horizon + 1

    def _sample_positions(self, batch, horizon, device_indices):
        if device_indices and not self.device_sampling:
            raise ValueError("Device indices require device_sampling=True")
        rows = torch.randint(self.num_envs, (batch,), device=self.device)
        if device_indices:
            # 上界随每次重放变化；FP64量化保持近似均匀分布，随机流不同于randint。
            bound = self.size_device - horizon
            logical = (torch.rand(batch, dtype=torch.float64, device=self.device) * bound).long()
            logical = logical.clamp_max(bound - 1)
            absolute = self.clock_device - self.size_device + logical
        else:
            logical = (torch.randint(0, self.size - horizon, (batch,), device=self.device) if horizon
                       else torch.randint(self.size, (batch,), device=self.device))
            absolute = self.clock - self.size + logical
        return rows, absolute

    def _sample_indices(self, batch: int, horizon: int, *, device_indices=False):
        if batch < 1 or horizon < 1 or not self.ready(horizon):
            raise ValueError("Replay has insufficient state slots for sampling")
        rows, absolute = self._sample_positions(batch, horizon, device_indices)
        return rows, absolute % self.capacity

    def sample_seeds(self, batch: int, *, device_indices=False) -> Seed:
        if batch < 1 or self.size == 0:
            raise ValueError("Cannot sample an empty replay or nonpositive seed batch")
        rows, absolute = self._sample_positions(batch, 0, device_indices)
        slots = absolute % self.capacity
        if self.history_storage == "full":
            history = self.histories[slots, rows].float()
        else:
            depth = self.history_depth[slots, rows]
            back = torch.minimum(self.history_offsets[None, :], depth[:, None] - 1)
            frame_slots = (absolute[:, None] - back) % self.frame_capacity
            history = self.observation_frames[frame_slots, rows[:, None]].float()
        return Seed(self.states.index((slots, rows)), history)

    def sample_windows(self, batch: int, horizon: int, *, device_indices=False) -> Window:
        rows, starts = self._sample_indices(batch, horizon, device_indices=device_indices)
        slots = (starts[:, None] + torch.arange(horizon, device=self.device)) % self.capacity
        done = self.done[slots, rows[:, None]]
        valid = (~done).cumprod(dim=1).bool()
        return Window(self.states.index((starts, rows)), self.actions[slots, rows[:, None]],
                      self.states.index(((slots + 1) % self.capacity, rows[:, None])), valid,
                      self.terminated[slots, rows[:, None]])


