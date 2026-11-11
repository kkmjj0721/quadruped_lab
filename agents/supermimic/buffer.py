# Copyright (c) 2026-2027 zh
"""内容：保存真实转移及带噪历史，支持单帧存储重建，并构造终止掩码与稀有事件归档。"""

from __future__ import annotations

from dataclasses import dataclass, fields

import torch
from torch import Tensor

from .interfaces import TaskState


@dataclass
class Seed:
    state: TaskState
    history: Tensor


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
    """

    def __init__(self, spec, num_envs: int, capacity: int, device, history_storage="full"):
        if capacity < 1:
            raise ValueError("Replay capacity must be positive")
        if history_storage not in ("full", "frames"):
            raise ValueError("Replay history storage must be 'full' or 'frames'")
        self.spec, self.num_envs, self.capacity = spec, num_envs, capacity
        self.device = torch.device(device)
        self.history_storage = history_storage
        self.size = self.clock = 0
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

    def ready(self, horizon: int) -> bool:
        return self.size > horizon + 1

    def _sample_indices(self, batch: int, horizon: int):
        if batch < 1 or horizon < 1 or not self.ready(horizon):
            raise ValueError("Replay has insufficient state slots for sampling")
        rows = torch.randint(self.num_envs, (batch,), device=self.device)
        logical = torch.randint(0, self.size - horizon, (batch,), device=self.device)
        return rows, (self.clock - self.size + logical) % self.capacity

    def sample_seeds(self, batch: int) -> Seed:
        if batch < 1 or self.size == 0:
            raise ValueError("Cannot sample an empty replay or nonpositive seed batch")
        rows = torch.randint(self.num_envs, (batch,), device=self.device)
        logical = torch.randint(self.size, (batch,), device=self.device)
        absolute = self.clock - self.size + logical
        slots = absolute % self.capacity
        if self.history_storage == "full":
            history = self.histories[slots, rows].float()
        else:
            depth = self.history_depth[slots, rows]
            back = torch.minimum(self.history_offsets[None, :], depth[:, None] - 1)
            frame_slots = (absolute[:, None] - back) % self.frame_capacity
            history = self.observation_frames[frame_slots, rows[:, None]].float()
        return Seed(self.states.index((slots, rows)), history)

    def sample_windows(self, batch: int, horizon: int) -> Window:
        rows, starts = self._sample_indices(batch, horizon)
        slots = (starts[:, None] + torch.arange(horizon, device=self.device)) % self.capacity
        done = self.done[slots, rows[:, None]]
        valid = (~done).cumprod(dim=1).bool()
        return Window(self.states.index((starts, rows)), self.actions[slots, rows[:, None]],
                      self.states.index(((slots + 1) % self.capacity, rows[:, None])), valid,
                      self.terminated[slots, rows[:, None]])


class WindowArchive:
    """只有首次出现可归档数据时才分配固定容量，保存完整窗口而非回放索引。"""

    def __init__(self, capacity: int):
        self.capacity, self.size, self.clock = capacity, 0, 0
        self.storage = None

    def clear(self):
        self.size = self.clock = 0

    @torch.no_grad()
    def add(self, windows: Window):
        count = min(windows.actions.shape[0], self.capacity)
        if count == 0:
            return
        windows = windows.index(slice(0, count))
        if self.storage is None:
            def allocate(value):
                return torch.zeros((self.capacity, *value.shape[1:]), dtype=value.dtype, device=value.device)
            self.storage = Window(windows.initial.map(allocate), allocate(windows.actions), windows.targets.map(allocate),
                                  allocate(windows.valid), allocate(windows.terminated))
        indices = (self.clock + torch.arange(count, device=windows.actions.device)) % self.capacity
        for name in ("initial", "targets"):
            for field in fields(TaskState):
                getattr(getattr(self.storage, name), field.name)[indices] = getattr(getattr(windows, name), field.name)
        for name in ("actions", "valid", "terminated"):
            getattr(self.storage, name)[indices] = getattr(windows, name)
        self.clock += count
        self.size = min(self.size + count, self.capacity)

    def sample(self, batch: int) -> Window:
        if self.storage is None or self.size == 0 or batch < 1:
            raise ValueError("Cannot sample an empty rare-event archive")
        indices = torch.randint(self.size, (batch,), device=self.storage.actions.device)
        return self.storage.index(indices)
