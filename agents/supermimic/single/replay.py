# Copyright (c) 2026-2027 zh
"""内容：保存真实终态并按抽取窗口局部构造历史、终止掩码及稀有事件归档。"""

from __future__ import annotations

from dataclasses import dataclass, fields

import torch
from torch import Tensor

from ..interfaces import TaskState


@dataclass
class Seed:
    state: TaskState
    history: Tensor


@dataclass
class Window:
    """批次优先；终止那一步有效，后续转移不参与同一条监督轨迹。"""

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
    """只检查被抽中的局部时间索引，不生成 capacity×num_envs 的全库掩码。"""

    def __init__(self, spec, num_envs: int, capacity: int, device):
        if capacity < spec.history_length + 1:
            raise ValueError("Replay capacity must exceed history length")
        self.spec, self.num_envs, self.capacity = spec, num_envs, capacity
        self.device = torch.device(device)
        self.size = self.clock = 0
        self.states = self._state_storage((capacity, num_envs))
        self.targets = self._state_storage((capacity, num_envs))
        self.actions = torch.zeros(capacity, num_envs, spec.action_dim, device=device)
        self.observations = torch.zeros(capacity, num_envs, spec.observation_dim, device=device, dtype=torch.float16)
        self.done = torch.zeros(capacity, num_envs, device=device, dtype=torch.bool)
        self.terminated = torch.zeros_like(self.done)
        self.ages = torch.zeros(capacity, num_envs, device=device, dtype=torch.long)

    def _state_storage(self, shape):
        widths = (self.spec.state_dim, self.spec.memory_dim, self.spec.command_dim, self.spec.context_dim)
        return TaskState(*(torch.zeros(*shape, width, device=self.device) for width in widths))

    def clear(self):
        self.size = self.clock = 0

    @torch.no_grad()
    def add(self, state: TaskState, observation: Tensor, action: Tensor, result, age: Tensor):
        slot = self.clock % self.capacity
        for field in fields(TaskState):
            getattr(self.states, field.name)[slot].copy_(getattr(state, field.name))
            getattr(self.targets, field.name)[slot].copy_(getattr(result.final_state, field.name))
        self.observations[slot].copy_(observation)
        self.actions[slot].copy_(action)
        self.done[slot].copy_(result.terminated | result.truncated)
        self.terminated[slot].copy_(result.terminated)
        self.ages[slot].copy_(age)
        self.clock += 1
        self.size = min(self.size + 1, self.capacity)

    def ready(self, horizon: int) -> bool:
        return self.size >= horizon + self.spec.history_length - 1

    def _sample_indices(self, batch: int, horizon: int):
        if batch < 1 or horizon < 1 or not self.ready(horizon):
            raise ValueError("Replay has insufficient history and future context for sampling")
        rows = torch.randint(self.num_envs, (batch,), device=self.device)
        logical = torch.randint(self.spec.history_length - 1, self.size - horizon + 1, (batch,), device=self.device)
        return rows, (self.clock - self.size + logical) % self.capacity

    def sample_seeds(self, batch: int) -> Seed:
        rows, slots = self._sample_indices(batch, 1)
        offsets = torch.arange(self.spec.history_length - 1, -1, -1, device=self.device)
        actual = torch.minimum(self.ages[slots, rows, None], offsets)
        history_slots = (slots[:, None] - actual) % self.capacity
        history = self.observations[history_slots, rows[:, None]].float()
        return Seed(self.states.index((slots, rows)), history)

    def sample_windows(self, batch: int, horizon: int) -> Window:
        rows, starts = self._sample_indices(batch, horizon)
        slots = (starts[:, None] + torch.arange(horizon, device=self.device)) % self.capacity
        done = self.done[slots, rows[:, None]]
        valid = torch.cat((torch.ones_like(done[:, :1]), (~done[:, :-1]).cumprod(dim=1).bool()), dim=1)
        return Window(self.states.index((starts, rows)), self.actions[slots, rows[:, None]],
                      self.targets.index((slots, rows[:, None])), valid, self.terminated[slots, rows[:, None]])


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
