# Copyright (c) 2026-2027 zh
"""
内容：
    保存逐环境真实转移，恢复观测历史并采样不跨回合的连续窗口。
"""

from __future__ import annotations

from dataclasses import fields

import torch
from torch import Tensor

from agents.supermimic.interfaces import (
    AlgorithmSpec, PolicyInput, ReplayWindow, TaskState, Transition, validate_policy_input,
)


class ReplayBuffer:
    """capacity为每个环境保留的时间步数，写入时复制且不保留计算图。"""

    def __init__(self, spec: AlgorithmSpec, num_envs: int, capacity: int, device: str | torch.device = "cpu", *, privileged_fn=None):
        if num_envs <= 0 or capacity < spec.history_length:
            raise ValueError("Replay requires positive environments and capacity at least history_length")
        self.spec = spec
        self.num_envs = num_envs
        self.capacity = capacity
        self.device = torch.empty(0, device=device).device
        shape = (capacity, num_envs)
        widths = {"physical": spec.state_dim, "memory": spec.memory_dim,
                  "command": spec.command_dim, "context": spec.context_dim}
        self.current_states = TaskState(**{name: torch.empty(*shape, width, dtype=torch.float32, device=self.device)
                                          for name, width in widths.items()})
        self.next_states = TaskState(**{name: torch.empty(*shape, width, dtype=torch.float32, device=self.device)
                                       for name, width in widths.items()})
        self.observations = torch.empty(*shape, spec.observation_dim, dtype=torch.float32, device=self.device)
        # 纯解析特权量可从保存状态重建，避免扩大奖励状态后再重复存储同一数据。
        self.privileged_fn = privileged_fn
        self.privileged = (torch.empty(*shape, spec.privileged_dim, dtype=torch.float32, device=self.device)
                           if privileged_fn is None else None)
        self.actions = torch.empty(*shape, spec.action_dim, dtype=torch.float32, device=self.device)
        self.rewards = torch.empty(shape, dtype=torch.float32, device=self.device)
        self.terminated = torch.empty(shape, dtype=torch.bool, device=self.device)
        self.truncated = torch.empty(shape, dtype=torch.bool, device=self.device)
        self.branches = torch.empty(shape, dtype=torch.bool, device=self.device)
        self.episodes = torch.empty(shape, dtype=torch.long, device=self.device)
        self.steps = torch.empty(shape, dtype=torch.long, device=self.device)
        self.context_versions = torch.empty(shape, dtype=torch.long, device=self.device)
        self.stamps = torch.full((capacity,), -1, dtype=torch.long, device=self.device)
        self._clock = 0
        self._size = 0

    def __len__(self) -> int:
        return self._size * self.num_envs

    def clear(self) -> None:
        self._clock = 0
        self._size = 0
        self.stamps.fill_(-1)

    def _validate(self, item: Transition) -> None:
        validate_policy_input(item.current, self.spec)
        validate_policy_input(item.next, self.spec)
        if item.current.observation.shape[0] != self.num_envs or item.next.observation.shape[0] != self.num_envs:
            raise ValueError("Transition must include every fixed environment row")
        if item.current.observation.device != self.device or item.next.observation.device != self.device:
            raise ValueError("Transition and replay must use one device")
        if not torch.equal(item.current.branch, item.next.branch):
            raise ValueError("A transition cannot change behavior branch")
        floating = [(item.action, (self.num_envs, self.spec.action_dim)), (item.reward, (self.num_envs,))]
        for tensor, shape in floating:
            if tensor.shape != shape or tensor.dtype != torch.float32 or tensor.device != self.device:
                raise ValueError("Invalid transition action or reward")
        for tensor in (item.terminated, item.truncated):
            if tensor.shape != (self.num_envs,) or tensor.dtype != torch.bool or tensor.device != self.device:
                raise ValueError("Termination flags must be boolean environment vectors")
        for tensor in (item.episode_id, item.step_id, item.context_version):
            if tensor.shape != (self.num_envs,) or tensor.dtype != torch.long or tensor.device != self.device or (tensor < 0).any():
                raise ValueError("Episode, step and context versions must be nonnegative int64 vectors")
        for data in (item.current, item.next):
            floating.extend((getattr(data.state, field.name), None) for field in fields(TaskState))
            floating.extend(((data.observation, None), (data.history, None), (data.privileged, None)))
            if not torch.equal(data.history[:, -1], data.observation):
                raise ValueError("History must include the current observation exactly once")
        if any(not torch.isfinite(tensor).all() for tensor, _ in floating):
            raise ValueError("Replay cannot store nonfinite transition data")

    @torch.no_grad()
    def add(self, item: Transition) -> None:
        self._validate(item)
        slot = self._clock % self.capacity
        for field in fields(TaskState):
            getattr(self.current_states, field.name)[slot].copy_(getattr(item.current.state, field.name))
            getattr(self.next_states, field.name)[slot].copy_(getattr(item.next.state, field.name))
        pairs = ((self.observations, item.current.observation),
                 (self.actions, item.action), (self.rewards, item.reward), (self.terminated, item.terminated),
                 (self.truncated, item.truncated), (self.branches, item.current.branch), (self.episodes, item.episode_id),
                 (self.steps, item.step_id), (self.context_versions, item.context_version))
        for destination, source in pairs:
            destination[slot].copy_(source)
        if self.privileged is not None:
            self.privileged[slot].copy_(item.current.privileged)
        self.stamps[slot] = self._clock
        self._clock += 1
        self._size = min(self.capacity, self._size + 1)

    def _history_mask(self) -> Tensor:
        """覆盖后缺少真实前缀的起点不被误当作reset起点。"""
        valid = (self.stamps >= 0).unsqueeze(-1).expand(-1, self.num_envs).clone()
        slots = torch.arange(self.capacity, device=self.device)
        done = self.terminated | self.truncated
        for offset in range(1, self.spec.history_length):
            previous = (slots - offset) % self.capacity
            needed = self.steps >= offset
            continuous = ((self.stamps[previous] == self.stamps - offset).unsqueeze(-1)
                          & (self.episodes[previous] == self.episodes)
                          & (self.steps[previous] == self.steps - offset) & ~done[previous])
            valid &= ~needed | continuous
        return valid

    def _window_mask(self, horizon: int) -> Tensor:
        if horizon <= 0:
            raise ValueError("Window horizon must be positive")
        valid = self._history_mask()
        alive = valid.clone()
        slots = torch.arange(self.capacity, device=self.device)
        done = self.terminated | self.truncated
        for offset in range(horizon):
            current = (slots + offset) % self.capacity
            continuous = ((self.stamps[current] == self.stamps + offset).unsqueeze(-1)
                          & (self.episodes[current] == self.episodes)
                          & (self.steps[current] == self.steps + offset)
                          & (self.context_versions[current] == self.context_versions)
                          & (self.branches[current] == self.branches))
            valid &= ~alive | continuous
            alive &= continuous & ~done[current]
        return valid

    def can_sample(self, horizon: int = 1) -> bool:
        return bool(self._window_mask(horizon).any())

    def _choose(self, mask: Tensor, count: int) -> tuple[Tensor, Tensor]:
        if count <= 0:
            raise ValueError("Sample count must be positive")
        choices = torch.nonzero(mask, as_tuple=False)
        if choices.shape[0] == 0:
            raise ValueError("No replay samples satisfy the continuity requirements")
        picked = choices[torch.randint(choices.shape[0], (count,), device=self.device)]
        return picked[:, 0], picked[:, 1]

    def _policy_input(self, slots: Tensor, environments: Tensor) -> PolicyInput:
        history = []
        for offset in reversed(range(self.spec.history_length)):
            actual_offset = torch.minimum(self.steps[slots, environments], torch.full_like(slots, offset))
            history.append(self.observations[(slots - actual_offset) % self.capacity, environments])
        state = self.current_states.map(lambda tensor: tensor[slots, environments].clone())
        privileged = (self.privileged_fn(state) if self.privileged_fn is not None
                      else self.privileged[slots, environments].clone())
        return PolicyInput(state, self.observations[slots, environments].clone(), torch.stack(history, dim=1),
                           privileged, self.branches[slots, environments].clone())

    def sample_seeds(self, batch_size: int, teacher_fraction: float = 0.75) -> PolicyInput:
        """按两组分别有放回采样；比例不改变后续组等权损失。"""
        if not 0 <= teacher_fraction <= 1 or batch_size <= 0:
            raise ValueError("Invalid imagination sample configuration")
        teacher_count = round(batch_size * teacher_fraction)
        if 0 < teacher_fraction < 1:
            if batch_size < 2:
                raise ValueError("Dual-branch sampling requires at least two samples")
            teacher_count = max(1, min(batch_size - 1, teacher_count))
        valid = self._history_mask()
        pairs = []
        for branch, count in ((True, teacher_count), (False, batch_size - teacher_count)):
            if count:
                pairs.append(self._choose(valid & (self.branches == branch), count))
        slots = torch.cat([pair[0] for pair in pairs])
        environments = torch.cat([pair[1] for pair in pairs])
        return self._policy_input(slots, environments)

    def sample_students(self, batch_size: int) -> PolicyInput | None:
        if batch_size <= 0:
            raise ValueError("Sample count must be positive")
        valid = self._history_mask() & ~self.branches
        if not valid.any():
            return None
        return self._policy_input(*self._choose(valid, batch_size))

    def sample_windows(self, batch_size: int, horizon: int) -> ReplayWindow:
        slots, environments = self._choose(self._window_mask(horizon), batch_size)
        initial = self._policy_input(slots, environments)
        offsets = torch.arange(horizon, device=self.device).unsqueeze(-1)
        window_slots = (slots.unsqueeze(0) + offsets) % self.capacity
        rows = environments.unsqueeze(0).expand(horizon, -1)
        terminated = self.terminated[window_slots, rows].clone()
        truncated = self.truncated[window_slots, rows].clone()
        alive = torch.ones(batch_size, dtype=torch.bool, device=self.device)
        masks = []
        for index in range(horizon):
            masks.append(alive.clone())
            alive &= ~(terminated[index] | truncated[index])
        valid = torch.stack(masks)
        actions = self.actions[window_slots, rows].clone()
        rewards = self.rewards[window_slots, rows].clone()
        next_states = self.next_states.map(lambda tensor: tensor[window_slots, rows].clone())
        # 无效补齐显式归零，不能把下一回合或未初始化值带入网络。
        actions.masked_fill_(~valid.unsqueeze(-1), 0)
        rewards.masked_fill_(~valid, 0)
        terminated &= valid
        truncated &= valid
        next_states = next_states.map(lambda tensor: tensor.masked_fill(~valid.unsqueeze(-1), 0))
        return ReplayWindow(initial, actions, next_states, rewards, terminated, truncated, valid)
