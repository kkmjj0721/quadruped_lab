# Copyright (c) 2026-2027 zh
"""
内容：
    保存终止前双帧深度、配对干净标签和推理前 GRU 状态，支持本体单帧压缩及历史重采噪基线。
    回放不缓存视觉潜变量；每次监督均由当前编码器重新编码图像。
"""

from dataclasses import fields

import torch
from torch.nn import functional as F

from agents.supermimic.interfaces import TaskState

from .interfaces import DepthSeed, DepthWindow


class DepthReplay:
    """深度保持 uint8 完整栈；逻辑年龄 0 表示当前保存的最旧转移。

    frames 模式要求连续控制步插入实际本体历史，按结束标志限制后续历史回退。
    首次插入和清空后的首次插入保留传入的完整前缀；单帧按 Go2 协议存为 FP16。
    """

    def __init__(self, task, num_envs, capacity, device, *, holdout_envs=0.1, device_sampling=False, history_storage="full"):
        if history_storage not in ("full", "frames"):
            raise ValueError("History storage must be full or frames")
        self.task, self.spec = task, task.spec
        self.num_envs, self.capacity, self.device = num_envs, capacity, torch.device(device)
        self.history_storage = history_storage
        self.device_sampling = device_sampling
        if device_sampling:
            self.ptr_device = torch.zeros((), dtype=torch.long, device=device)
            self.size_device = torch.zeros_like(self.ptr_device)
            self.age_indices = torch.arange(capacity, device=device)
        spec = self.spec
        shape = (capacity, num_envs)
        self.states = self._state_storage(shape)
        self.targets = self._state_storage(shape)
        self.actions = torch.zeros(*shape, spec.action_dim, device=device)
        self.contacts = torch.zeros(*shape, spec.contact_dim, device=device)
        self.depth = torch.zeros(*shape, *spec.depth_shape, dtype=torch.uint8, device=device)
        self.next_depth = torch.zeros_like(self.depth)
        self.next_clean_depth = torch.zeros_like(self.depth)
        self.hidden = torch.zeros(*shape, spec.latent_dim, device=device)
        self.done = torch.zeros(*shape, dtype=torch.bool, device=device)
        self.terminated = torch.zeros_like(self.done)
        self.episode_steps = torch.zeros(*shape, dtype=torch.int32, device=device)
        self.time_left = torch.zeros(*shape, device=device)
        if history_storage == "frames":
            self.frame_capacity = capacity + spec.history_length - 1
            self.observation_frames = torch.zeros(self.frame_capacity, num_envs, spec.observation_dim, dtype=torch.float16, device=device)
            self.history_depth = torch.zeros(*shape, dtype=torch.int32, device=device)
            self.next_history_depth = torch.full((num_envs,), spec.history_length, dtype=torch.int32, device=device)
            self.history_offsets = torch.arange(spec.history_length - 1, -1, -1, device=device)
            if device_sampling:
                self.clock_device = torch.zeros((), dtype=torch.long, device=device)
        count = int(num_envs * holdout_envs)
        self.train_envs = (0, num_envs - count)
        self.holdout_envs = (num_envs - count, num_envs) if count else (0, num_envs)
        self.ptr = self.size = self.clock = 0

    def _state_storage(self, shape):
        spec = self.spec
        return TaskState(*(torch.zeros(*shape, width, device=self.device)
                           for width in (spec.state_dim, spec.memory_dim, spec.command_dim, spec.context_dim)))

    def clear(self):
        self.ptr = self.size = self.clock = 0
        if self.device_sampling:
            self.ptr_device.zero_()
            self.size_device.zero_()
        if self.history_storage == "frames":
            self.history_depth.zero_()
            self.next_history_depth.fill_(self.spec.history_length)
            if self.device_sampling:
                self.clock_device.zero_()

    @torch.no_grad()
    def add(self, state, depth, action, result, episode_steps, *, history=None, hidden):
        if self.history_storage == "frames" and (history is None or history.shape != (self.num_envs, self.spec.history_length, self.spec.observation_dim)):
            raise ValueError("Frames history storage requires the collected proprioceptive history [N,H,D]")
        if hidden.shape != (self.num_envs, self.spec.latent_dim):
            raise ValueError("Visual hidden state must match the collected pre-action state [N,64]")
        for field in fields(TaskState):
            getattr(self.states, field.name)[self.ptr].copy_(getattr(state, field.name))
            getattr(self.targets, field.name)[self.ptr].copy_(getattr(result.final_state, field.name))
        self.actions[self.ptr].copy_(action)
        self.contacts[self.ptr].copy_(self.task.contact(result.final_state))
        self.depth[self.ptr].copy_((depth * 255).round().to(torch.uint8))
        self.next_depth[self.ptr].copy_((result.final_depth * 255).round().to(torch.uint8))
        self.next_clean_depth[self.ptr].copy_((result.final_clean_depth * 255).round().to(torch.uint8))
        self.hidden[self.ptr].copy_(hidden)
        self.done[self.ptr].copy_(result.terminated | result.truncated)
        self.terminated[self.ptr].copy_(result.terminated)
        self.episode_steps[self.ptr].copy_(episode_steps)
        self.time_left[self.ptr].copy_(self.task.time_left(state))
        if self.history_storage == "frames":
            if self.clock == 0:
                # 负逻辑时刻的历史前缀位于单帧环末尾，不受物理转移环的覆盖影响。
                self.observation_frames[self.capacity:].copy_(history[:, :-1].transpose(0, 1))
            self.observation_frames[self.clock % self.frame_capacity].copy_(history[:, -1])
            self.history_depth[self.ptr].copy_(self.next_history_depth)
            self.next_history_depth.copy_(torch.where(self.done[self.ptr], 1, (self.next_history_depth + 1).clamp_max(self.spec.history_length)))
            if self.device_sampling:
                self.clock_device.add_(1)
        self.clock += 1
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        if self.device_sampling:
            self.ptr_device.add_(1).remainder_(self.capacity)
            self.size_device.add_(1).clamp_max_(self.capacity)

    def _row(self, age, *, device_indices=False):
        pointer, size = (self.ptr_device, self.size_device) if device_indices else (self.ptr, self.size)
        return (pointer - size + age) % self.capacity

    def _state_at(self, storage, rows, envs):
        return storage.map(lambda value: value[rows, envs])

    def sample_windows(self, batch, length, *, env_range=None, failure_frac=0.0, failure_horizon=16, device_indices=False):
        lo, hi = self.train_envs if env_range is None else env_range
        if device_indices and not self.device_sampling:
            raise ValueError("Device indices require device_sampling=True")
        size = self.size_device if device_indices else self.size
        windows, environments = size - length + 1, hi - lo
        if length < 1 or length > self.capacity or batch < 1 or environments <= 0:
            raise ValueError("Replay has insufficient transitions or environments for this window")
        if device_indices:
            torch._assert_async(windows > 0, "Replay has insufficient transitions for this window")
            # 与 Go2 一样在设备上量化动态上界，重放时不冻结 Python 的 size。
            age = (torch.rand(batch, dtype=torch.float64, device=self.device) * windows).long().clamp_max(windows - 1)
        else:
            if windows <= 0:
                raise ValueError("Replay has insufficient transitions or environments for this window")
            age = torch.randint(windows, (batch,), device=self.device)
        env = torch.randint(lo, hi, (batch,), device=self.device)
        weight = torch.ones(batch, device=self.device)
        failure_count = int(batch * failure_frac)
        if failure_count:
            ages = self.age_indices if device_indices else torch.arange(self.size, device=self.device)
            rows = self._row(ages, device_indices=device_indices)
            terminated = self.terminated[rows, lo:hi].int()
            cumulative = F.pad(terminated.cumsum(0), (0, 0, 1, 0))
            starts = self.age_indices[:self.capacity - length + 1] if device_indices else torch.arange(windows, device=self.device)
            ends = (starts + failure_horizon).clamp_max(size)
            active = starts[:, None] < windows
            failed = (cumulative[ends] - cumulative[starts] > 0) & active
            count = failed.sum()
            probabilities = torch.where(count > 0, failed.float(), active.expand_as(failed).float()).flatten()
            picked = torch.multinomial(probabilities, failure_count, replacement=True)
            age[:failure_count] = picked // environments
            env[:failure_count] = picked % environments + lo
            permutation = torch.randperm(batch, device=self.device)
            age, env = age[permutation], env[permutation]
            share = (failure_count / batch) * (count > 0).float()
            # 固定 capacity 的占位行不属于目标总体，权重仍按实际有效窗口数量计算。
            population = windows * environments
            weight = torch.where(failed[age, env - lo], 1 / ((1 - share) + share * population / count.clamp_min(1)), 1 / (1 - share))
        rows = self._row(age[:, None] + torch.arange(length, device=self.device), device_indices=device_indices)
        env_rows = env[:, None].expand_as(rows)
        done = self.done[rows, env_rows].float()
        valid = torch.cumprod(torch.cat((torch.ones(batch, 1, device=self.device), 1 - done[:, :-1]), -1), -1)
        return DepthWindow(self._state_at(self.states, rows, env_rows), self._state_at(self.targets, rows, env_rows),
                           self.actions[rows, env_rows], self.contacts[rows, env_rows], self.depth[rows, env_rows],
                           self.next_depth[rows, env_rows], valid, self.terminated[rows, env_rows], weight, env,
                           self.next_clean_depth[rows, env_rows])

    def starts_at(self, age, env, *, device_indices=False):
        rows = self._row(age, device_indices=device_indices)
        if self.history_storage == "frames":
            clock, size = (self.clock_device, self.size_device) if device_indices else (self.clock, self.size)
            absolute = clock - size + age
            depth = self.history_depth[rows, env]
            back = torch.minimum(self.history_offsets[None], depth[:, None] - 1)
            slots = (absolute[:, None] - back) % self.frame_capacity
            history = self.observation_frames[slots, env[:, None]].float()
        else:
            episode_step = self.episode_steps[rows, env].long()
            back = torch.arange(self.spec.history_length - 1, -1, -1, device=self.device)
            back = torch.minimum(back[None], episode_step[:, None])
            back = torch.minimum(back, age[:, None])
            history_rows = self._row(age[:, None] - back, device_indices=device_indices)
            history_env = env[:, None].expand_as(history_rows)
            states = self._state_at(self.states, history_rows, history_env)
            flat_states = states.map(lambda value: value.flatten(0, 1))
            plain = self.task.observe(flat_states, noisy=False).reshape(len(age), self.spec.history_length, -1)
            noise = self.task.observe(flat_states, noisy=True).reshape_as(plain) - plain
            # reset 或回放边界补齐的相同逻辑帧必须复用同一次噪声采样。
            slot = (back[:, :, None] == back[:, None, :]).float().argmax(-1)
            history = plain + noise.gather(1, slot[..., None].expand_as(noise))
        state = self._state_at(self.states, rows, env)
        return DepthSeed(state, history, self.depth[rows, env].float() / 255, self.time_left[rows, env], env,
                         self.hidden[rows, env])

    def sample_starts(self, batch, horizon, margin_steps, *, device_indices=False):
        if device_indices and not self.device_sampling:
            raise ValueError("Device indices require device_sampling=True")
        ages = self.age_indices if device_indices else torch.arange(self.size, device=self.device)
        rows = self._row(ages, device_indices=device_indices)
        eligible = self.time_left[rows] > (horizon + margin_steps) * self.task.dt
        if device_indices:
            eligible = eligible & (ages[:, None] < self.size_device)
            torch._assert_async(eligible.any(), "No buffered command stays for the complete imagination horizon")
        elif not eligible.any():
            raise RuntimeError("No buffered command stays for the complete imagination horizon")
        picked = torch.multinomial(eligible.flatten().float(), batch, replacement=True)
        return self.starts_at(picked // self.num_envs, picked % self.num_envs, device_indices=device_indices)
