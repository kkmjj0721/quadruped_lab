# Copyright (c) 2026-2027 zh
"""内容：保存重置前真实转移，用独立图像环和版本引用构造连续视觉窗口。"""

from __future__ import annotations

from dataclasses import fields

import torch

from .interfaces import Decision, EnvStep, TaskState, VisualObservation, VisualSeed, VisualWindow, validate_decision


class Replay:
    """状态与历史保留float32；每份带噪／干净深度仅存一次float16。"""

    def __init__(self, spec, num_envs: int, capacity: int, image_capacity: int, device):
        for name, value in (("num_envs", num_envs), ("capacity", capacity), ("image_capacity", image_capacity)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        self.spec, self.num_envs, self.capacity, self.image_capacity = spec, num_envs, capacity, image_capacity
        self.device = torch.device(device)
        self.size = self.clock = 0
        self._env = torch.arange(num_envs, device=self.device)
        self.device = self._env.device
        self.states = self._state_storage()
        self.targets = self._state_storage()
        self.actions = torch.zeros(capacity, num_envs, spec.action_dim, device=self.device)
        self.histories = torch.zeros(capacity, num_envs, spec.history_length, spec.observation_dim, device=self.device)
        self.contacts = torch.zeros(capacity, num_envs, spec.contact_dim, device=self.device)
        self.rewards = torch.zeros(capacity, num_envs, device=self.device)
        self.terminated = torch.zeros(capacity, num_envs, dtype=torch.bool, device=self.device)
        self.truncated = torch.zeros_like(self.terminated)
        self.done = torch.zeros_like(self.terminated)
        self.episodes = torch.zeros(capacity, num_envs, dtype=torch.int64, device=self.device)
        self.frames = torch.zeros_like(self.episodes)
        self.context_versions = torch.zeros_like(self.episodes)
        self.final_context_versions = torch.zeros_like(self.episodes)
        self.current_images = torch.full_like(self.episodes, -1)
        self.final_images = torch.full_like(self.episodes, -1)
        image_shape = (image_capacity, num_envs, *spec.depth_shape)
        self.images = torch.zeros(image_shape, dtype=torch.float16, device=self.device)
        self.clean_images = torch.zeros_like(self.images)
        self.image_versions = torch.full((image_capacity, num_envs), -1, dtype=torch.int64, device=self.device)
        self._image_clock = torch.zeros(num_envs, dtype=torch.int64, device=self.device)
        self._next_refs = torch.full_like(self._image_clock, -1)
        self._next_episodes = torch.zeros_like(self._image_clock)
        self._next_frames = torch.zeros_like(self._image_clock)
        self._has_next = False

    def _state_storage(self) -> TaskState:
        widths = (self.spec.state_dim, self.spec.memory_dim, self.spec.command_dim, self.spec.context_dim)
        return TaskState(*(torch.zeros(self.capacity, self.num_envs, width, device=self.device) for width in widths))

    @property
    def image_count(self) -> int:
        """当前仍占用的图像对数量，按环境分别计数。"""
        return int(self._image_clock.clamp(max=self.image_capacity).sum())

    @property
    def occupied_bytes(self) -> int:
        """预分配张量占用；不含Python对象和采样返回值。"""
        tensors = [value for value in vars(self).values() if isinstance(value, torch.Tensor)]
        tensors.extend(getattr(state, field.name) for state in (self.states, self.targets) for field in fields(TaskState))
        return sum(value.numel() * value.element_size() for value in tensors)

    def clear(self) -> None:
        """复用已分配内存，并使全部旧转移与图像引用失效。"""
        self.size = self.clock = 0
        self._image_clock.zero_()
        self.image_versions.fill_(-1)
        self._next_refs.fill_(-1)
        self._has_next = False

    def _validate_transition(self, current: Decision, history, action, result: EnvStep) -> None:
        for value in (current, result.final, result.next):
            validate_decision(value, self.spec, self.num_envs, self.device)
        entries = (("history", history, (self.num_envs, self.spec.history_length, self.spec.observation_dim), torch.float32),
                   ("action", action, (self.num_envs, self.spec.action_dim), torch.float32),
                   ("reward", result.reward, (self.num_envs,), torch.float32),
                   ("terminated", result.terminated, (self.num_envs,), torch.bool),
                   ("truncated", result.truncated, (self.num_envs,), torch.bool),
                   ("contacts", result.contacts, (self.num_envs, self.spec.contact_dim), torch.float32))
        for name, tensor, shape, dtype in entries:
            if tensor.shape != shape or tensor.dtype != dtype or tensor.device != self.device:
                raise ValueError(f"Invalid {name}: expected shape {shape}, dtype {dtype}, device {self.device}")
            if tensor.is_floating_point() and not torch.isfinite(tensor).all():
                raise FloatingPointError(f"Replay received non-finite {name}")
        if ((result.contacts < 0) | (result.contacts > 1)).any():
            raise ValueError("Contact targets must be within [0,1]")
        obs, final, next_obs = current.observation, result.final.observation, result.next.observation
        if not torch.equal(history[:, -1], obs.proprio):
            raise ValueError("The latest history frame must equal the current proprioception")
        if self._has_next and ((obs.episode_id != self._next_episodes) | (obs.frame_id != self._next_frames)).any():
            raise ValueError("Current frame must match the previous next decision; clear replay before restarting collection")
        if (final.episode_id != obs.episode_id).any() or (final.frame_id <= obs.frame_id).any():
            raise ValueError("A synchronous transition must advance to a new frame inside the same episode")
        done = result.terminated | result.truncated
        if (next_obs.episode_id[done] == final.episode_id[done]).any():
            raise ValueError("A reset decision must have a different episode ID")
        keep = ~done
        equal_values = [(getattr(final, field.name), getattr(next_obs, field.name)) for field in fields(VisualObservation)]
        equal_values.extend((getattr(result.final.state, field.name), getattr(result.next.state, field.name)) for field in fields(TaskState))
        equal_values.append((result.final.context_version, result.next.context_version))
        if any(not torch.equal(left[keep], right[keep]) for left, right in equal_values):
            raise ValueError("A continuing next decision must equal the pre-reset final decision")

    def _store_images(self, observation: VisualObservation, mask: torch.Tensor) -> torch.Tensor:
        env = self._env[mask]
        refs = self._image_clock[mask].clone()
        rows = refs % self.image_capacity
        self.images[rows, env] = observation.depth[mask].half()
        self.clean_images[rows, env] = observation.depth_clean[mask].half()
        self.image_versions[rows, env] = refs
        self._image_clock[mask] += 1
        return refs

    @torch.no_grad()
    def add(self, current: Decision, history, action, result: EnvStep) -> None:
        """统一检查外部转移，再保存当前与最终状态；终止目标始终早于reset。"""
        self._validate_transition(current, history, action, result)
        all_envs = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        current_refs = self._next_refs.clone() if self._has_next else self._store_images(current.observation, all_envs)
        final_refs = self._store_images(result.final.observation, all_envs)
        done = result.terminated | result.truncated
        next_refs = final_refs.clone()
        next_refs[done] = self._store_images(result.next.observation, done)
        slot = self.clock % self.capacity
        for field in fields(TaskState):
            getattr(self.states, field.name)[slot].copy_(getattr(current.state, field.name))
            getattr(self.targets, field.name)[slot].copy_(getattr(result.final.state, field.name))
        for destination, value in ((self.histories, history), (self.actions, action), (self.contacts, result.contacts),
                                   (self.rewards, result.reward), (self.terminated, result.terminated),
                                   (self.truncated, result.truncated), (self.done, done),
                                   (self.episodes, current.observation.episode_id), (self.frames, current.observation.frame_id),
                                   (self.context_versions, current.context_version),
                                   (self.final_context_versions, result.final.context_version),
                                   (self.current_images, current_refs), (self.final_images, final_refs)):
            destination[slot].copy_(value)
        self._next_refs.copy_(next_refs)
        self._next_episodes.copy_(result.next.observation.episode_id)
        self._next_frames.copy_(result.next.observation.frame_id)
        self._has_next = True
        self.clock += 1
        self.size = min(self.size + 1, self.capacity)

    def _image_alive(self, refs, env) -> torch.Tensor:
        return (refs >= 0) & (self.image_versions[refs.remainder(self.image_capacity), env] == refs)

    def _eligible_starts(self, window_length: int, *, require_final: bool = True) -> torch.Tensor:
        if isinstance(window_length, bool) or not isinstance(window_length, int) or window_length < 1:
            raise ValueError("Window length must be a positive integer")
        if self.size == 0 or window_length > self.capacity:
            return torch.empty(0, 2, dtype=torch.int64, device=self.device)
        logical = torch.arange(self.size, device=self.device)
        rows = (self.clock - self.size + logical) % self.capacity
        current, final = self.current_images[rows], self.final_images[rows]
        eligible = self._image_alive(current, self._env)
        if require_final:
            eligible &= self._image_alive(final, self._env)
            eligible &= self.context_versions[rows] == self.final_context_versions[rows]
            tail = logical[logical + window_length > self.size]
            if tail.numel():
                times = tail[:, None] + torch.arange(window_length, device=self.device)
                available = times < self.size
                tail_rows = (self.clock - self.size + times.clamp(max=self.size - 1)) % self.capacity
                versions = self.context_versions[tail_rows]
                stops = self.done[tail_rows] | (versions != self.final_context_versions[tail_rows])
                # 仅已观察到的终止或上下文断点可以截短；不把尚无未来记录视为终止。
                stops[:, 1:] |= versions[:, 1:] != versions[:, :-1]
                eligible[tail] &= (stops & available.unsqueeze(-1)).any(dim=1)
        starts = eligible.nonzero(as_tuple=False)
        starts[:, 0] = rows[starts[:, 0]]
        return starts

    def ready(self, window_length: int) -> bool:
        return self._eligible_starts(window_length).shape[0] > 0

    def _choose_starts(self, batch: int, window_length: int, *, require_final: bool = True) -> torch.Tensor:
        if isinstance(batch, bool) or not isinstance(batch, int) or batch < 1:
            raise ValueError("Replay batch must be a positive integer")
        candidates = self._eligible_starts(window_length, require_final=require_final)
        if candidates.shape[0] == 0:
            raise ValueError("Replay has no valid starts with live image references for the requested window")
        return candidates[torch.randint(candidates.shape[0], (batch,), device=self.device)]

    def _read_images(self, refs, env, valid, *, clean: bool = False):
        output = torch.zeros((*refs.shape, *self.spec.depth_shape), dtype=torch.float32, device=self.device)
        env = torch.broadcast_to(env, refs.shape)
        source = self.clean_images if clean else self.images
        output[valid] = source[refs[valid] % self.image_capacity, env[valid]].float()
        return output

    @torch.no_grad()
    def sample_seeds(self, batch: int) -> VisualSeed:
        selected = self._choose_starts(batch, 1, require_final=False)
        rows, env = selected[:, 0], selected[:, 1]
        refs = self.current_images[rows, env]
        valid = torch.ones(batch, dtype=torch.bool, device=self.device)
        return VisualSeed(self.states.index((rows, env)), self.histories[rows, env],
                          self._read_images(refs, env, valid))

    @torch.no_grad()
    def sample_windows(self, batch: int, window_length: int) -> VisualWindow:
        selected = self._choose_starts(batch, window_length)
        logical = (selected[:, :1] - (self.clock - self.size)) % self.capacity
        times = logical + torch.arange(window_length, device=self.device)
        available = times < self.size
        rows = (self.clock - self.size + times.clamp(max=self.size - 1)) % self.capacity
        env = selected[:, 1:]
        current, final = self.current_images[rows, env], self.final_images[rows, env]
        valid = available & self._image_alive(current, env) & self._image_alive(final, env)
        versions, episodes = self.context_versions[rows, env], self.episodes[rows, env]
        valid &= versions == self.final_context_versions[rows, env]
        linked = (versions[:, 1:] == versions[:, :-1]) & (episodes[:, 1:] == episodes[:, :-1])
        linked &= current[:, 1:] == final[:, :-1]
        linked &= ~self.done[rows[:, :-1], env]
        valid[:, 1:] &= linked
        valid = valid.cumprod(dim=1).bool()
        depth = self._read_images(current[:, 0], selected[:, 1], valid[:, 0])
        states, targets = self.states.index((rows, env)), self.targets.index((rows, env))
        # 缺少未来记录时以已知最后final填充状态，保留任务可能要求的合法旋转表示。
        for field in fields(TaskState):
            value = getattr(states, field.name)
            value.copy_(torch.where(available.unsqueeze(-1), value, getattr(targets, field.name)))
        actions = self.actions[rows, env] * available.unsqueeze(-1)
        contacts = self.contacts[rows, env] * available.unsqueeze(-1)
        return VisualWindow(states, actions, targets, contacts, valid, depth,
                            self._read_images(final, env, valid), self._read_images(final, env, valid, clean=True))
