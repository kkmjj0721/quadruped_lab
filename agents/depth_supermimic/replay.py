# Copyright (c) 2026-2027 zh
"""内容：组合原生物理转移、四帧引用、实际动作环和有限未来感知监督。"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .buffer import Replay
from .frame_storage import FrameStorage
from .interfaces import DepthEnvStep, DepthFrame, TaskState
from .supervision import project_read_targets, strictly_future_contact_indices


@dataclass
class RawDepthSeed:
    """像素在CPU；其余字段在回放设备，编码时使用当前CNN而非旧token。"""

    state: TaskState
    history: Tensor
    images: Tensor
    action_history: Tensor
    age: Tensor
    future_xy: Tensor
    future_valid: Tensor
    read_targets: Tensor
    read_valid: Tensor

    def index(self, index):
        host_index = index.detach().cpu() if isinstance(index, Tensor) else index
        return RawDepthSeed(self.state.index(index), self.history[index], self.images[host_index],
                            self.action_history[index], self.age[index], self.future_xy[index],
                            self.future_valid[index], self.read_targets[index], self.read_valid[index])


@dataclass
class RawDepthAuxiliary:
    """固定八次刷新；不足部分保持形状并由refresh_valid显式屏蔽。"""

    seed: RawDepthSeed
    refresh_states: TaskState
    refresh_actions: Tensor
    refresh_images: Tensor
    refresh_valid: Tensor
    reconstruction_clean: Tensor
    reconstruction_valid: Tensor


class DepthReplay(Replay):
    """GPU保存结构化物理数据，CPU帧池保存实际新帧，环覆盖及时归还所有权。

    四帧历史和原生final新帧均用ID引用。非终止next_frame复用final帧，
    重置帧独立记录；每步重置且final也刷新时仍有明确的硬容量上界。
    """

    def __init__(self, spec, num_envs: int, capacity: int, device, task, history_storage="frames", *,
                 device_sampling=False, depth_shape=(1, 64, 64), frame_window=4,
                 action_history_length=5, depth_scale=5.0):
        if frame_window != 4 or action_history_length != 5 or depth_scale <= 0:
            raise ValueError("Depth replay requires four frames, five actions and a positive depth scale")
        super().__init__(spec, num_envs, capacity, device, history_storage, device_sampling=device_sampling)
        self.task, self.frame_window = task, frame_window
        self.action_history_length, self.depth_scale = action_history_length, depth_scale
        self.frames = FrameStorage(num_envs * (2 * capacity + frame_window + 2), depth_shape)
        self.frame_ids = torch.full((capacity, num_envs, frame_window), -1, dtype=torch.long)
        self.refresh_ids = torch.full((capacity, num_envs), -1, dtype=torch.long)
        self.host_done = torch.zeros(capacity, num_envs, dtype=torch.bool)
        self.onsets = torch.zeros(capacity, num_envs, 4, dtype=torch.bool, device=self.device)
        self.foot_positions = torch.zeros(capacity, num_envs, 4, 3, device=self.device)
        self.ages = torch.zeros(capacity, num_envs, dtype=torch.uint8, device=device)
        self.action_frame_capacity = capacity + action_history_length
        self.executed_frames = torch.zeros(self.action_frame_capacity, num_envs, spec.action_dim,
                                           dtype=torch.float16, device=device)
        self.action_depths = torch.zeros(capacity, num_envs, dtype=torch.uint8, device=device)
        self._next_action_depth = torch.full((num_envs,), action_history_length, dtype=torch.uint8, device=device)
        self._live_refs = None

    def clear(self):
        super().clear()
        self.frames.clear()
        self.frame_ids.fill_(-1)
        self.refresh_ids.fill_(-1)
        self.host_done.zero_()
        self.onsets.zero_()
        self.action_depths.zero_()
        self._next_action_depth.fill_(self.action_history_length)
        self._live_refs = None

    def _initial_frames(self, frame):
        ids = self.frames.append(frame, torch.arange(self.num_envs))
        self._live_refs = ids[:, None].expand(-1, self.frame_window).clone()
        self.frames.retain(self._live_refs)
        self.frames.release(ids)

    def _advance_frames(self, result, done, final_fresh, next_fresh, final_ids):
        updated = self._live_refs.clone()
        reused = (~done & next_fresh & final_fresh).nonzero().flatten()
        if reused.numel():
            updated[reused, :-1] = updated[reused, 1:].clone()
            updated[reused, -1] = final_ids[reused]
        new_rows = (done | (~done & next_fresh & ~final_fresh)).nonzero().flatten()
        ids = self.frames.append(result.next_frame, new_rows)
        if new_rows.numel():
            reset = done[new_rows]
            updated[new_rows, :-1] = torch.where(reset[:, None], ids[:, None], updated[new_rows, 1:])
            updated[new_rows, -1] = ids
        self.frames.retain(updated)
        self.frames.release(self._live_refs)
        self.frames.release(ids)
        self._live_refs = updated

    @torch.no_grad()
    def add(self, state: TaskState, history: Tensor, action: Tensor, result: DepthEnvStep, *,
            frame: DepthFrame, action_history: Tensor, age: Tensor):
        tick, slot = self.clock, self.clock % self.capacity
        if self.size == self.capacity:
            self.frames.release(self.frame_ids[slot])
            self.frames.release(self.refresh_ids[slot])
        if self._live_refs is None:
            self._initial_frames(frame)
        self.frame_ids[slot].copy_(self._live_refs)
        self.frames.retain(self.frame_ids[slot])
        done_device = result.terminated | result.truncated
        onset = (self.task.contacts(state) <= .5) & (result.contacts > .5)
        self.onsets[slot].copy_(onset)
        self.foot_positions[slot].copy_(result.foot_positions)
        # 触地索引与位置留在物理回放设备；主机只读取相机引用管理所需的三种标志。
        packed = torch.cat((done_device[:, None].float(),
                            result.final_frame.fresh[:, None].float(), result.next_frame.fresh[:, None].float()), 1).cpu()
        done, final_fresh, next_fresh = packed[:, 0].bool(), packed[:, 1].bool(), packed[:, 2].bool()
        self.host_done[slot].copy_(done)
        fresh_rows = final_fresh.nonzero().flatten()
        final_ids = torch.full((self.num_envs,), -1, dtype=torch.long)
        final_ids[fresh_rows] = self.frames.append(result.final_frame, fresh_rows)
        self.refresh_ids[slot].copy_(final_ids)
        self._advance_frames(result, done, final_fresh, next_fresh, final_ids)
        if tick == 0:
            self.executed_frames[self.capacity:].copy_(action_history.transpose(0, 1))
        self.executed_frames[tick % self.action_frame_capacity].copy_(result.executed_action)
        self.action_depths[slot].copy_(self._next_action_depth)
        self._next_action_depth.copy_(torch.where(done_device, 0, (self._next_action_depth + 1).clamp_max(5)))
        self.ages[slot].copy_(age.reshape(self.num_envs).to(torch.uint8))
        super().add(state, history, action, result)

    def _history_at(self, rows, absolute):
        slots = absolute % self.capacity
        if self.history_storage == "full":
            return self.histories[slots, rows].float()
        depth = self.history_depth[slots, rows]
        back = torch.minimum(self.history_offsets[None], depth[:, None] - 1)
        return self.observation_frames[(absolute[:, None] - back) % self.frame_capacity, rows[:, None]].float()

    def _actions_at(self, rows, absolute, *, after=False):
        """after=True返回该native final之前的五动作；否则返回决策前动作环。"""
        depth = self.action_depths[absolute % self.capacity, rows].long() + int(after)
        offsets = torch.arange(5, 0, -1, device=self.device) - int(after)
        frames = self.executed_frames[(absolute[..., None] - offsets) % self.action_frame_capacity,
                                      rows[..., None]].float()
        return torch.where((offsets < depth[..., None] + int(not after))[..., None], frames, 0)

    def _future_at(self, rows, absolute):
        """在物理回放设备上找首次事件，只返回B×4世界足点和有效位。"""
        rows, absolute = rows.to(self.device), absolute.to(self.device)
        future = absolute[:, None] + torch.arange(self.size, device=self.device)[None]
        available = future < self.clock
        slots = future % self.capacity
        first, valid = strictly_future_contact_indices(self.onsets[slots, rows[:, None]],
                                                       self.done[slots, rows[:, None]], available)
        selected = (absolute[:, None] + first) % self.capacity
        feet = torch.arange(self.onsets.shape[-1], device=self.device)
        points = self.foot_positions[selected, rows[:, None], feet[None]]
        return torch.where(valid[..., None], points, 0), valid

    def _raw_at(self, rows, absolute, *, auxiliary=False):
        slots = absolute % self.capacity
        state = self.states.index((slots, rows))
        frame_ids = self.frame_ids[slots.cpu(), rows.cpu()]
        requested = ("noisy", "clean", "valid") if auxiliary else ("noisy", "world_to_camera", "intrinsics")
        images = self.frames.gather(frame_ids, fields=requested)
        future_world, future_valid = self._future_at(rows, absolute)
        future_xy = self.task.foothold_xy(state, future_world)
        future_xy = torch.where(future_valid[..., None], future_xy, 0)
        if auxiliary:
            # 辅助优化不消费read监督；保留种子固定协议，不查询像素或生成空用标签。
            read_targets = state.physical.new_zeros(rows.shape[0], 4, self.frame_window * 16)
            read_valid = torch.zeros(rows.shape[0], 4, dtype=torch.bool, device=self.device)
        else:
            read_targets, read_valid = project_read_targets(future_world, future_valid, self.frames, frame_ids,
                                                           images, depth_scale=self.depth_scale)
            read_targets, read_valid = read_targets.to(self.device), read_valid.to(self.device)
        seed = RawDepthSeed(state, self._history_at(rows, absolute), images.noisy,
                            self._actions_at(rows, absolute), self.ages[slots, rows, None].float(),
                            future_xy, future_valid, read_targets, read_valid)
        return seed, images

    def seed_at(self, rows: Tensor, absolute: Tensor) -> RawDepthSeed:
        """用于确定性检查；索引为仍在回放内的绝对采集时刻。"""
        rows, absolute = rows.to(self.device).long(), absolute.to(self.device).long()
        if rows.shape != absolute.shape or rows.ndim != 1 or ((rows < 0) | (rows >= self.num_envs)).any():
            raise ValueError("Invalid depth seed row indices")
        if ((absolute < self.clock - self.size) | (absolute >= self.clock)).any():
            raise ValueError("Depth seed times are outside retained replay")
        return self._raw_at(rows, absolute)[0]

    def sample_seeds(self, batch: int, *, device_indices=False) -> RawDepthSeed:
        if device_indices:
            raise ValueError("Raw image sampling must run outside CUDA Graph capture")
        if batch < 1 or self.size == 0:
            raise ValueError("Cannot sample an empty depth replay or nonpositive seed batch")
        return self._raw_at(*self._sample_positions(batch, 0, False))[0]

    def ready_auxiliary(self, refresh_steps=8):
        return refresh_steps > 0 and self.size >= refresh_steps * self.action_history_length

    def sample_auxiliary(self, batch: int, refresh_steps=8) -> RawDepthAuxiliary:
        if batch < 1 or self.size == 0 or refresh_steps < 1:
            raise ValueError("Invalid auxiliary batch, refresh count or empty replay")
        rows, absolute = self._sample_positions(batch, 0, False)
        seed, reconstruction = self._raw_at(rows, absolute, auxiliary=True)
        host_rows, host_time = rows.cpu(), absolute.cpu()
        future = host_time[:, None] + torch.arange(self.size)[None]
        slots = future % self.capacity
        done = self.host_done[slots, host_rows[:, None]]
        alive = torch.cat((torch.ones_like(done[:, :1]), (~done[:, :-1]).cumprod(1).bool()), 1)
        ids = self.refresh_ids[slots, host_rows[:, None]]
        fresh = (ids >= 0) & alive & (future < self.clock)
        positions = torch.arange(self.size).expand(batch, -1)
        ranked = torch.where(fresh, positions, self.size).topk(
            min(refresh_steps, self.size), dim=1, largest=False, sorted=True).values
        if self.size < refresh_steps:
            ranked = torch.nn.functional.pad(ranked, (0, refresh_steps - self.size), value=self.size)
        selected = ranked[:, :refresh_steps]
        valid = selected < self.size
        times = host_time[:, None] + selected.clamp_max(self.size - 1)
        target_slots = times % self.capacity
        chosen = self.refresh_ids[target_slots, host_rows[:, None]]
        # 不足位置只借用当前窗口末帧作定形占位；refresh_valid明确禁止其监督。
        chosen = torch.where(valid, chosen, self.frame_ids[host_time % self.capacity, host_rows, -1, None])
        frames = self.frames.gather(chosen, fields=("noisy",))
        device_time, device_rows = times.to(self.device), rows[:, None].expand(-1, refresh_steps)
        return RawDepthAuxiliary(seed, self.targets.index((device_time % self.capacity, device_rows)),
                                 self._actions_at(device_rows, device_time, after=True), frames.noisy,
                                 valid.to(self.device), reconstruction.clean, reconstruction.valid)
