# Copyright (c) 2026-2027 zh
"""内容：按相机帧保存完整回合前缀，复用本体单帧环并保留归档关联图像。"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import torch

from .replay_storage import Replay, Window

from .frame_storage import HostDepthFrames
from .interfaces import DepthObservation, DepthSeed, DepthWindow


@dataclass
class _CameraFrame:
    clean: torch.Tensor
    noisy: torch.Tensor
    valid: torch.Tensor


@dataclass
class _Episode:
    identifier: int
    frames: list[_CameraFrame]
    last_index: int = -1
    references: int = 0


@dataclass
class _FrameRef:
    episode: _Episode
    index: int


@dataclass
class _WindowRecord:
    window: Window
    history: torch.Tensor
    phase: torch.Tensor
    prefix: _FrameRef
    next_frames: list[_FrameRef | None]


@dataclass
class _WindowBatch:
    """候选和训练窗口保留整批张量；Python对象只关联原始相机帧。"""

    window: Window
    history: torch.Tensor
    phase: torch.Tensor
    prefixes: list[_FrameRef]
    next_frames: list[list[_FrameRef | None]]

    def index(self, indices, rows):
        return _WindowBatch(self.window.index(indices), self.history[indices], self.phase[indices],
                            [self.prefixes[row] for row in rows], [self.next_frames[row] for row in rows])


class DepthReplay:
    """覆盖控制环时不截断活跃前缀；归档持有相同原始相机帧而非旧 latent。

    每环境必须在任务最大回合长度内重置。图像预算同时限制主环与归档的全部
    回合前缀；预算不足先淘汰旧回合归档，主环仍不可容纳时明确报错。
    """

    frame_block_rows = 64

    def __init__(self, spec, num_envs, capacity, device, *, image_capacity, archive_capacity=0):
        if image_capacity < num_envs or archive_capacity < 0:
            raise ValueError("Image capacity must cover all environments and archive capacity must be nonnegative")
        self.base = Replay(spec, num_envs, capacity, device, history_storage="frames")
        self.spec, self.num_envs, self.capacity = spec, num_envs, capacity
        self.device = torch.device(device)
        self.image_capacity = image_capacity
        self.archive_capacity = archive_capacity
        self.clear()

    @property
    def size(self):
        return self.base.size

    def clear(self):
        self.base.clear()
        self.episodes = [None] * self.num_envs
        self.prefixes = [None] * self.capacity
        self.next_frames = [None] * self.capacity
        self.phases = torch.zeros(self.capacity, self.num_envs, dtype=torch.long, device=self.device)
        self.archive = deque(maxlen=self.archive_capacity)
        self._image_count = 0

    def ready(self, horizon):
        return self.base.ready(horizon)

    def _retain(self, episode):
        if episode.references == 0:
            self._image_count += len(episode.frames)
        episode.references += 1

    def _release(self, episode):
        episode.references -= 1
        if episode.references == 0:
            self._image_count -= len(episode.frames)

    def _retain_rows(self, references):
        for reference in references:
            if reference is not None:
                self._retain(reference.episode)

    def _release_rows(self, references):
        if references is not None:
            for reference in references:
                if reference is not None:
                    self._release(reference.episode)

    def _record_frames(self, depth: DepthObservation):
        metadata = torch.stack((depth.episode_id, depth.frame_index, depth.new_frame.long()), -1).cpu().tolist()
        references = []
        pending = []
        for row, (identifier, frame_index, fresh) in enumerate(metadata):
            episode = self.episodes[row]
            if episode is None or episode.identifier != identifier:
                if not fresh:
                    raise ValueError("A new replay episode must start with a fresh depth frame")
                if episode is not None:
                    self._release(episode)
                episode = _Episode(identifier, [])
                self.episodes[row] = episode
                self._retain(episode)
            if frame_index != episode.last_index:
                if not fresh:
                    raise ValueError("Depth frame index changed without a camera refresh")
                pending.append((row, episode, frame_index))
                references.append(_FrameRef(episode, len(episode.frames)))
            else:
                references.append(_FrameRef(episode, len(episode.frames) - 1))
        # 按块搬运实际新帧到CPU，再逐行独立持有，长期保存不会钉住同块其他回合。
        for start in range(0, len(pending), self.frame_block_rows):
            block = pending[start:start + self.frame_block_rows]
            rows = torch.tensor([row for row, _, _ in block], device=self.device)
            clean = depth.clean[rows].to(torch.float16).cpu()
            noisy = depth.noisy[rows].to(torch.float16).cpu()
            valid = depth.valid[rows].cpu()
            for index, (_, episode, frame_index) in enumerate(block):
                episode.frames.append(_CameraFrame(clean[index].clone(), noisy[index].clone(), valid[index].clone()))
                episode.last_index = frame_index
                self._image_count += 1
        return references

    @torch.no_grad()
    def add(self, state, history, action, result, before_depth, after_depth):
        before = self._record_frames(before_depth)
        # after中的reset会释放旧active引用，先持有本次动作起点所属回合。
        self._retain_rows(before)
        after = self._record_frames(after_depth)
        slot = self.base.clock % self.capacity
        done = result.terminated | result.truncated
        fresh = (after_depth.new_frame & ~done).cpu().tolist()
        next_frames = [reference if active else None for reference, active in zip(after, fresh)]
        self._retain_rows(next_frames)
        self._release_rows(self.prefixes[slot])
        self._release_rows(self.next_frames[slot])
        self.prefixes[slot], self.next_frames[slot] = before, next_frames
        self.phases[slot].copy_(before_depth.phase)
        self.base.add(state, history, action, result)
        self._enforce_image_budget()

    @property
    def image_count(self):
        return self._image_count

    def _enforce_image_budget(self):
        while self.archive and self.image_count > self.image_capacity:
            oldest = self.archive[0].prefix.episode
            retained = deque(maxlen=self.archive_capacity)
            for record in self.archive:
                if record.prefix.episode is oldest:
                    self._release(oldest)
                else:
                    retained.append(record)
            self.archive = retained
        if self.image_count > self.image_capacity:
            raise MemoryError("Depth replay image budget cannot retain active episodes and the control ring; "
                              "increase image_capacity or reduce environment/control replay capacity")

    def _history(self, rows, absolute):
        slots = absolute % self.capacity
        depth = self.base.history_depth[slots, rows]
        back = torch.minimum(self.base.history_offsets[None, :], depth[:, None] - 1)
        frame_slots = (absolute[:, None] - back) % self.base.frame_capacity
        return self.base.observation_frames[frame_slots, rows[:, None]].float()

    def _pack_seed(self, state, history, phase, references):
        prefixes = [reference.episode.frames[:reference.index + 1] for reference in references]
        images = HostDepthFrames(tuple(tuple(frame.noisy for frame in prefix) for prefix in prefixes),
                                 prefixes[0][0].noisy.shape)
        lengths = torch.tensor(images.lengths, device=self.device)
        valid = torch.arange(images.shape[1], device=self.device)[None, :] < lengths[:, None]
        return DepthSeed(state, history, phase, images, valid,
                         torch.stack([prefix[-1].clean for prefix in prefixes]).to(self.device, dtype=torch.float32),
                         torch.stack([prefix[-1].valid for prefix in prefixes]).to(self.device))

    def sample_seeds(self, batch):
        if batch < 1 or self.size == 0:
            raise ValueError("Cannot sample an empty depth replay or nonpositive batch")
        rows, absolute = self.base._sample_positions(batch, 0, False)
        slots = absolute % self.capacity
        indices = torch.stack((slots, rows), -1).cpu().tolist()
        references = [self.prefixes[slot][row] for slot, row in indices]
        return self._pack_seed(self.base.states.index((slots, rows)), self._history(rows, absolute),
                               self.phases[slots, rows], references)

    def current_seed(self, state, history, depth):
        """采集前用当前权重重建活跃回合，避免跨优化轮沿用旧 encoder 的 hidden。"""
        references = self._record_frames(depth)
        self._enforce_image_budget()
        return self._pack_seed(state, history, depth.phase, references)

    def _sample_windows(self, batch, horizon):
        if batch < 1 or not self.ready(horizon):
            raise ValueError("Depth replay has insufficient transitions for model windows")
        rows, absolute = self.base._sample_positions(batch, horizon, False)
        starts = absolute % self.capacity
        slots = (starts[:, None] + torch.arange(horizon, device=self.device)) % self.capacity
        done = self.base.done[slots, rows[:, None]]
        window = Window(self.base.states.index((starts, rows)), self.base.actions[slots, rows[:, None]],
                        self.base.states.index(((slots + 1) % self.capacity, rows[:, None])),
                        (~done).cumprod(1).bool(), self.base.terminated[slots, rows[:, None]])
        histories = self._history(rows, absolute)
        indices = torch.stack((starts, rows), -1).cpu().tolist()
        valid_rows = window.valid.cpu().tolist()
        prefixes = [self.prefixes[start][row] for start, row in indices]
        next_frames = [[self.next_frames[(start + offset) % self.capacity][row]
                        if valid_rows[index][offset] else None for offset in range(horizon)]
                       for index, (start, row) in enumerate(indices)]
        return _WindowBatch(window, histories, self.phases[starts, rows], prefixes, next_frames)

    def _records_batch(self, records):
        return _WindowBatch(Window.concatenate([record.window for record in records]),
                            torch.cat([record.history for record in records]),
                            torch.cat([record.phase for record in records]),
                            [record.prefix for record in records], [record.next_frames for record in records])

    def _archive_windows(self, candidates, keep):
        """仅复制实际保留的行，单行最多持有64行归档块，覆盖时显式释放回合。"""
        selected = [row for row, active in enumerate(keep) if active]
        for start in range(0, len(selected), self.frame_block_rows):
            rows = selected[start:start + self.frame_block_rows]
            indices = torch.tensor(rows, device=self.device)
            copied = candidates.index(indices, rows)
            for index, row in enumerate(rows):
                record = _WindowRecord(copied.window.index(slice(index, index + 1)),
                                       copied.history[index:index + 1], copied.phase[index:index + 1],
                                       candidates.prefixes[row], candidates.next_frames[row])
                self._retain(record.prefix.episode)
                if len(self.archive) == self.archive_capacity:
                    self._release(self.archive.popleft().prefix.episode)
                self.archive.append(record)

    def _pack_windows(self, records):
        batch = records if isinstance(records, _WindowBatch) else self._records_batch(records)
        seed = self._pack_seed(batch.window.initial, batch.history, batch.phase, batch.prefixes)
        frames = tuple(tuple(None if reference is None else reference.episode.frames[reference.index].noisy
                             for reference in row) for row in batch.next_frames)
        images = HostDepthFrames(frames, seed.clean.shape[1:])
        masks = torch.tensor([[frame is not None for frame in row] for row in frames], device=self.device)
        return DepthWindow(seed, batch.window.actions, batch.window.targets, batch.window.valid,
                           batch.window.terminated, images, masks)

    def sample_windows(self, batch, horizon):
        return self._pack_windows(self._sample_windows(batch, horizon))

    def model_batch(self, task, config):
        prioritized = int(config.wm_batch * config.wm_prioritized_frac)
        archived = int(config.wm_batch * config.wm_archive_frac) if self.archive else 0
        uniform = config.wm_batch - prioritized - archived
        parts = [self._sample_windows(uniform, config.wm_window)] if uniform else []
        if archived:
            picks = torch.randint(len(self.archive), (archived,)).tolist()
            parts.append(self._records_batch([self.archive[index] for index in picks]))
        if prioritized or self.archive_capacity:
            candidates = self._sample_windows(4 * max(prioritized, 1), config.wm_window)
            windows = candidates.window
            scores = task.replay_priority(windows)
            if prioritized:
                selected = (scores + 1e-3 * torch.rand_like(scores)).topk(prioritized).indices
                chosen = selected.cpu().tolist()
                parts.append(candidates.index(selected, chosen))
            if self.archive_capacity:
                keep = task.archive_worthy(windows) & (torch.rand_like(scores) < config.archive_keep_prob)
                self._archive_windows(candidates, keep.cpu().tolist())
                self._enforce_image_budget()
        batch = _WindowBatch(Window.concatenate([part.window for part in parts]),
                             torch.cat([part.history for part in parts]), torch.cat([part.phase for part in parts]),
                             [reference for part in parts for reference in part.prefixes],
                             [row for part in parts for row in part.next_frames])
        return self._pack_windows(batch)
