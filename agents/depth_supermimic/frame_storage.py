# Copyright (c) 2026-2027 zh
"""内容：只保存实际新深度的CPU半精度帧池，以引用计数管理回放覆盖和局部重置。"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .interfaces import DepthFrame


@dataclass
class StoredFrames:
    """按请求索引顺序取出的CPU批次；未请求字段为None，掩码来自干净深度。"""

    noisy: Tensor | None = None
    clean: Tensor | None = None
    valid: Tensor | None = None
    world_to_camera: Tensor | None = None
    intrinsics: Tensor | None = None


_FRAME_FIELDS = ("noisy", "clean", "valid", "world_to_camera", "intrinsics")
_FRAME_DTYPES = dict(zip(_FRAME_FIELDS, (torch.float16, torch.float16, torch.bool, torch.float32, torch.float32),
                         strict=True))


class FrameStorage:
    """像素按块分配，ID在最后一个所有者释放后复用；不保存编码器输出。"""

    def __init__(self, capacity: int, image_shape=(1, 64, 64), chunk_size=65536):
        if capacity < 1 or chunk_size < 1 or len(image_shape) != 3 or min(image_shape) < 1:
            raise ValueError("Invalid frame-storage capacity or image shape")
        self.capacity, self.image_shape = capacity, tuple(image_shape)
        self.chunk_size = min(chunk_size, capacity)
        self.refcounts = torch.zeros(capacity, dtype=torch.int32)
        self._chunks: list[StoredFrames] = []
        self._staging: dict[str, Tensor] = {}
        self._free: list[int] = []
        self._next = 0

    @property
    def active_frames(self):
        return self._next - len(self._free)

    @property
    def bytes_allocated(self):
        return self.refcounts.numel() * self.refcounts.element_size() + sum(
            value.numel() * value.element_size() for block in self._chunks for value in vars(block).values())

    def clear(self):
        self.refcounts.zero_()
        self._chunks.clear()
        self._staging.clear()
        self._free.clear()
        self._next = 0

    def _allocate(self, count: int):
        if count > len(self._free) + self.capacity - self._next:
            raise RuntimeError("Bounded depth frame storage exhausted; check reference ownership")
        reuse = min(count, len(self._free))
        ids = [self._free.pop() for _ in range(reuse)]
        growth = count - reuse
        ids.extend(range(self._next, self._next + growth))
        self._next += growth
        while len(self._chunks) * self.chunk_size < self._next:
            rows = min(self.chunk_size, self.capacity - len(self._chunks) * self.chunk_size)
            self._chunks.append(StoredFrames(
                torch.empty(rows, *self.image_shape, dtype=torch.float16),
                torch.empty(rows, *self.image_shape, dtype=torch.float16),
                torch.empty(rows, *self.image_shape, dtype=torch.bool),
                torch.empty(rows, 4, 4, dtype=torch.float32), torch.empty(rows, 3, 3, dtype=torch.float32)))
        return torch.tensor(ids, dtype=torch.long)

    def _groups(self, flat: Tensor):
        """只排序一次，用连续区间定位块，避免逐块重新扫描所有请求ID。"""
        ordered, order = flat.sort()
        blocks, counts = torch.unique_consecutive(ordered // self.chunk_size, return_counts=True)
        start = 0
        for block, count in zip(blocks.tolist(), counts.tolist()):
            stop = start + count
            yield self._chunks[block], order[start:stop], ordered[start:stop] % self.chunk_size
            start = stop

    def _checked_ids(self, ids: Tensor):
        """CPU取样入口统一检查所有权；各块沿用已确认的ID范围。"""
        ids = ids.detach().cpu().long()
        flat = ids.flatten()
        if flat.numel() and ((flat < 0).any() or (flat >= self._next).any()
                           or (self.refcounts[flat] <= 0).any()):
            raise ValueError("Requested an unowned depth frame")
        return ids, flat

    @torch.no_grad()
    def append(self, frame: DepthFrame, rows: Tensor):
        """只拷贝指定的新帧；返回ID初始由调用方拥有一次引用。"""
        rows = rows.to(device=frame.noisy.device, dtype=torch.long)
        ids = self._allocate(rows.numel())
        if not ids.numel():
            return ids
        if rows.is_cuda:
            if not self._staging or self._staging["noisy"].shape[0] < rows.numel():
                capacity = min(self.capacity, max(rows.numel(), frame.noisy.shape[0]))
                self._staging = {name: torch.empty((capacity, *getattr(frame, name).shape[1:]), dtype=dtype,
                                                   device="cpu", pin_memory=True)
                                 for name, dtype in _FRAME_DTYPES.items()}
            host = {name: value[:rows.numel()] for name, value in self._staging.items()}
            for name, dtype in _FRAME_DTYPES.items():
                host[name].copy_(getattr(frame, name)[rows].to(dtype=dtype), non_blocking=True)
            torch.cuda.current_stream(rows.device).synchronize()
        else:
            host = {name: getattr(frame, name)[rows].to(device="cpu", dtype=dtype)
                    for name, dtype in _FRAME_DTYPES.items()}
        for block, selected, offsets in self._groups(ids):
            selected, offsets = selected.numpy(), offsets.numpy()
            for name, value in host.items():
                getattr(block, name).numpy()[offsets] = value.numpy()[selected]
        self.refcounts[ids] = 1
        return ids

    def retain(self, ids: Tensor):
        indices, counts = torch.unique(ids.detach().cpu().long().flatten(), return_counts=True)
        indices, counts = indices[indices >= 0], counts[indices >= 0]
        if indices.numel() and ((indices >= self._next).any() or (self.refcounts[indices] <= 0).any()):
            raise ValueError("Cannot retain an unowned depth frame")
        self.refcounts[indices] += counts.to(torch.int32)

    def release(self, ids: Tensor):
        indices, counts = torch.unique(ids.detach().cpu().long().flatten(), return_counts=True)
        indices, counts = indices[indices >= 0], counts[indices >= 0]
        remaining = self.refcounts[indices] - counts.to(torch.int32)
        if (remaining < 0).any():
            raise RuntimeError("Depth frame reference count became negative")
        self.refcounts[indices] = remaining
        self._free.extend(indices[remaining == 0].tolist())

    def gather(self, ids: Tensor, *, fields=_FRAME_FIELDS) -> StoredFrames:
        """图外只分配、读取被消费的字段；默认保持完整帧接口及任意索引形状。"""
        if len(fields) != len(set(fields)) or any(name not in _FRAME_FIELDS for name in fields):
            raise ValueError("Frame fields must contain unique supported field names")
        ids, flat = self._checked_ids(ids)
        result = StoredFrames()
        for name in fields:
            shape = ((4, 4) if name == "world_to_camera" else (3, 3) if name == "intrinsics"
                     else self.image_shape)
            dtype = (torch.bool if name == "valid" else torch.float16 if name in ("noisy", "clean")
                     else torch.float32)
            setattr(result, name, torch.empty(*ids.shape, *shape, dtype=dtype))
        if not fields:
            return result
        for block, selected, offsets in self._groups(flat):
            selected, offsets = selected.numpy(), offsets.numpy()
            for name in fields:
                value = getattr(result, name)
                target = value.reshape(flat.numel(), *value.shape[ids.ndim:])
                target.numpy()[selected] = getattr(block, name).numpy()[offsets]
        return result

    def sample_pixels(self, ids: Tensor, linear: Tensor):
        """只读取[B,K,F]指定的channel0像素，不展开四帧完整clean与valid。"""
        ids, flat = self._checked_ids(ids)
        linear = linear.detach().cpu().long()
        pixels_per_frame = self.image_shape[-2] * self.image_shape[-1]
        if ids.ndim != 2 or linear.ndim != 3 or linear.shape[:2] != ids.shape:
            raise ValueError("Sparse pixels require frame IDs [B,K] and pixel indices [B,K,F]")
        if ((linear < 0) | (linear >= pixels_per_frame)).any():
            raise ValueError("Sparse depth pixel indices are outside the image")
        clean = torch.empty(linear.shape, dtype=torch.float16)
        valid = torch.empty(linear.shape, dtype=torch.bool)
        count = linear.shape[-1]
        pixels = linear.reshape(flat.numel(), count)
        clean_flat, valid_flat = clean.reshape(flat.numel(), count), valid.reshape(flat.numel(), count)
        for block, selected, offsets in self._groups(flat):
            source_clean = block.clean[:, 0].reshape(-1, pixels_per_frame)
            source_valid = block.valid[:, 0].reshape(-1, pixels_per_frame)
            clean_flat[selected] = source_clean[offsets[:, None], pixels[selected]]
            valid_flat[selected] = source_valid[offsets[:, None], pixels[selected]]
        return clean, valid
