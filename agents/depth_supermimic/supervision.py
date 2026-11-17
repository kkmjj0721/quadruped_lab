# Copyright (c) 2026-2027 zh
"""内容：构造严格未来触地标签，并以同次干净深度检查四帧视觉读取监督。"""

from __future__ import annotations

import torch
from torch import Tensor

from .frame_storage import FrameStorage, StoredFrames


def strictly_future_contact_indices(onsets: Tensor, done: Tensor, available: Tensor):
    """输入未来原生转移；终止转移可产生事件，重置后的事件与回放尾部无效。

    onsets必须已经由当前决策态接触与原生final接触的上升沿构造，因此
    当前正在接触的足不会把当前接触位置作为下一次触地标签。
    """
    alive = torch.cat((torch.ones_like(done[:, :1]), (~done[:, :-1]).cumprod(1).bool()), 1)
    candidates = onsets.bool() & (alive & available).unsqueeze(-1)
    valid = candidates.any(1)
    # 先求每脚首次事件，调用方仅从原始存储读取最终B×4个位置。
    return candidates.to(torch.int8).argmax(1), valid


def project_read_targets(world_points: Tensor, future_valid: Tensor, frames: FrameStorage,
                         frame_ids: Tensor, projection: StoredFrames, *,
                         depth_scale: float, visibility_tolerance=0.08, token_grid=4):
    """未来足端site投影到历史相机，经视野、clean有效像素和遮挡检查后归一化。

    每帧4×4 token采用半token标准差的高斯软标签。人工深度掉点不参与
    可见性判断；干净像素缺失、投影在相机后方或被近处表面遮挡均不监督。
    """
    points = world_points.detach().cpu().float()
    future_valid = future_valid.detach().cpu().bool()
    _, height, width = frames.image_shape
    batch, window = frame_ids.shape
    camera = torch.einsum("bkij,bfj->bkfi", projection.world_to_camera[..., :3, :3], points)
    camera = camera + projection.world_to_camera[..., None, :3, 3]
    projected = torch.einsum("bkij,bkfj->bkfi", projection.intrinsics, camera)
    z = camera[..., 2]
    uv = projected[..., :2] / z.clamp_min(1e-6)[..., None]
    inside = ((uv[..., 0] >= 0) & (uv[..., 0] <= width - 1)
              & (uv[..., 1] >= 0) & (uv[..., 1] <= height - 1) & (z > 0))
    pixel = uv.round().long()
    linear = pixel[..., 1].clamp(0, height - 1) * width + pixel[..., 0].clamp(0, width - 1)
    clean, mask = frames.sample_pixels(frame_ids, linear)
    clean = clean.float()
    visible = (inside & mask & torch.isfinite(clean)
               & ((clean * depth_scale - z).abs() <= visibility_tolerance + z.abs() * .01)
               & future_valid[:, None])
    coordinates = (uv + .5) * uv.new_tensor((token_grid / width, token_grid / height)) - .5
    axis = torch.arange(token_grid, dtype=torch.float32)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    centers = torch.stack((xx.flatten(), yy.flatten()), -1)
    squared = (coordinates[..., None, :] - centers).square().sum(-1)
    weights = torch.exp(-squared / .5) * visible[..., None]
    weights = weights.permute(0, 2, 1, 3).reshape(batch, points.shape[1], -1)
    total = weights.sum(-1, keepdim=True)
    targets = weights / total.clamp_min(1e-12)
    return targets, total.squeeze(-1) > 0
