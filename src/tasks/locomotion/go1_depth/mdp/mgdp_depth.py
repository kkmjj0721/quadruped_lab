# Copyright (c) 2026-2027 zh
"""内容：按 MGDP Warp 相机实际顺序处理米制深度和采集时的高度标签。"""

import torch
import torch.nn.functional as functional


def process_mgdp_depth(raw: torch.Tensor, *, noisy: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
    """输入 [B,120,120,1] 正向光轴深度，返回带噪/干净的 [B,1,16,16]。

    对照 Warp 图像后处理及 random_dog.py:232–263；不在 resize 前归一化。
    mujoco_warp/render.py 的无命中为零，官方 cam_kernel.py 则为1000m；
    在统一裁剪前将零映射到官方无命中值，NaN 不静默修复。
    """
    if raw.ndim != 4 or raw.shape[1:] != (120, 120, 1):
        raise ValueError("MGDP raw depth must have shape [batch,120,120,1].")
    meters = torch.where(raw == 0, 1000.0, raw).permute(0, 3, 1, 2).clamp(0.1, 2.5)
    clean = functional.interpolate(meters, size=(16, 16), mode="bilinear", align_corners=False)
    image = clean.clone()
    if noisy:
        image = image + torch.randn_like(image) * 0.03
        image = image * (torch.rand_like(image) > 0.1)
    return image.clamp(0, 5) / 5, clean.clamp(0, 5) / 5


def capture_height_label(clean_height: torch.Tensor, *, noisy: bool = True) -> torch.Tensor:
    """官方高度观测噪声除以尺度5之后为±.1m，最后再夹到[-1,1]。"""
    height = clean_height
    if noisy:
        height = height + (2 * torch.rand_like(height) - 1) * 0.1
    return height.clamp(-1, 1)
