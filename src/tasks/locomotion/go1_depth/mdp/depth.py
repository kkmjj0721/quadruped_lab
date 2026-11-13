# Copyright (c) 2026-2027 zh
"""内容：把原生光轴米制深度转换为同帧带噪输入与干净监督。"""

import torch
import torch.nn.functional as functional


def process_depth(raw, cfg):
    """无命中和非有限值表示远端；先缩放米制图再加噪，缺失像素输入编码为零。"""
    meters = raw.permute(0, 3, 1, 2)
    meters = torch.where(torch.isfinite(meters) & (meters > 0), meters, cfg.depth_far)
    meters = functional.interpolate(meters.clamp(cfg.depth_near, cfg.depth_far),
                                    size=(cfg.depth_size, cfg.depth_size), mode="bilinear", align_corners=False)
    scale = cfg.depth_far - cfg.depth_near
    clean = (meters - cfg.depth_near) / scale
    if cfg.depth_noise:
        noisy = ((meters + torch.randn_like(meters) * cfg.depth_noise_std).clamp(cfg.depth_near, cfg.depth_far) - cfg.depth_near) / scale
        noisy = noisy * (torch.rand_like(noisy) >= cfg.depth_dropout)
    else:
        noisy = clean.clone()
    return noisy.float(), clean.float()
