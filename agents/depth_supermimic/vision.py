# Copyright (c) 2026-2027 zh
"""内容：提供仅接收深度的轻量 CNN 和仅从策略地形 latent 解码的重建网络。"""

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class DepthEncoder(nn.Module):
    """卷积使用GroupNorm，投影隐藏层使用LayerNorm，不读取本体或干净标签。"""

    def __init__(self, config):
        super().__init__()
        self.convolutions = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1), nn.GroupNorm(8, 16), nn.ELU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.GroupNorm(8, 32), nn.ELU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.GroupNorm(8, 64), nn.ELU())
        self.projection = nn.Sequential(nn.Linear(64 * 8 * 8, 128), nn.LayerNorm(128), nn.ELU(),
                                        nn.Linear(128, config.feature_dim))

    def forward(self, depth: Tensor) -> Tensor:
        return self.projection(self.convolutions(depth.float()).flatten(1))


class DepthDecoder(nn.Module):
    """解码器没有 CNN 跳接，输出线性深度，训练时不裁剪偏差。"""

    def __init__(self, config):
        super().__init__()
        self.projection = nn.Linear(config.latent_dim, 64 * 8 * 8)
        self.convolution1 = nn.Conv2d(64, 32, 3, padding=1)
        self.normalization1 = nn.GroupNorm(8, 32)
        self.convolution2 = nn.Conv2d(32, 16, 3, padding=1)
        self.normalization2 = nn.GroupNorm(8, 16)
        self.convolution3 = nn.Conv2d(16, 1, 3, padding=1)

    def forward(self, latent: Tensor) -> Tensor:
        values = self.projection(latent).reshape(-1, 64, 8, 8)
        values = F.interpolate(values, size=(16, 16), mode="bilinear", align_corners=False)
        values = F.elu(self.normalization1(self.convolution1(values)))
        values = F.interpolate(values, size=(32, 32), mode="bilinear", align_corners=False)
        values = F.elu(self.normalization2(self.convolution2(values)))
        return self.convolution3(F.interpolate(values, size=(64, 64), mode="bilinear", align_corners=False))


def reconstruction_loss(prediction: Tensor, target: Tensor, valid: Tensor) -> Tensor:
    """仅屏蔽目标无效像素；人工掉点仍由调用方保留在有效监督中。"""
    error = torch.where(valid, prediction - target.detach(), 0)
    return error.square().sum() / valid.sum().clamp_min(1)
