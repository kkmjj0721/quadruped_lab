# Copyright (c) 2026-2027 zh
"""内容：用高度图重建与整批跨模态InfoNCE监督地形特征，不参与部署或动力学递推。"""

import torch
from torch import Tensor, nn
import torch.nn.functional as functional


class TerrainPerception(nn.Module):
    """无跳连、无批次统计的高度自编码器，支持确定性微批重算。"""

    def __init__(self, shape: tuple[int, int], latent_dim: int):
        super().__init__()
        self.shape = shape
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((2, 2)), nn.Flatten(), nn.Linear(256, latent_dim),
        )
        self.projection = nn.Sequential(nn.Linear(latent_dim, 256), nn.ReLU())
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 4, 2, 1),
        )

    def reconstruct(self, latent: Tensor) -> Tensor:
        image = self.decoder(self.projection(latent).reshape(-1, 64, 2, 2))
        return torch.tanh(functional.interpolate(image, size=self.shape, mode="bilinear", align_corners=False))


def contrastive_loss(visual: Tensor, height: Tensor, temperature: float) -> Tensor:
    """同索引为正对，其余整批样本为负对；方向与MGDP的高度到深度交叉熵一致。"""
    visual = functional.normalize(visual, p=2, dim=-1, eps=1e-8)
    height = functional.normalize(height, p=2, dim=-1, eps=1e-8)
    logits = height @ visual.T / temperature
    return functional.cross_entropy(logits, torch.arange(logits.shape[0], device=logits.device))
