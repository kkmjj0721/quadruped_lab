# Copyright (c) 2026-2027 zh
"""内容：将已处理深度图编码为潜变量，并重建同一时刻的干净深度。"""

import torch
from torch import Tensor, nn
import torch.nn.functional as functional


class DepthEncoder(nn.Module):
    """三层步幅卷积保留空间结构，输入遵循任务声明的单通道深度协议。"""

    def __init__(self, depth_shape: tuple[int, int, int], latent_dim: int):
        super().__init__()
        channels, height, width = depth_shape
        self.conv = nn.Sequential(nn.Conv2d(channels, 32, 5, 2, 2), nn.ReLU(),
                                  nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU(),
                                  nn.Conv2d(64, 64, 3, 2, 1), nn.ReLU())
        self.head = nn.Linear(64 * ((height + 7) // 8) * ((width + 7) // 8), latent_dim)

    def forward(self, depth: Tensor) -> Tensor:
        return self.head(self.conv(depth).flatten(1))


class DepthDecoder(nn.Module):
    """训练用解码器；末尾缩放支持不能被八整除的矩形相机尺寸。"""

    def __init__(self, depth_shape: tuple[int, int, int], latent_dim: int):
        super().__init__()
        channels, height, width = depth_shape
        self.height, self.width = height, width
        self.base_height, self.base_width = (height + 7) // 8, (width + 7) // 8
        self.fc = nn.Sequential(nn.Linear(latent_dim, 64 * self.base_height * self.base_width), nn.ReLU())
        self.conv = nn.Sequential(nn.ConvTranspose2d(64, 64, 4, 2, 1), nn.ReLU(),
                                  nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(),
                                  nn.ConvTranspose2d(32, channels, 4, 2, 1))

    def forward(self, latent: Tensor) -> Tensor:
        image = self.conv(self.fc(latent).reshape(-1, 64, self.base_height, self.base_width))
        return torch.sigmoid(functional.interpolate(image, size=(self.height, self.width), mode="bilinear", align_corners=False))
