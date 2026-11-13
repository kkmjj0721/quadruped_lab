# Copyright (c) 2026-2027 zh
"""内容：按MGDP f43139d实现深度与高度U-Net，显式传递同批跳连以支持微批和部署。"""

import torch
from torch import Tensor, nn
import torch.nn.functional as functional


class DoubleConv(nn.Module):
    """原版两次无偏置卷积、分组归一化与SiLU。"""

    def __init__(self, input_channels: int, output_channels: int):
        super().__init__()
        groups = min(8, output_channels)
        while output_channels % groups:
            groups -= 1
        self.block = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups, output_channels), nn.SiLU(inplace=True),
            nn.Conv2d(output_channels, output_channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups, output_channels), nn.SiLU(inplace=True),
        )

    def forward(self, value: Tensor) -> Tensor:
        return self.block(value)


class UpFuseBlock(nn.Module):
    """原版解码器按对应编码特征的尺寸插值并融合。"""

    def __init__(self, input_channels: int, skip_channels: int, output_channels: int):
        super().__init__()
        self.fuse = DoubleConv(input_channels + skip_channels, output_channels)

    def forward(self, value: Tensor, skip: Tensor) -> Tensor:
        if value.shape[-2:] != skip.shape[-2:]:
            value = functional.interpolate(value, size=skip.shape[-2:], mode="bilinear", align_corners=True)
        return self.fuse(torch.cat((value, skip), dim=1))


class MGDPEncoder(nn.Module):
    """网络拓扑及参数命名与原版对应；跳连不保存为模块可变状态。"""

    def __init__(self, input_channels: int, hidden_channels: tuple[int, int, int], latent_dim: int, pool: int):
        super().__init__()
        first, second, third = hidden_channels
        self.conv1 = DoubleConv(input_channels, first)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = DoubleConv(first, second)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = DoubleConv(second, third)
        self.bottleneck = DoubleConv(third, third)
        self.global_pool = nn.AdaptiveAvgPool2d((pool, pool))
        self.fc = nn.Sequential(nn.Linear(third * pool * pool, third * 2), nn.LayerNorm(third * 2),
                                nn.SiLU(inplace=True), nn.Linear(third * 2, latent_dim))

    def encode(self, value: Tensor) -> tuple[Tensor, tuple[Tensor, Tensor, Tensor]]:
        first = self.conv1(value)
        second = self.conv2(self.pool1(first))
        third = self.bottleneck(self.conv3(self.pool2(second)))
        return self.fc(self.global_pool(third).flatten(1)), (first, second, third)

    def forward(self, value: Tensor) -> Tensor:
        latent, _ = self.encode(value)
        return latent


class MGDPDecoder(nn.Module):
    """原版三组跳连解码；两帧深度使用Sigmoid，高度图使用Tanh。"""

    def __init__(self, latent_dim: int, hidden_channels: tuple[int, int, int], pool: int,
                 output_shape: tuple[int, int, int], *, height_map: bool = False):
        super().__init__()
        first, second, third = hidden_channels
        self.pool, self.channels = pool, first
        self.height, self.width = output_shape[1:]
        self.fc = nn.Sequential(nn.Linear(latent_dim, first * pool * pool), nn.LayerNorm(first * pool * pool), nn.SiLU(inplace=True))
        self.input_refine = DoubleConv(first, first)
        self.up1 = UpFuseBlock(first, first, first)
        self.up2 = UpFuseBlock(first, second, second)
        self.up3 = UpFuseBlock(second, third, third)
        self.refine = DoubleConv(third, third)
        self.out_conv = nn.Conv2d(third, output_shape[0], 1)
        self.final_activation = nn.Tanh() if height_map else nn.Sigmoid()

    def forward(self, latent: Tensor, skips: tuple[Tensor, Tensor, Tensor]) -> Tensor:
        value = self.fc(latent).reshape(-1, self.channels, self.pool, self.pool)
        value = self.up1(self.input_refine(value), skips[2])
        value = self.up2(value, skips[1])
        value = self.up3(value, skips[0])
        if value.shape[-2:] != (self.height, self.width):
            value = functional.interpolate(value, size=(self.height, self.width), mode="bilinear", align_corners=True)
        return self.final_activation(self.out_conv(self.refine(value)))


class MGDPTerrainPerception(nn.Module):
    """原版高度32维自编码器，仅供训练使用。"""

    def __init__(self):
        super().__init__()
        self.encoder = MGDPEncoder(1, (16, 32, 64), 32, 2)
        self.decoder = MGDPDecoder(32, (64, 32, 16), 2, (1, 17, 11), height_map=True)
