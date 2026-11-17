# Copyright (c) 2026-2027 zh
"""内容：单帧空间token、干净深度重建、动作条件视觉prior和四足未来落足预测。"""

import torch
from torch import nn
from torch.nn import functional as F

from .policy import Mlp


class DepthEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(1, 16, 3, 2, 1), nn.GroupNorm(8, 16), nn.ELU(),
            nn.Conv2d(16, 32, 3, 2, 1), nn.GroupNorm(8, 32), nn.ELU(),
            nn.Conv2d(32, 32, 3, 2, 1), nn.GroupNorm(8, 32), nn.ELU(),
            nn.Conv2d(32, 16, 1), nn.AvgPool2d(2))

    def forward(self, images):
        return self.layers(images.float()).flatten(2).transpose(1, 2)


class DepthDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.ConvTranspose2d(16, 32, 4, 2, 1), nn.ELU(),
            nn.ConvTranspose2d(32, 16, 4, 2, 1), nn.ELU(),
            nn.ConvTranspose2d(16, 8, 4, 2, 1), nn.ELU(),
            nn.ConvTranspose2d(8, 1, 4, 2, 1), nn.Sigmoid())

    def forward(self, tokens):
        return self.layers(tokens.transpose(1, 2).reshape(-1, 16, 4, 4))


class Perception(nn.Module):
    """辅助Adam独占本模块；策略更新只保留其输入雅可比。"""

    def __init__(self):
        super().__init__()
        self.encoder = DepthEncoder()
        self.decoder = DepthDecoder()
        self.prior = Mlp([1206, 256, 256, 256])
        self.fpm = Mlp([235, 128, 128, 8])

    def predict_frame(self, tokens, physical, action_history):
        values = torch.cat((tokens.flatten(1), physical, action_history.flatten(1)), -1)
        return self.prior(values).reshape(-1, 16, 16)

    def advance(self, tokens, physical, action_history, age):
        next_age = age + 1
        predicted = self.predict_frame(tokens, physical, action_history)
        shifted = torch.cat((tokens[:, 16:], predicted), 1)
        fresh = next_age >= 5
        # 静态全batch保持图地址与形状；写入10Hz不等于计算仅10Hz。
        return torch.where(fresh[:, :, None], shifted, tokens), torch.where(fresh, 0, next_age)

    def points(self, normalized_history):
        return self.fpm(normalized_history).reshape(-1, 4, 2)

    def encode_window(self, images, *, device, microbatch=256):
        shape = images.shape
        flat = images.reshape(-1, 1, shape[-2], shape[-1])
        encoded = []
        for part in flat.split(microbatch):
            values = part.to(device=device, dtype=torch.float32, non_blocking=True)
            count = values.shape[0]
            # CUDA推理固定卷积批量，避免异步重置产生的新shape反复触发cuDNN搜索。
            # GroupNorm逐样本计算；补齐行在输出处裁掉，不参与策略或辅助损失。
            if values.is_cuda and count < microbatch:
                values = F.pad(values, (0, 0, 0, 0, 0, 0, 0, microbatch - count))
            encoded.append(self.encoder(values)[:count])
        return torch.cat(encoded).reshape(shape[0], shape[1] * 16, 16)
