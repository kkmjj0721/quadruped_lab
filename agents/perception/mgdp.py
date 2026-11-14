# Copyright (c) 2026-2027 zh
"""
内容：
    按固定 MGDP 结构独立实现重建与对齐，并提供显式记忆的米制地图预测。
    来源、结构对应关系与本项目扩展见同目录 NOTICE.md。
"""

from copy import deepcopy

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .config import PerceptionConfig


# ---------- MGDP 编解码结构 ----------

class _ConvPair(nn.Sequential):
    """所选通道数均可被八整除，每次卷积后使用八组归一化及 SiLU。"""

    def __init__(self, input_width: int, output_width: int):
        layers = []
        for width in (input_width, output_width):
            layers.extend((nn.Conv2d(width, output_width, 3, padding=1, bias=False),
                           nn.GroupNorm(8, output_width), nn.SiLU()))
        super().__init__(*layers)


class _Encoder(nn.Module):
    """返回 token 和本次调用的跳接张量，不在模块中缓存上次输入。"""

    def __init__(self, input_channels: int, channels: tuple[int, int, int], token_dim: int, pool_size: int):
        super().__init__()
        a, b, c = channels
        self.stages = nn.ModuleList((_ConvPair(input_channels, a), _ConvPair(a, b),
                                     nn.Sequential(_ConvPair(b, c), _ConvPair(c, c))))
        self.reduce = nn.MaxPool2d(2)
        self.pool = nn.AdaptiveAvgPool2d((pool_size, pool_size))
        self.project = nn.Sequential(nn.Linear(c * pool_size * pool_size, 2 * c), nn.LayerNorm(2 * c),
                                     nn.SiLU(), nn.Linear(2 * c, token_dim))

    def forward(self, images: Tensor) -> tuple[Tensor, list[Tensor]]:
        features = torch.jit.annotate(list[Tensor], [])
        value = images
        for index, stage in enumerate(self.stages):
            value = stage(value if index == 0 else self.reduce(value))
            features.append(value)
        token = self.project(self.pool(value).flatten(1))
        return token, features


class _Decoder(nn.Module):
    """逐层融合本分支跳接；深度输出为 Sigmoid，高度自重建输出为 Tanh。"""

    def __init__(self, channels: tuple[int, int, int], token_dim: int, pool_size: int, output_channels: int):
        super().__init__()
        a, b, c = channels
        self.width, self.pool_size = c, pool_size
        projected = c * pool_size * pool_size
        self.project = nn.Sequential(nn.Linear(token_dim, projected), nn.LayerNorm(projected), nn.SiLU())
        self.seed = _ConvPair(c, c)
        self.fusions = nn.ModuleList((_ConvPair(2 * c, c), _ConvPair(c + b, b), _ConvPair(b + a, a)))
        self.output = nn.Sequential(_ConvPair(a, a), nn.Conv2d(a, output_channels, 1),
                                    nn.Sigmoid() if output_channels == 2 else nn.Tanh())

    def forward(self, token: Tensor, features: list[Tensor]) -> Tensor:
        value = self.seed(self.project(token).reshape(-1, self.width, self.pool_size, self.pool_size))
        for index, fusion in enumerate(self.fusions):
            skip = features[2 - index]
            value = F.interpolate(value, size=skip.shape[-2:], mode="bilinear", align_corners=True)
            value = fusion(torch.cat((value, skip), dim=1))
        return self.output(value)


# ---------- 部署感知与时序地图 ----------

class DepthPerception(nn.Module):
    """只消费深度与本体；调用方负责帧刷新、token 缓存、回合重置与前缀重放。"""

    def __init__(self, config: PerceptionConfig):
        super().__init__()
        self.proprio_dim = config.proprio_dim
        self.history_length = config.history_length
        self.token_dim = config.token_dim
        self.hidden_dim = config.hidden_dim
        self.depth_shape = config.depth_shape
        self.map_shape = config.map_shape
        self.use_bf16 = False
        self.depth_encoder = _Encoder(2, (32, 64, 96), config.token_dim, 4)
        self.gru = nn.GRUCell(config.token_dim + config.proprio_dim * config.history_length, config.hidden_dim)
        self.map_head = nn.Sequential(nn.Linear(config.hidden_dim, 256), nn.SiLU(), nn.Linear(256, 128),
                                      nn.SiLU(), nn.Linear(128, config.map_shape[0] * config.map_shape[1]))

    @torch.jit.export
    def encode_depth(self, depth_pair: Tensor) -> Tensor:
        """输入已按固定量程归一化的双帧 [B,2,16,16]。"""
        if (depth_pair.dim() != 4 or depth_pair.shape[1] != 2
                or depth_pair.shape[2] != self.depth_shape[0] or depth_pair.shape[3] != self.depth_shape[1]):
            raise ValueError("Depth input must have shape [B,2,16,16]")
        if self.use_bf16:
            with torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
                token = self.depth_encoder(depth_pair)[0]
            return token.float()
        return self.depth_encoder(depth_pair)[0]

    @torch.jit.export
    def step(self, depth_token: Tensor, proprio_history: Tensor, hidden: Tensor) -> tuple[Tensor, Tensor]:
        """每控制步推进一次；相机未刷新时传入此前缓存的同一 token。"""
        if depth_token.dim() != 2 or depth_token.shape[1] != self.token_dim:
            raise ValueError("Depth token must have shape [B,64]")
        batch = depth_token.shape[0]
        if (proprio_history.dim() != 3 or proprio_history.shape[0] != batch
                or proprio_history.shape[1] != self.history_length or proprio_history.shape[2] != self.proprio_dim):
            raise ValueError("Proprioceptive history shape does not match the perception configuration")
        if hidden.dim() != 2 or hidden.shape[0] != batch or hidden.shape[1] != self.hidden_dim:
            raise ValueError("Perception hidden state must have shape [B,128]")
        if self.use_bf16:
            with torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
                height_map, next_hidden = self._advance_memory(depth_token, proprio_history, hidden)
            return height_map.float(), next_hidden.float()
        return self._advance_memory(depth_token, proprio_history, hidden)

    def _advance_memory(self, depth_token: Tensor, proprio_history: Tensor, hidden: Tensor) -> tuple[Tensor, Tensor]:
        """只执行时序网络；对外记忆的 FP32 边界由 step 维护。"""
        next_hidden = self.gru(torch.cat((depth_token, proprio_history.flatten(1)), dim=1), hidden)
        height_map = self.map_head(next_hidden).reshape(depth_token.shape[0], self.map_shape[0], self.map_shape[1])
        return height_map, next_hidden

    def forward(self, depth_pair: Tensor, proprio_history: Tensor, hidden: Tensor) -> tuple[Tensor, Tensor]:
        """相机刷新步的组合入口；帧保持步直接使用 encode_depth／step 分离接口。"""
        return self.step(self.encode_depth(depth_pair), proprio_history, hidden)


# ---------- 训练分支与监督损失 ----------

def map_loss(predicted: Tensor, target: Tensor, valid: Tensor) -> Tensor:
    """在有效米制标签上求均方误差；无效区域允许 NaN，空掩码返回可反传的零。"""
    if predicted.shape != target.shape or valid.shape != target.shape or valid.dtype != torch.bool:
        raise ValueError("Map predictions, targets and boolean validity must have matching shapes")
    difference = torch.where(valid, predicted, 0.0) - torch.where(valid, target.detach(), 0.0)
    return difference.square().sum() / valid.sum().clamp_min(1)


class MGDPPerception(nn.Module):
    """部署地图预测与训练自重建共享深度编码器，GT 分支没有通向部署输出的路径。"""

    def __init__(self, config: PerceptionConfig):
        super().__init__()
        self.config = config
        self.deployment = DepthPerception(config)
        self.depth_decoder = _Decoder((32, 64, 96), config.token_dim, 4, 2)
        self.height_encoder = _Encoder(1, (16, 32, 64), 32, 2)
        self.height_decoder = _Decoder((16, 32, 64), 32, 2, 1)

    def configure_bf16(self, enabled: bool):
        """部署和重建共享同一开关；设备支持由学习器入口统一校验。"""
        self.deployment.use_bf16 = enabled

    def encode_depth(self, depth_pair: Tensor) -> Tensor:
        return self.deployment.encode_depth(depth_pair)

    def step(self, depth_token: Tensor, proprio_history: Tensor, hidden: Tensor) -> tuple[Tensor, Tensor]:
        return self.deployment.step(depth_token, proprio_history, hidden)

    def forward(self, depth_pair: Tensor, proprio_history: Tensor, hidden: Tensor) -> tuple[Tensor, Tensor]:
        return self.deployment(depth_pair, proprio_history, hidden)

    def reconstruction_loss(self, depth_pair: Tensor, clean_pair: Tensor, height_normalized: Tensor) -> dict[str, Tensor]:
        """官方单向高度→深度 InfoNCE；GT 高度输入仅用于自重建及对齐。"""
        if (depth_pair.dim() != 4 or depth_pair.shape[1:] != (2, *self.config.depth_shape)
                or clean_pair.shape != depth_pair.shape or depth_pair.shape[0] < 1):
            raise ValueError("Reconstruction depth inputs must have matching nonempty [B,2,16,16] shapes")
        if height_normalized.shape != (depth_pair.shape[0], 1, *self.config.map_shape):
            raise ValueError("Normalized height targets must have shape [B,1,17,11]")
        clean = clean_pair.detach().float().clamp(0.0, 1.0)
        height = height_normalized.detach().float()
        if self.deployment.use_bf16:
            with torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
                depth_token, height_token, depth_reconstruction, height_reconstruction = self._reconstruct(depth_pair, height)
        else:
            depth_token, height_token, depth_reconstruction, height_reconstruction = self._reconstruct(depth_pair, height)
        # 编解码及跳接参与 BF16；监督归约和跨模态相似度保持 FP32。
        depth_token, height_token = depth_token.float(), height_token.float()
        depth_error = F.mse_loss(depth_reconstruction.float().clamp(0.0, 1.0), clean)
        height_error = F.mse_loss(height_reconstruction.float(), height)
        contrastive = depth_error.new_zeros(())
        if depth_pair.shape[0] >= 2:
            depth_unit = F.normalize(depth_token[:, :32], dim=1, eps=1e-8)
            height_unit = F.normalize(height_token, dim=1, eps=1e-8)
            labels = torch.arange(depth_pair.shape[0], device=depth_pair.device)
            contrastive = F.cross_entropy(height_unit @ depth_unit.T / 0.1, labels)
        total = 1.2 * depth_error + height_error + 0.3 * contrastive
        return {"loss": total, "depth": depth_error, "height": height_error, "contrastive": contrastive}

    def _reconstruct(self, depth_pair: Tensor, height: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """统一执行双分支编解码，便于在同一精度范围内保留全部跳接梯度。"""
        depth_token, depth_features = self.deployment.depth_encoder(depth_pair)
        height_token, height_features = self.height_encoder(height)
        return depth_token, height_token, self.depth_decoder(depth_token, depth_features), self.height_decoder(height_token, height_features)

    def inference_module(self) -> DepthPerception:
        """导出独立副本；不携带训练解码器、GT 编码器或任何运行时缓存。"""
        return deepcopy(self.deployment)
