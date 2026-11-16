# Copyright (c) 2026-2027 zh
"""内容：实现深度专用循环记忆、五帧本体单 MLP 策略及训练专用循环价值网络。"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .mlp import Mlp
from .normalization import RunningNorm
from .interfaces import PreparedDepthSeed
from .frame_storage import HostDepthFrames

from .vision import DepthEncoder


class DepthActor(nn.Module):
    """视觉分支只读深度，外部显式持有记忆；策略仅在动作 MLP 前融合本体。"""

    def __init__(self, config):
        super().__init__()
        self.observation_dim = config.observation_dim
        self.history_length = config.history_length
        self.hidden_dim = config.hidden_dim
        self.feature_dim = config.feature_dim
        self.latent_dim = config.latent_dim
        self.action_dim = config.action_dim
        self.history_order = "frame_major_oldest_first"
        self.input_order = "history_terrain_latent"
        self.encoder = DepthEncoder(config)
        self.feature_norm = nn.LayerNorm(config.feature_dim)
        self.gru = nn.GRUCell(config.feature_dim, config.hidden_dim)
        self.projection = nn.Linear(config.hidden_dim, config.latent_dim)
        self.normalizer = RunningNorm(config.history_length * config.observation_dim)
        self.actor = Mlp([config.history_length * config.observation_dim + config.latent_dim,
                          *config.actor_hidden, config.action_dim])

    def forward(self, history: Tensor, latent: Tensor) -> Tensor:
        history = history.reshape(-1, self.history_length * self.observation_dim)
        return self.actor(torch.cat((self.normalizer(history), latent), -1))

    @torch.jit.export
    def latent(self, hidden: Tensor) -> Tensor:
        return self.projection(hidden)

    @torch.jit.export
    def update_memory(self, features: Tensor, hidden: Tensor, new_frame: Tensor) -> tuple[Tensor, Tensor]:
        """预测特征使用固定形状 mask，非刷新步严格保持 hidden 的值与导数。"""
        candidate = self.gru(self.feature_norm(features), hidden)
        updated = torch.where(new_frame[:, None], candidate, hidden)
        return updated, self.latent(updated)

    @torch.jit.export
    def encode(self, depth: Tensor, hidden: Tensor, new_frame: Tensor) -> tuple[Tensor, Tensor]:
        """仅编码新帧子集，不把缓存图像在五个控制步中重复输入 GRU。"""
        indices = new_frame.nonzero().flatten()
        features = self.encoder(depth.index_select(0, indices))
        updated = self.gru(self.feature_norm(features), hidden.index_select(0, indices))
        next_hidden = hidden.index_copy(0, indices, updated)
        return next_hidden, self.latent(next_hidden)

    @torch.jit.ignore
    def _prepare_visual(self, seed):
        """可变长度旧前缀无梯度重建，最新图像索引保留输入梯度。"""
        batch, length = seed.images.shape[:2]
        hidden = seed.history.new_zeros(batch, self.hidden_dim)
        if isinstance(seed.images, HostDepthFrames):
            lengths = seed.images.lengths
            with torch.no_grad():
                for time in range(length - 1):
                    rows = [row for row, count in enumerate(lengths) if time < count - 1]
                    indices = torch.tensor(rows, device=hidden.device, dtype=torch.long)
                    features = self.encoder(seed.images.gather(rows, [time] * len(rows), hidden.device))
                    updated = self.gru(self.feature_norm(features), hidden.index_select(0, indices))
                    hidden = hidden.index_copy(0, indices, updated)
            image = seed.images.gather(range(batch), [count - 1 for count in lengths], hidden.device)
            return hidden.detach(), image
        final = seed.image_valid.long().sum(1) - 1
        with torch.no_grad():
            for time in range(length - 1):
                earlier = seed.image_valid[:, time] & (time < final)
                hidden, _ = self.encode(seed.images[:, time], hidden, earlier)
        rows = torch.arange(batch, device=hidden.device)
        image = seed.images[rows, final]
        return hidden.detach(), image

    @torch.jit.ignore
    def prepare(self, seed):
        """仅把固定形状输入交给更新图，不缓存旧编码器产生的 latent。"""
        hidden, image = self._prepare_visual(seed)
        return PreparedDepthSeed(seed.state, seed.history, seed.phase, hidden, image, seed.clean, seed.pixel_valid)

    @torch.jit.ignore
    def rebuild_prepared(self, seed):
        """最新 CNN、GRU 和投影位于可微训练核心内。"""
        features = self.encoder(seed.latest_image)
        hidden = self.gru(self.feature_norm(features), seed.prefix_hidden)
        return hidden, self.latent(hidden)

    @torch.jit.ignore
    def rebuild(self, seed):
        """从完整同回合前缀重建，最后实际图像及其 GRU 更新保留梯度。"""
        hidden, image = self._prepare_visual(seed)
        hidden = self.gru(self.feature_norm(self.encoder(image)), hidden)
        return hidden, self.latent(hidden)

    @torch.jit.ignore
    def perception_parameters(self):
        return (list(self.encoder.parameters()) + list(self.feature_norm.parameters())
                + list(self.gru.parameters()) + list(self.projection.parameters()))

    @torch.jit.ignore
    def policy_parameters(self):
        return self.parameters()

    @torch.jit.ignore
    def policy_normalizers(self):
        return (self.normalizer,)

    @torch.jit.ignore
    @torch.no_grad()
    def update_statistics(self, history: Tensor):
        self.normalizer.update(history.reshape(-1, self.history_length * self.observation_dim))


class DepthCritic(nn.Module):
    """训练专用特权信息、完整循环记忆和相机相位，不进入部署 Actor。"""

    def __init__(self, config):
        super().__init__()
        self.camera_period = config.camera_period
        width = config.privileged_dim + config.hidden_dim + 1
        self.normalizer = RunningNorm(width)
        self.net = Mlp([width, *config.critic_hidden, 1])

    def features(self, privileged: Tensor, hidden: Tensor, phase: Tensor) -> Tensor:
        return torch.cat((privileged, hidden, phase[..., None].to(hidden.dtype) / (self.camera_period - 1)), -1)

    def value(self, features: Tensor) -> Tensor:
        return self.net(self.normalizer(features)).squeeze(-1)

    def forward(self, privileged: Tensor, hidden: Tensor | None = None, phase: Tensor | None = None) -> Tensor:
        """单参数形式消费已拼接的拟合特征；三参数形式用于闭合想象状态。"""
        if hidden is None:
            return self.value(privileged)
        return self.value(self.features(privileged, hidden, phase))
