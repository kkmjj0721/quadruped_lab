# Copyright (c) 2026-2027 zh
"""内容：将 MLP 历史估计、米制地图与锁存落足目标接入动作策略和纯部署组合。"""

from copy import deepcopy

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ..policy import HistoryActor, Mlp
from .interfaces import MODEL_ID


class VisualActor(HistoryActor):
    """采用用户指定的 256／128／64 隐藏层，估计 32 维 latent 与 3 维当前速度。"""

    def __init__(self, spec, config, observation_term_dims, map_shape=(17, 11), feet=4):
        if spec.model_id != MODEL_ID:
            raise ValueError("VisualActor requires the visual foothold model identifier")
        if len(map_shape) != 2 or any(width <= 0 for width in map_shape) or feet < 1:
            raise ValueError("Map dimensions and foot count must be positive")
        super().__init__(spec, observation_term_dims, latent_dim=32)
        self.encoder = Mlp([self.observation_dim * self.history_length, 256, 128, 64, self.velocity_dim + self.latent_dim])
        self.map_shape = tuple(map_shape)
        self.feet = feet
        self.input_order = "latent_body_velocity_current_frame_height_map_latched_targets_valid"
        width = self.latent_dim + self.velocity_dim + self.observation_dim + map_shape[0] * map_shape[1] + feet * 4
        self.actor = Mlp([width, *config.actor_hidden, self.action_dim])

    @torch.jit.export
    def encode(self, history: Tensor) -> tuple[Tensor, Tensor]:
        """按速度在前、latent 在后拆分；保留输入和参数梯度，仅归一化 latent。"""
        flattened = history.reshape(-1, self.observation_dim * self.history_length)
        parts = self.encoder(self.normalizer(flattened))
        return F.normalize(parts[:, self.velocity_dim:], p=2.0, dim=-1), parts[:, :self.velocity_dim]

    def forward(self, history: Tensor, height_map: Tensor, targets: Tensor, valid: Tensor) -> Tensor:
        flattened = self.flatten_history(history)
        latent, velocity = self.encode(flattened)
        current = self.frame_normalizer(flattened[:, -self.observation_dim:])
        # 无效目标可以带未初始化值；必须先屏蔽，再进入动作网络。
        targets = torch.where(valid.unsqueeze(-1), targets, torch.zeros_like(targets))
        inputs = torch.cat((latent, velocity, current, height_map.flatten(1), targets.flatten(1), valid.to(current.dtype)), -1)
        return self.actor(inputs)


class DeploymentPolicy(nn.Module):
    """部署不含 GT 分支；锁存及相机调度由调用方维护，不在 forward 中隐式改状态。"""

    def __init__(self, perception, planner, actor):
        super().__init__()
        self.perception = perception.inference_module()
        self.planner = deepcopy(planner)
        self.actor = deepcopy(actor)
        # 部署副本沿用CPU／TorchScript的FP32协议，训练对象的精度开关保持不变。
        self.perception.use_bf16 = self.planner.use_bf16 = False
        for module in self.actor.modules():
            if isinstance(module, Mlp):
                module.use_bf16 = False
        self.eval()

    @torch.jit.export
    def perceive(self, depth_pair: Tensor, history: Tensor, hidden: Tensor) -> tuple[Tensor, Tensor]:
        """新帧的完整感知；低频相机使用 encode_depth 和 step 的显式缓存接口。"""
        return self.perception(depth_pair, history, hidden)

    @torch.jit.export
    def encode_depth(self, depth_pair: Tensor) -> Tensor:
        return self.perception.encode_depth(depth_pair)

    @torch.jit.export
    def perceive_step(self, token: Tensor, history: Tensor, hidden: Tensor) -> tuple[Tensor, Tensor]:
        return self.perception.step(token, history, hidden)

    @torch.jit.export
    def propose(self, history: Tensor, command: Tensor, height_map: Tensor) -> Tensor:
        return self.planner.predict(history, command, height_map).targets

    def forward(self, history: Tensor, height_map: Tensor, targets: Tensor, valid: Tensor) -> Tensor:
        return self.actor(history, height_map, targets, valid)
