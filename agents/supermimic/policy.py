# Copyright (c) 2026-2027 zh
"""内容：提供含当前机身线速度估计的混合专家历史策略及价值网络。"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .world_model import RunningNorm


# ---------- 基础网络与混合专家 ----------

class Mlp(nn.Module):
    """使用ELU隐藏层和线性输出层的前馈网络。"""

    def __init__(self, dims, last_activation=False):
        super().__init__()
        layers, width = [], dims[0]
        for hidden_width in dims[1:-1]:
            layers.extend((nn.Linear(width, hidden_width), nn.ELU()))
            width = hidden_width
        layers.append(nn.Linear(width, dims[-1]))
        if last_activation:
            layers.append(nn.ELU())
        self.network = nn.Sequential(*layers)

    def forward(self, values: Tensor) -> Tensor:
        return self.network(values)


class L2Norm(nn.Module):
    """只把混合后的潜特征投影到单位球面。"""

    def forward(self, values: Tensor) -> Tensor:
        return F.normalize(values, p=2.0, dim=-1)


class Experts(nn.Module):
    """共享骨干后按专家分组的 1x1 卷积，各专家输出参数互相独立。"""

    def __init__(self, expert_num, input_dim, backbone_hidden_dims, expert_hidden_dim, output_dim):
        super().__init__()
        self.expert_num = expert_num
        self.output_dim = output_dim
        self.backbone = Mlp([input_dim, *backbone_hidden_dims, expert_num * expert_hidden_dim], last_activation=True)
        self.experts = nn.Conv1d(expert_num * expert_hidden_dim, expert_num * output_dim,
                                 kernel_size=1, groups=expert_num)

    def forward(self, values: Tensor) -> Tensor:
        shared = self.backbone(values).unsqueeze(-1)
        outputs = self.experts(shared).squeeze(-1)
        return outputs.reshape(-1, self.expert_num, self.output_dim)


class MoE(nn.Module):
    """门控网络对全部专家的潜特征与速度输出使用同一组权重融合。"""

    def __init__(self, expert_num, input_dim, hidden_dims, output_dim):
        super().__init__()
        self.experts = Experts(expert_num, input_dim, hidden_dims[:-1], hidden_dims[-1], output_dim)
        self.gating_network = nn.Sequential(
            Mlp([input_dim, *hidden_dims[:-1], expert_num]), nn.Softmax(dim=-1))

    def forward(self, values: Tensor) -> tuple[Tensor, Tensor]:
        weights = self.gating_network(values)
        experts = self.experts(values)
        return torch.sum(weights.unsqueeze(-1) * experts, dim=1), weights


class StudentMoEEncoder(nn.Module):
    """历史融合后拆分潜特征与当前机身线速度，只归一化潜特征。"""

    def __init__(self, expert_num, input_dim, hidden_dims, latent_dim):
        super().__init__()
        self.latent_dim = latent_dim
        self.moe = MoE(expert_num, input_dim, hidden_dims, latent_dim + 3)
        self.normalization = L2Norm()

    def forward(self, history: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        encoded, weights = self.moe(history)
        return self.normalization(encoded[:, :self.latent_dim]), encoded[:, self.latent_dim:], weights


class MoeActor(nn.Module):
    """盲本体历史编码为潜特征与机身线速度，再与当前帧拼接输出动作。

    历史与当前帧使用两套独立归一化统计；速度保持m/s量纲，
    与潜特征一同保留策略目标的梯度，并由真实状态提供速度监督。
    """

    def __init__(self, spec, config, observation_term_dims):
        super().__init__()
        if (not observation_term_dims or any(not isinstance(width, int) or isinstance(width, bool) or width < 1
                                            for width in observation_term_dims)
                or sum(observation_term_dims) != spec.observation_dim):
            raise ValueError("Observation term widths must be positive integers summing to observation_dim")
        self.observation_dim = spec.observation_dim
        self.history_length = spec.history_length
        self.action_dim = spec.action_dim
        self.latent_dim = config.moe_latent_dim
        self.velocity_dim = 3
        self.history_order = "frame_major_oldest_first"
        self.input_order = "latent_body_velocity_current_frame"
        self.normalizer = RunningNorm(self.observation_dim * self.history_length)
        self.frame_normalizer = RunningNorm(self.observation_dim)
        self.encoder = StudentMoEEncoder(
            config.moe_expert_num, self.observation_dim * self.history_length, tuple(config.moe_hidden),
            self.latent_dim)
        self.actor = Mlp([self.latent_dim + self.velocity_dim + self.observation_dim, *config.actor_hidden, self.action_dim])

    def policy_normalizers(self):
        """导出与播放校验需要检查的归一化模块。"""
        return (self.normalizer, self.frame_normalizer)

    def policy_parameters(self):
        """想象目标与真实速度监督共同更新完整历史策略。"""
        return self.parameters()

    def framed(self, history: Tensor) -> Tensor:
        """接受[B,T,D]或展平的[B,T*D]；导出图传入展平历史，训练传入分帧历史。"""
        return history if history.dim() == 3 else history.reshape(-1, self.history_length, self.observation_dim)

    def flatten_history(self, history: Tensor) -> Tensor:
        """外部历史按帧排列（旧到新）；展平后即专家的逐帧输入布局。"""
        return self.framed(history).reshape(-1, self.observation_dim * self.history_length)

    @torch.no_grad()
    def prepare_statistics(self, history: Tensor):
        """两套独立统计均以环境为样本轴，保留历史帧各位置的特征。"""
        history = self.framed(history)
        return (self.normalizer.prepare_update(self.flatten_history(history)),
                self.frame_normalizer.prepare_update(history[:, -1]))

    @torch.no_grad()
    def apply_statistics(self, statistics):
        """只提交本控制步已经统一校验的历史与当前帧统计。"""
        self.normalizer.apply_update(statistics[0])
        self.frame_normalizer.apply_update(statistics[1])

    @torch.no_grad()
    def update_statistics(self, history: Tensor):
        """独立入口即时校验；采集热路径通过准备／提交合并检查。"""
        statistics = self.prepare_statistics(history)
        if not torch.stack([item.finite for item in statistics]).all():
            raise FloatingPointError("Normalization data contains non-finite values")
        self.apply_statistics(statistics)

    @torch.jit.export
    def encode(self, history: Tensor) -> tuple[Tensor, Tensor]:
        """从包含当前帧的历史估计潜特征与同一时刻的机身xyz线速度。"""
        flattened = history.reshape(-1, self.observation_dim * self.history_length)
        latent, velocity, _ = self.encoder(self.normalizer(flattened))
        return latent, velocity

    def forward(self, history: Tensor) -> Tensor:
        """部署仍只接收本体历史，真实速度不进入策略输入。"""
        flattened = history.reshape(-1, self.observation_dim * self.history_length)
        latent, velocity = self.encode(flattened)
        current = self.frame_normalizer(flattened[:, -self.observation_dim:])
        return self.actor(torch.cat((latent, velocity, current), -1))


# ---------- 装配与价值网络 ----------

def make_actor(spec, config, observation_term_dims=None):
    """按明确模型标识装配混合专家历史策略。"""
    if spec.model_id == "supermimic_moe_velocity_v1":
        return MoeActor(spec, config, observation_term_dims)
    raise ValueError(f"Unsupported policy model identifier: {spec.model_id}")


class Critic(nn.Module):
    def __init__(self, width: int, hidden):
        super().__init__()
        self.normalizer = RunningNorm(width)
        self.net = Mlp([width, *hidden, 1])

    def forward(self, features: Tensor) -> Tensor:
        return self.net(self.normalizer(features)).squeeze(-1)
