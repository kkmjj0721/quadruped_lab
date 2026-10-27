# Copyright (c) 2026-2027 zh
"""内容：提供混合专家历史策略、价值网络和物理动力学网络。"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ..interfaces import TaskState


class RunningNorm(nn.Module):
    """按参考实现合并总体方差；前向固定统计，不隐式更新。"""

    def __init__(self, width: int):
        super().__init__()
        self.register_buffer("mean", torch.zeros(width))
        self.register_buffer("var", torch.ones(width))
        self.register_buffer("count", torch.tensor(1e-4))

    @torch.no_grad()
    def update(self, values: Tensor):
        values = values.reshape(-1, self.mean.numel())
        if values.shape[0] == 0:
            return
        if not torch.isfinite(values).all():
            raise FloatingPointError("Normalization data contains non-finite values")
        variance, mean = torch.var_mean(values, dim=0, correction=0)
        count = values.shape[0]
        total = self.count + count
        difference = mean - self.mean
        self.var.copy_((self.var * self.count + variance * count + difference.square() * self.count * count / total) / total)
        self.mean.add_(difference * count / total)
        self.count.copy_(total)

    def forward(self, values: Tensor) -> Tensor:
        return ((values - self.mean) / (self.var.sqrt() + 1e-6)).clamp(-10.0, 10.0)

    def unnormalize(self, values: Tensor) -> Tensor:
        return values * (self.var.sqrt() + 1e-6) + self.mean


# ---------- 混合专家编码器 ----------


class Mlp(nn.Module):
    """与 go2_rl_gym 的 MLP 同名同层级，使专家权重可原样热启动。"""

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
    """把专家输出投影到单位球面；对应 go2_rl_gym 的 norm_type='l2norm'。"""

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
    """门控网络对专家输出加权求和，并返回门控权重供负载均衡监督。"""

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
    """完整历史经共享骨干、分组专家和门控得到定长隐变量。"""

    def __init__(self, expert_num, input_dim, hidden_dims, output_dim, output_normalization):
        super().__init__()
        self.moe = MoE(expert_num, input_dim, hidden_dims, output_dim)
        self.normalization = output_normalization

    def forward(self, history: Tensor) -> tuple[Tensor, Tensor]:
        latent, weights = self.moe(history)
        return self.normalization(latent), weights


class MoeActor(nn.Module):
    """盲本体历史经混合专家编码为隐变量，再与当前帧拼接输出动作。

    历史与当前帧使用两套独立归一化统计：历史统计喂给专家编码器，当前帧统计
    只用于动作网络输入，与 go2_rl_gym 的 student 编码器和单帧归一化分工一致。
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
        self.history_order = "frame_major_oldest_first"
        self.input_order = "frame_major_oldest_first"
        self.normalizer = RunningNorm(self.observation_dim * self.history_length)
        self.frame_normalizer = RunningNorm(self.observation_dim)
        self.encoder = StudentMoEEncoder(
            config.moe_expert_num, self.observation_dim * self.history_length, tuple(config.moe_hidden),
            config.moe_latent_dim, L2Norm())
        self.actor = Mlp([config.moe_latent_dim + self.observation_dim, *config.actor_hidden, self.action_dim])

    def policy_normalizers(self):
        """导出与播放校验需要检查的归一化模块。"""
        return (self.normalizer, self.frame_normalizer)

    def framed(self, history: Tensor) -> Tensor:
        """接受[B,T,D]或展平的[B,T*D]；导出图传入展平历史，训练传入分帧历史。"""
        return history if history.dim() == 3 else history.reshape(-1, self.history_length, self.observation_dim)

    def flatten_history(self, history: Tensor) -> Tensor:
        """外部历史按帧排列（旧到新）；展平后即专家的逐帧输入布局。"""
        return self.framed(history).reshape(-1, self.observation_dim * self.history_length)

    @torch.no_grad()
    def update_statistics(self, history: Tensor):
        """历史与当前帧分别累计统计；采集热路径每步只调用一次。"""
        history = self.framed(history)
        self.normalizer.update(self.flatten_history(history))
        self.frame_normalizer.update(history[:, -1])

    def forward(self, history: Tensor) -> Tensor:
        """导出的部署图只接收展平历史，因此这里只按展平历史解释输入。"""
        flattened = history.reshape(-1, self.observation_dim * self.history_length)
        latent, _ = self.encoder(self.normalizer(flattened))
        return self.actor(torch.cat((latent, self.frame_normalizer(flattened[:, -self.observation_dim:])), -1))

    def load_balance_loss(self, history: Tensor) -> tuple[Tensor, Tensor]:
        """门控使用率偏离均匀分布的平方误差，并返回最大使用率便于监控塌缩。"""
        _, weights = self.encoder(self.normalizer(self.flatten_history(history)))
        usage = weights.mean(dim=0)
        target = torch.full_like(usage, 1.0 / weights.shape[1])
        return (usage - target).square().mean(), usage.max()

    @torch.no_grad()
    def load_pretrained(self, state_dict):
        """按 go2_rl_gym 导出名的展平命名载入专家与动作网络，保留本地归一化统计。

        导出的 TorchScript 把 MLP 的 network 容器展平，因此这里显式重建映射；
        缺少或不匹配的张量直接报错，避免静默半载入。
        """
        def pairs(local_prefix, remote_prefix):
            return {f"{local_prefix}.{kind}": f"{remote_prefix}.{kind}" for kind in ("weight", "bias")}

        def linears(module):
            return [index for index, layer in enumerate(module) if isinstance(layer, nn.Linear)]

        mapping = {}
        backbone = linears(self.encoder.moe.experts.backbone.network)
        # 展平导出把骨干最后一层并入了 experts_hidden，其余线性层按序号一一对应。
        for index in backbone[:-1]:
            mapping.update(pairs(f"encoder.moe.experts.backbone.network.{index}",
                                 f"student_moe_encoder.experts_backbone.{index}"))
        mapping.update(pairs(f"encoder.moe.experts.backbone.network.{backbone[-1]}",
                             "student_moe_encoder.experts_hidden.0"))
        mapping.update(pairs("encoder.moe.experts.experts", "student_moe_encoder.experts_out"))
        for index in linears(self.encoder.moe.gating_network[0].network):
            mapping.update(pairs(f"encoder.moe.gating_network.0.network.{index}",
                                 f"student_moe_encoder.gating_network.{index}"))
        for index in linears(self.actor.network):
            mapping.update(pairs(f"actor.network.{index}", f"actor.{index}"))
        current = self.state_dict()
        missing = set(mapping.values()) - set(state_dict)
        if missing:
            raise ValueError(f"Pretrained policy is missing tensors: {sorted(missing)}")
        merged = dict(current)
        for name, source in mapping.items():
            value = state_dict[source]
            if current[name].shape != value.shape:
                raise ValueError(f"Pretrained tensor {source} has shape {tuple(value.shape)}, "
                                 f"expected {tuple(current[name].shape)}")
            merged[name] = value
        self.load_state_dict(merged, strict=True)


def make_actor(spec, config, observation_term_dims=None):
    """装配唯一的混合专家盲策略。"""
    return MoeActor(spec, config, observation_term_dims)


class Critic(nn.Module):
    def __init__(self, width: int, hidden):
        super().__init__()
        self.normalizer = RunningNorm(width)
        self.net = Mlp([width, *hidden, 1])

    def forward(self, features: Tensor) -> Tensor:
        return self.net(self.normalizer(features)).squeeze(-1)


class WorldModel(nn.Module):
    """任务决定残差坐标；网络只学习记录物理转移，不学习奖励或失败头。"""

    def __init__(self, task, hidden):
        super().__init__()
        self.task = task
        self.spec = task.spec
        layers, width = [], task.model_input_dim
        for hidden_width in hidden:
            layers.extend((nn.Linear(width, hidden_width), nn.SiLU()))
            width = hidden_width
        layers.append(nn.Linear(width, task.spec.delta_dim))
        self.net = nn.Sequential(*layers)
        self.input_norm = RunningNorm(task.model_input_dim)
        self.output_norm = RunningNorm(task.spec.delta_dim)

    @torch.no_grad()
    def update_stats(self, state: TaskState, action: Tensor, target: TaskState):
        self.input_norm.update(self.task.model_input(state, action))
        self.output_norm.update(self.task.model_target(state, target))

    def forward(self, state: TaskState, action: Tensor, recorded_current: TaskState | None = None) -> TaskState:
        inputs = (self.task.model_input(state, action) if recorded_current is None
                  else self.task.supervised_model_input(state, action, recorded_current))
        delta = self.output_norm.unnormalize(self.net(self.input_norm(inputs)))
        physical = self.task.integrate(state, action, delta)
        return self.task.advance(state, action, physical)

    def loss(self, window) -> Tensor:
        current = window.initial
        recorded_current = window.initial
        scale = self.output_norm.var.sqrt() + 1e-6
        total = current.physical.new_zeros(())
        for step in range(window.actions.shape[1]):
            target = window.targets.index((slice(None), step))
            action = window.actions[:, step]
            predicted = self(current, action, recorded_current)
            error = self.task.model_error(current, predicted, target, recorded_current) / scale
            total = total + (error.square().mean(-1) * window.valid[:, step]).sum()
            # 本协议的记忆只含记录动作与外生计时，按日志恢复命令重采样；物理状态始终回喂预测。
            current = TaskState(predicted.physical, target.memory, target.command, target.context)
            recorded_current = target
        return total / window.valid.sum().clamp_min(1)
