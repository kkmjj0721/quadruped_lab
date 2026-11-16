# Copyright (c) 2026-2027 zh
"""内容：在原物理残差模型统计之上预测下一相机特征，并提供循环八步自回归监督。"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .compilation import compile_kernel
from .tensor_state import TaskState
from .optimization import frozen
from .normalization import RunningNorm
from .interfaces import PreparedDepthWindow
from .frame_storage import HostDepthFrames


class DepthWorldModel(nn.Module):
    """物理规则和误差尺度由任务提供；此模块只新增视觉条件与特征输出。"""

    def __init__(self, physical_model, actor, config, feature_coef=1.0):
        super().__init__()
        if (physical_model.task.model_input_dim != config.physical_input_dim
                or physical_model.spec.delta_dim != config.physical_output_dim
                or actor.latent_dim != config.latent_dim):
            raise ValueError("Physical model and visual network dimensions differ")
        # 保留任务侧双统计、接触 logit 尺度和误差下限，移除被新网络替代的权重。
        physical_model.net = nn.Identity()
        self.physical_model = physical_model
        self.task = physical_model.task
        self.spec = physical_model.spec
        self.physical_output_dim = config.physical_output_dim
        self.feature_dim = config.feature_dim
        self.camera_period = config.camera_period
        self.feature_coef = feature_coef
        self.visual_input_norm = RunningNorm(config.latent_dim + 1)
        self.feature_norm = RunningNorm(config.feature_dim)
        layers, width = [], config.physical_input_dim + config.latent_dim + 1
        for hidden in config.wm_hidden:
            layers.extend((nn.Linear(width, hidden), nn.LayerNorm(hidden), nn.SiLU()))
            width = hidden
        layers.append(nn.Linear(width, config.physical_output_dim + config.feature_dim))
        self.net = nn.Sequential(*layers)
        with torch.no_grad():
            self.net[-1].weight[:self.physical_output_dim].zero_()
            self.net[-1].bias[:self.physical_output_dim].zero_()
        self.use_bf16 = False
        self._compiled_components = None

    def configure_supervised_compile(self, *, enabled: bool, backend: str):
        """前缀与标签重编码位于图外，编译完整固定长度的循环模型损失。"""
        self._compiled_components = compile_kernel(self._loss_components, backend) if enabled else None

    def _visual_input(self, latent: Tensor, phase: Tensor) -> Tensor:
        return torch.cat((latent, phase[:, None].to(latent.dtype) / (self.camera_period - 1)), -1)

    def forward(self, state: TaskState, action: Tensor, latent: Tensor, phase: Tensor,
                recorded_current: TaskState | None = None):
        physical_input = (self.task.model_input(state, action) if recorded_current is None else
                          self.task.supervised_model_input(state, action, recorded_current))
        inputs = torch.cat((self.physical_model.input_norm(physical_input),
                            self.visual_input_norm(self._visual_input(latent, phase))), -1)
        if self.use_bf16:
            with torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
                outputs = self.net(inputs)
            outputs = outputs.float()
        else:
            outputs = self.net(inputs)
        delta = self.physical_model.output_norm.unnormalize(outputs[:, :self.physical_output_dim])
        feature = self.feature_norm.unnormalize(outputs[:, self.physical_output_dim:])
        physical = self.task.integrate(state, action, delta)
        return self.task.advance(state, action, physical), feature

    @torch.no_grad()
    def _feature_labels(self, window, actor):
        active = window.valid & window.new_frame
        indices = active.flatten().nonzero().flatten()
        if isinstance(window.next_images, HostDepthFrames):
            width = active.shape[1]
            positions = indices.cpu().tolist()
            images = window.next_images.gather([index // width for index in positions],
                                               [index % width for index in positions], window.actions.device)
        else:
            images = window.next_images.flatten(0, 1).index_select(0, indices)
        labels = window.actions.new_zeros(active.numel(), self.feature_dim, dtype=torch.float32)
        labels.index_copy_(0, indices, actor.encoder(images))
        return labels.reshape(*active.shape, self.feature_dim)

    @torch.no_grad()
    def update_statistics(self, window, actor):
        """提交真实统计，并返回本次已经重建的记忆和相机标签供模型更新复用。"""
        labels = self._feature_labels(window, actor)
        previous = TaskState(*(torch.cat((getattr(window.initial, name)[:, None],
                                         getattr(window.targets, name)[:, :-1]), 1).flatten(0, 1)
                               for name in ("physical", "memory", "command", "context")))
        target = window.targets.map(lambda values: values.flatten(0, 1))
        self.physical_model.update_stats(previous, window.actions.flatten(0, 1), target, window.valid.flatten())
        hidden, latent = actor.rebuild(window.seed)
        prepared = PreparedDepthWindow(window.initial, window.actions, window.targets, window.valid, window.new_frame,
                                       window.seed.phase, hidden.detach(), latent.detach(), labels)
        phase = window.seed.phase
        inputs = []
        for time in range(window.actions.shape[1]):
            inputs.append(self._visual_input(latent, phase))
            phase = (phase + 1) % self.camera_period
            hidden, latent = actor.update_memory(labels[:, time], hidden, window.new_frame[:, time] & window.valid[:, time])
        self.visual_input_norm.update(torch.stack(inputs, 1).flatten(0, 1), window.valid.flatten())
        self.feature_norm.update(labels.flatten(0, 1), (window.new_frame & window.valid).flatten())
        return prepared

    @torch.no_grad()
    def prepare(self, window, actor):
        """真实编码与可变新帧索引位于图外；准备结果不保留感知梯度。"""
        hidden, latent = actor.rebuild(window.seed)
        labels = self._feature_labels(window, actor)
        return PreparedDepthWindow(window.initial, window.actions, window.targets, window.valid, window.new_frame,
                                   window.seed.phase, hidden.detach(), latent.detach(), labels)

    def prepared_loss_components(self, window, actor):
        """冻结编码器参数但保留其对预测特征的输入导数，仅更新世界模型。"""
        with frozen(actor):
            kernel = self._loss_components if self._compiled_components is None else self._compiled_components
            return kernel(window, actor)

    def loss_components(self, window, actor):
        return self.prepared_loss_components(self.prepare(window, actor), actor)

    def loss(self, window, actor):
        physical, feature = self.loss_components(window, actor)
        return physical + self.feature_coef * feature

    def _loss_components(self, window, actor):
        hidden, latent, labels = window.hidden, window.latent, window.labels
        current = window.initial.map(lambda values: values.detach())
        recorded = window.initial.map(lambda values: values.detach())
        phase = window.phase
        physical_scale = self.physical_model.loss_scale()
        feature_scale = self.feature_norm.var.sqrt() + 1e-6
        physical_sum = current.physical.new_zeros(())
        feature_sum = current.physical.new_zeros(())
        for time in range(window.actions.shape[1]):
            target = window.targets.index((slice(None), time)).map(lambda values: values.detach())
            predicted, feature = self(current, window.actions[:, time].detach(), latent, phase, recorded)
            error = self.task.model_error(current, predicted, target, recorded) / physical_scale
            physical_sum = physical_sum + torch.where(window.valid[:, time], error.square().mean(-1), 0).sum()
            phase = (phase + 1) % self.camera_period
            refresh = window.valid[:, time] & window.new_frame[:, time]
            feature_error = ((feature - labels[:, time]) / feature_scale).square().mean(-1)
            feature_sum = feature_sum + torch.where(refresh, feature_error, 0).sum()
            hidden, latent = actor.update_memory(feature, hidden, (phase == 0) & window.valid[:, time])
            current = TaskState(predicted.physical, target.memory, target.command, target.context)
            recorded = target
        return (physical_sum / window.valid.sum().clamp_min(1),
                feature_sum / (window.valid & window.new_frame).sum().clamp_min(1))
