# Copyright (c) 2026-2027 zh
"""内容：以独占参数组联合更新深度策略和重建，分别拟合视觉物理模型与固定价值目标。"""

from contextlib import nullcontext
from dataclasses import replace
import math

import torch

from .compilation import compile_kernel
from .execution_config import ExecutionConfig
from .optimization import step_if_finite, clip_gradients, frozen
from .imagine import DepthImagination
from .optim import MuonAdam


class DepthLearner:
    """独立持有模型、策略及价值优化器，共享感知与解码器只属于策略复合优化器。"""

    _counter_names = ("actor_updates", "model_updates", "critic_updates", "nonfinite_steps")
    _schedule_names = ("lr_multiplier", "ramp_start", "actor_frozen_until", "policy_start")
    supports_cuda_graph = False

    def __init__(self, task, actor, decoder, model, critic, config, execution=None):
        self.task, self.actor, self.decoder = task, actor, decoder
        self.model, self.critic, self.config = model, critic, config
        self.execution = replace(execution) if execution is not None else ExecutionConfig()
        if self.execution.cuda_graph and not self.supports_cuda_graph:
            raise ValueError("Use CudaGraphDepthLearner for CUDA-graph updates")
        self._device_counts = torch.zeros(4, dtype=torch.int64, device=next(model.parameters()).device)
        self.actor_optimizer = self._make_optimizer("actor")
        self.model_optimizer = self._make_optimizer("model")
        self.critic_optimizer = self._make_optimizer("critic")
        self.imagination = DepthImagination(task, actor, model, critic, config)
        self.critic_loss_kernel = (compile_kernel(self._critic_loss, config.compile_backend)
                                   if config.compile_imagination else self._critic_loss)
        self.model.configure_supervised_compile(enabled=self.execution.wm_compile, backend=config.compile_backend)
        self.lr_multiplier = 1.0
        self.ramp_start = 0
        self.actor_frozen_until = 0
        self.policy_start = config.warmup_iterations
        self._measure_gradients = True

    # 恢复代码按字段赋值时沿用设备存储，热路径只在日志或保存边界读取。
    @property
    def actor_updates(self):
        return self._device_counts[0]

    @actor_updates.setter
    def actor_updates(self, value):
        self._device_counts[0].copy_(torch.as_tensor(value, device=self._device_counts.device))

    @property
    def model_updates(self):
        return self._device_counts[1]

    @model_updates.setter
    def model_updates(self, value):
        self._device_counts[1].copy_(torch.as_tensor(value, device=self._device_counts.device))

    @property
    def critic_updates(self):
        return self._device_counts[2]

    @critic_updates.setter
    def critic_updates(self, value):
        self._device_counts[2].copy_(torch.as_tensor(value, device=self._device_counts.device))

    @property
    def nonfinite_steps(self):
        return self._device_counts[3]

    @nonfinite_steps.setter
    def nonfinite_steps(self, value):
        self._device_counts[3].copy_(torch.as_tensor(value, device=self._device_counts.device))

    def _make_optimizer(self, name, *, capturable=False):
        cfg = self.config
        fused = True if self.execution.fused_adam else None
        if name == "actor":
            if cfg.actor_optimizer == "muon_adam":
                # 视觉与解码器仍由同一复合优化器的Adam独占，仅动作MLP矩阵使用Muon。
                module = torch.nn.ModuleList((self.actor, self.decoder))
                return MuonAdam(module, [self.actor.actor], cfg.learning_rate, betas=(0.9, 0.95),
                                fused=fused, capturable=capturable, io=True, weight_decay=cfg.muon_weight_decay)
            groups = [
                {"params": list(self.actor.actor.parameters()), "role": "actor"},
                {"params": self.actor.perception_parameters(), "role": "perception"},
                {"params": list(self.decoder.parameters()), "role": "decoder"},
            ]
            return torch.optim.Adam(groups, lr=cfg.learning_rate, betas=(0.9, 0.95),
                                    fused=fused, capturable=capturable)
        module = getattr(self, name)
        if name == "model" and cfg.wm_optimizer == "muon_adam":
            return MuonAdam(module, [module.net], cfg.learning_rate, betas=(0.9, 0.999),
                            fused=fused, capturable=capturable, weight_decay=cfg.muon_weight_decay)
        return torch.optim.Adam(module.parameters(), lr=cfg.learning_rate, betas=(0.9, 0.999),
                                fused=fused, capturable=capturable)

    def reset_policy_optimizers(self):
        """回滚后重建策略联合优化器和价值优化器，保留世界模型及其优化器。"""
        self.actor_optimizer = self._make_optimizer("actor")
        self.critic_optimizer = self._make_optimizer("critic")

    def schedule(self, iteration: int):
        """策略斜率从冻结结束、预热结束和回滚点三者最晚处起算。"""
        cfg = self.config
        progress = min(1.0, iteration / cfg.lr_decay_iterations)
        rate = cfg.learning_rate * (cfg.lr_final_frac + (1 - cfg.lr_final_frac) * .5 * (1 + math.cos(math.pi * progress)))
        rate *= self.lr_multiplier
        start = max(self.ramp_start, self.actor_frozen_until, self.policy_start)
        ramp = min(1.0, max(0.0, iteration - start) / max(1, cfg.actor_ramp_iterations))
        for group in self.actor_optimizer.param_groups:
            group["lr"] = rate * cfg.actor_lr_scale * ramp
        for optimizer in (self.model_optimizer, self.critic_optimizer):
            for group in optimizer.param_groups:
                group["lr"] = rate

    def _critic_loss(self, features, targets):
        return (self.critic(features) - targets).square().mean()

    def _critic_core(self, features, targets):
        """归一化和固定目标拟合共用同一核心；全部统计检查保持在设备上。"""
        cfg = self.config
        critic, optimizer = self.critic, self.critic_optimizer
        statistics = critic.normalizer.prepare_update(features)
        torch._assert_async(statistics.finite, "Normalization data contains non-finite values")
        critic.normalizer.apply_update(statistics._replace(
            count=torch.where(statistics.finite, statistics.count, 0)))
        losses, flags = [], []
        minibatch = features.shape[0] // cfg.critic_minibatches
        for _ in range(cfg.critic_epochs):
            indices = torch.randperm(features.shape[0], device=features.device)
            # 参考按整除的小批宽度取片，余数不参与本轮critic监督。
            for index in range(cfg.critic_minibatches):
                selected = indices[index * minibatch:(index + 1) * minibatch]
                optimizer.zero_grad(set_to_none=True)
                value_loss = features.new_zeros(())
                for part in selected.split(self.execution.critic_microbatch):
                    partial_loss = self.critic_loss_kernel(features[part], targets[part])
                    weighted_loss = partial_loss * (part.numel() / selected.numel())
                    weighted_loss.backward()
                    value_loss += weighted_loss.detach()
                    del partial_loss, weighted_loss
                gradient = clip_gradients(critic.parameters(), cfg.grad_clip)
                finite = torch.isfinite(gradient) & torch.isfinite(value_loss) & statistics.finite
                step_if_finite(optimizer, finite)
                flags.append(finite)
                losses.append(value_loss)
        return torch.stack(losses).mean(), torch.stack(flags).sum()

    def _run_critic_core(self, features, targets):
        return self._critic_core(features, targets)

    def _fit_critic(self, features, targets, *, timing=None):
        """只在图外检查批次结构，统计提交和优化共同计入每次真实更新。"""
        with timing.section("critic_prepare") if timing is not None else nullcontext():
            minibatch = features.shape[0] // self.config.critic_minibatches
            if minibatch == 0:
                raise ValueError("Imagined transitions must fill at least one row per critic minibatch")
        with timing.section("critic_update") if timing is not None else nullcontext():
            value_loss, updates = self._run_critic_core(features, targets)
        skipped = self.config.critic_epochs * self.config.critic_minibatches - updates
        return value_loss.clone(), skipped, updates.clone(), minibatch

    def _prepare_model(self, window):
        """各微批只准备一次视觉数据；所有真实统计提交后才交给模型更新。"""
        batch = window.actions.shape[0]
        parts = [window.index(slice(start, min(start + self.execution.imag_microbatch, batch)))
                 for start in range(0, batch, self.execution.imag_microbatch)]
        return tuple(self.model.update_statistics(part, self.actor) for part in parts)

    def _model_core(self, parts):
        """完整自回归模型梯度与更新；两类监督分别按全批有效数量归一化。"""
        self.model_optimizer.zero_grad(set_to_none=True)
        self.actor_optimizer.zero_grad(set_to_none=True)
        physical_count = sum(part.valid.sum() for part in parts).clamp_min(1)
        feature_count = sum((part.valid & part.new_frame).sum() for part in parts).clamp_min(1)
        physical_loss = parts[0].actions.new_zeros(())
        feature_loss = parts[0].actions.new_zeros(())
        for part in parts:
            physical, feature = self.model.prepared_loss_components(part, self.actor)
            physical_weighted = physical * part.valid.sum() / physical_count
            feature_weighted = feature * (part.valid & part.new_frame).sum() / feature_count
            (physical_weighted + self.config.feature_coef * feature_weighted).backward()
            physical_loss += physical_weighted.detach()
            feature_loss += feature_weighted.detach()
        loss = physical_loss + self.config.feature_coef * feature_loss
        gradient = clip_gradients(self.model.parameters(), self.config.grad_clip)
        finite = torch.isfinite(gradient) & torch.isfinite(loss)
        step_if_finite(self.model_optimizer, finite)
        return loss, physical_loss, feature_loss, gradient.detach(), finite

    def _run_model_core(self, parts):
        return self._model_core(parts)

    def fit_model(self, window):
        parts = self._prepare_model(window)
        loss, physical_loss, feature_loss, gradient, finite = self._run_model_core(parts)
        self.model_updates.add_(finite)
        self.nonfinite_steps.add_(~finite)
        return {"wm_loss": loss.detach().clone(), "wm_physical_loss": physical_loss.clone(),
                "wm_feature_loss": feature_loss.clone(), "wm_grad_norm": gradient.clone(), "wm_valid_steps": window.valid.sum(),
                "wm_feature_steps": (window.valid & window.new_frame).sum(), "wm_skipped": (~finite).float()}

    def _actor_core(self, seed, *, freeze_actor=False):
        """共享的完整三十二步策略、同一latent重建及联合感知反传核心。"""
        batch = seed.history.shape[0]
        cfg = self.config
        metrics = {}
        all_features, all_targets = [], []
        pixel_count = seed.pixel_valid.sum().clamp_min(1)
        perception = self.actor.perception_parameters()
        policy_contributions = [torch.zeros_like(parameter) for parameter in perception] if self._measure_gradients else []
        reconstruction_contributions = [torch.zeros_like(parameter) for parameter in perception] if self._measure_gradients else []
        self.actor_optimizer.zero_grad(set_to_none=True)
        self.model_optimizer.zero_grad(set_to_none=True)
        self.critic_optimizer.zero_grad(set_to_none=True)
        with frozen(self.model, self.critic):
            for start in range(0, batch, self.execution.imag_microbatch):
                end = min(start + self.execution.imag_microbatch, batch)
                part = seed.index(slice(start, end))
                weight = (end - start) / batch
                hidden, latent = self.actor.rebuild_prepared(part)
                data = self.imagination.rollout(part, initial_visual=(hidden, latent))
                targets = self.imagination.lambda_returns(data)
                control_term = -data.objective.mean() * weight
                reconstruction = self.decoder(latent)
                difference = torch.where(part.pixel_valid, reconstruction - part.clean.detach(), 0)
                reconstruction_term = difference.square().sum() / pixel_count
                weighted_reconstruction = cfg.reconstruction_coef * reconstruction_term
                # 首次更新测量两个实际损失对感知参数的贡献；不据此宣称主任务收益。
                if self._measure_gradients:
                    policy_gradient = torch.autograd.grad(control_term, perception, retain_graph=True, allow_unused=True)
                    reconstruction_gradient = torch.autograd.grad(weighted_reconstruction, perception,
                                                                 retain_graph=True, allow_unused=True)
                    for accumulated, value in zip(policy_contributions, policy_gradient):
                        if value is not None:
                            accumulated.add_(value.detach())
                    for accumulated, value in zip(reconstruction_contributions, reconstruction_gradient):
                        if value is not None:
                            accumulated.add_(value.detach())
                (control_term + weighted_reconstruction).backward()
                all_features.append(data.features[:, :-1].flatten(0, 1).detach())
                all_targets.append(targets.flatten().detach())
                values = {"control_loss": control_term.detach(),
                          "reconstruction_loss": reconstruction_term.detach(),
                          "actor_loss": (control_term + weighted_reconstruction).detach(),
                          "imagination/survival_h": data.survival.detach().mean() * weight,
                          "imagination/discounted_reward": data.discounted_reward.detach().mean() * weight,
                          "imagination/bootstrap": data.bootstrap.detach().mean() * weight,
                          "imagination/reward": data.rewards.mean() * weight}
                for name, value in values.items():
                    metrics[name] = metrics.get(name, 0) + value
            parameters = list(self.actor.parameters()) + list(self.decoder.parameters())
            gradient = clip_gradients(parameters, cfg.grad_clip, cfg.grad_clip_element)
            finite = torch.isfinite(gradient) & torch.isfinite(metrics["actor_loss"])
            if not freeze_actor:
                step_if_finite(self.actor_optimizer, finite)
        if self._measure_gradients:
            policy_squared = sum(value.square().sum() for value in policy_contributions)
            reconstruction_squared = sum(value.square().sum() for value in reconstruction_contributions)
            dot = sum((left * right).sum() for left, right in zip(policy_contributions, reconstruction_contributions))
            metrics.update({"perception/policy_grad_norm": policy_squared.sqrt(),
                            "perception/weighted_reconstruction_grad_norm": reconstruction_squared.sqrt(),
                            "perception/gradient_cosine": dot / (policy_squared * reconstruction_squared).sqrt().clamp_min(1e-12)})
            self._measure_gradients = False
        metrics["control_grad_norm"] = gradient.detach()
        features, targets = torch.cat(all_features), torch.cat(all_targets)
        return metrics, features, targets, finite

    def _run_actor_core(self, seed, *, freeze_actor=False):
        return self._actor_core(seed, freeze_actor=freeze_actor)

    def improve(self, seed, *, freeze_actor=False, timing=None):
        batch = seed.history.shape[0]
        if batch < 1:
            raise ValueError("Imagination seed batch must be nonempty")
        prepared = self.actor.prepare(seed)
        with timing.section("actor_update") if timing is not None else nullcontext():
            metrics, features, targets, finite = self._run_actor_core(prepared, freeze_actor=freeze_actor)
            self.nonfinite_steps.add_(~finite)
            if not freeze_actor:
                self.actor_updates.add_(finite)
        loss, skipped, updates, minibatch = (self._fit_critic(features, targets) if timing is None else
                                            self._fit_critic(features, targets, timing=timing))
        self.critic_updates.add_(updates)
        metrics = {name: value.clone() for name, value in metrics.items()}
        rate = self.actor_optimizer.param_groups[0]["lr"]
        metrics.update({"critic_loss": loss,
                        "control_skipped": (~finite).float(), "critic_skipped": skipped,
                        "control_learning_rate": rate.clone() if isinstance(rate, torch.Tensor) else rate,
                        "execution/imag_batch": batch,
                        "execution/imag_microbatch": min(batch, self.execution.imag_microbatch),
                        "execution/critic_microbatch": min(minibatch, self.execution.critic_microbatch),
                        "reconstruction_valid_pixels": seed.pixel_valid.sum(),
                        "nonfinite_steps": self.nonfinite_steps.clone()})
        return metrics

    def state_dict(self):
        state = {"actor": self.actor_optimizer.state_dict(), "model": self.model_optimizer.state_dict(),
                 "critic": self.critic_optimizer.state_dict()}
        state.update(zip(self._counter_names, self._device_counts.detach().cpu().tolist()))
        state.update({name: getattr(self, name) for name in self._schedule_names})
        return state

    def load_optimizer_state_dict(self, name, data):
        """恢复当前类型的优化器状态，执行精度、融合与设备学习率由当前运行器持有。"""
        optimizer = getattr(self, name + "_optimizer")
        if isinstance(optimizer, MuonAdam):
            optimizer.load_state_dict(data)
            return
        fused = True if self.execution.fused_adam else None
        capturable = self.execution.cuda_graph
        references = [group["lr"] for group in optimizer.param_groups] if capturable else []
        state = {**data, "param_groups": [{**group, "fused": fused, "capturable": capturable,
                                          "lr": float(group["lr"])} for group in data["param_groups"]]}
        optimizer.load_state_dict(state)
        if not fused and not capturable:
            for saved in optimizer.state.values():
                saved["step"] = saved["step"].cpu()
        if capturable:
            for group, reference in zip(optimizer.param_groups, references, strict=True):
                reference.fill_(group["lr"])
                group["lr"] = reference

    def load_state_dict(self, data):
        for name in ("actor", "model", "critic"):
            self.load_optimizer_state_dict(name, data[name])
        for name in self._counter_names:
            setattr(self, name, data[name])
        # 早期检查点未保存排期，沿用该检查点当时的默认排期。
        for name in self._schedule_names:
            if name in data:
                setattr(self, name, data[name])
