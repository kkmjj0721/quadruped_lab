# Copyright (c) 2026-2027 zh
"""内容：独立拟合物理、深度与落足，以完整想象回报和查询监督更新策略。"""

from contextlib import contextmanager, nullcontext
from dataclasses import replace
import math

import torch
from torch import nn

from .compilation import compile_kernel
from .config import ExecutionConfig
from .imagine import Imagination
from .optim import MuonAdam


@contextmanager
def frozen(*modules):
    parameters = [parameter for module in modules for parameter in module.parameters()]
    previous = [parameter.requires_grad for parameter in parameters]
    try:
        for parameter in parameters:
            parameter.requires_grad_(False)
        yield
    finally:
        for parameter, enabled in zip(parameters, previous):
            parameter.requires_grad_(enabled)


def clip_gradients(parameters, norm: float, element: float | None = None):
    parameters = list(parameters)
    value = nn.utils.clip_grad_norm_(parameters, norm)
    if element is not None:
        for parameter in parameters:
            if parameter.grad is not None:
                parameter.grad.clamp_(-element, element)
    return value


def _step_if_finite(optimizer, finite):
    """按优化器执行模式跳过非有限更新，组合优化器同时保护两套状态。"""
    if isinstance(optimizer, MuonAdam):
        optimizer.step(ok=finite)
    elif optimizer.param_groups[0]["fused"]:
        optimizer.found_inf = (~finite).float()
        try:
            optimizer.step()
        finally:
            optimizer.found_inf = None
    elif bool(finite):
        optimizer.step()


class Learner:
    _counter_names = ("actor_updates", "model_updates", "critic_updates", "nonfinite_steps")

    def __init__(self, task, actor, model, critic, config, execution=None, perception=None):
        self.task, self.actor, self.model, self.critic, self.config = task, actor, model, critic, config
        self.perception = perception
        self.execution = replace(execution) if execution is not None else ExecutionConfig()
        self._device_counts = torch.zeros(4, dtype=torch.int64, device=next(model.parameters()).device)
        self.reset_policy_optimizers()
        self.model_optimizer = self._make_optimizer("model")
        self.imagination = Imagination(task, actor, model, critic, config, perception)
        self.critic_loss_kernel = (compile_kernel(self._critic_loss, config.compile_backend)
                                   if config.compile_imagination else self._critic_loss)
        self.actor_updates = 0
        self.model_updates = 0
        self.critic_updates = 0
        self.nonfinite_steps = 0
        self.lr_multiplier = 1.0
        self.ramp_start = 0
        self.actor_frozen_until = 0
        self.policy_start = config.warmup_iterations
        self.auxiliary_updates = torch.zeros((), dtype=torch.long, device=self._device_counts.device)
        self.reset_auxiliary_optimizer()

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
        """按网络职责选择PartyMimic分组；critic始终使用Adam。"""
        cfg, module = self.config, getattr(self, name)
        kind = cfg.actor_optimizer if name == "actor" else cfg.wm_optimizer if name == "model" else "adam"
        betas = (0.9, 0.95) if name == "actor" else (0.9, 0.999)
        # 关闭时沿用原默认自动选择foreach；显式False会改为单张量更新。
        fused = True if self.execution.fused_adam else None
        if kind == "muon_adam":
            return MuonAdam(module, [module] if name == "actor" else [module.net], cfg.learning_rate,
                            betas=betas, fused=fused, capturable=capturable, io=name == "actor",
                            weight_decay=cfg.muon_weight_decay)
        parameters = list(module.policy_parameters() if name == "actor" else module.parameters())
        rate = torch.tensor(cfg.learning_rate, device=parameters[0].device) if capturable else cfg.learning_rate
        return torch.optim.Adam(parameters, lr=rate, betas=betas, fused=fused, capturable=capturable)

    def reset_policy_optimizers(self):
        """重建策略和价值优化器；世界模型优化器由其独立生命周期管理。"""
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
        for optimizer in (self.model_optimizer, self.critic_optimizer, self.auxiliary_optimizer):
            for group in optimizer.param_groups:
                group["lr"] = rate

    def fit_model(self, window):
        self.model_optimizer.zero_grad(set_to_none=True)
        loss = self.model.loss(window)
        loss.backward()
        gradient = clip_gradients(self.model.parameters(), self.config.grad_clip)
        finite = torch.isfinite(gradient)
        _step_if_finite(self.model_optimizer, finite)
        self.model_updates.add_(finite)
        self.nonfinite_steps.add_(~finite)
        return {"wm_loss": loss.detach(), "wm_grad_norm": gradient.detach(),
                "wm_valid_steps": window.valid.sum(), "wm_skipped": (~finite).float()}

    def _critic_loss(self, features, targets):
        return (self.critic(features) - targets).square().mean()

    def timing_setup_pending(self, *, control_enabled=False, freeze_actor=False):
        """普通执行没有独立的Graph预热／捕获阶段。"""
        return False

    def _fit_critic(self, features, targets, *, timing=None):
        """以固定想象目标拟合价值网络，保留尾批与分批累积语义。"""
        cfg = self.config
        critic, optimizer = self.critic, self.critic_optimizer
        with timing.section("critic_prepare") if timing is not None else nullcontext():
            critic.normalizer.update(features)
            losses, flags = [], []
            minibatch = features.shape[0] // cfg.critic_minibatches
            if minibatch == 0:
                raise ValueError("Imagined transitions must fill at least one row per critic minibatch")
        with timing.section("critic_update") if timing is not None else nullcontext():
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
                    finite = torch.isfinite(gradient)
                    _step_if_finite(optimizer, finite)
                    flags.append(finite)
                    losses.append(value_loss)
            updates = torch.stack(flags).sum()
            skipped = cfg.critic_epochs * cfg.critic_minibatches - updates
            value_loss = torch.stack(losses).mean()
        return value_loss, skipped, updates, minibatch

    def reset_auxiliary_optimizer(self):
        fused = True if self.execution.fused_adam and self._device_counts.device.type == "cuda" else None
        self.auxiliary_optimizer = torch.optim.Adam(self.perception.parameters(), lr=self.config.learning_rate, fused=fused)

    def fit_auxiliary(self, batch):
        """固定八次相机刷新开环监督；CNN仅重建、FPM仅真实未来落足。"""
        device = self._device_counts.device
        noisy = batch.seed.images.to(device=device, dtype=torch.float32, non_blocking=True)
        clean = batch.reconstruction_clean.to(device=device, dtype=torch.float32, non_blocking=True)
        valid = batch.reconstruction_valid.to(device=device, dtype=torch.float32, non_blocking=True)
        b, k = noisy.shape[:2]
        encoded = self.perception.encoder(noisy.flatten(0, 1))
        reconstruction = self.perception.decoder(encoded).reshape_as(clean)
        reconstruction_loss = ((reconstruction - clean).abs() * valid).sum() / valid.sum().clamp_min(1)
        baseline = ((noisy - clean).abs() * valid).sum() / valid.sum().clamp_min(1)
        tokens = encoded.detach().reshape(b, k * 16, 16)
        future = batch.refresh_images.to(device=device, dtype=torch.float32, non_blocking=True)
        with torch.no_grad():
            target = self.perception.encoder(future.flatten(0, 1)).reshape(b, -1, 16, 16)
        xy_prediction = self.perception.points(self.actor.normalized_history(batch.seed.history).detach())
        xy_error = (xy_prediction - batch.seed.future_xy).square().mean(-1)
        xy_valid = batch.seed.future_valid
        xy_loss = (xy_error * xy_valid).sum() / xy_valid.sum().clamp_min(1)
        latent_losses, copy_losses = [], []
        initial_last = tokens[:, -16:]
        for index in range(future.shape[1]):
            state = batch.refresh_states.index((slice(None), index))
            prediction = self.perception.predict_frame(tokens, self.task.physical_features(state),
                                                        batch.refresh_actions[:, index])
            latent_losses.append((prediction - target[:, index]).square().mean((1, 2)))
            copy_losses.append((initial_last - target[:, index]).square().mean((1, 2)))
            shifted = torch.cat((tokens[:, 16:], prediction), 1)
            tokens = torch.where(batch.refresh_valid[:, index, None, None], shifted, tokens)
        mask = batch.refresh_valid
        latent_loss = (torch.stack(latent_losses, 1) * mask).sum() / mask.sum().clamp_min(1)
        copy_loss = (torch.stack(copy_losses, 1) * mask).sum() / mask.sum().clamp_min(1)
        loss = reconstruction_loss + latent_loss + xy_loss
        self.auxiliary_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient = clip_gradients(self.perception.parameters(), self.config.grad_clip)
        finite = torch.isfinite(loss) & torch.isfinite(gradient)
        _step_if_finite(self.auxiliary_optimizer, finite)
        self.auxiliary_updates.add_(finite)
        self.nonfinite_steps.add_(~finite)
        return {"perception/loss": loss.detach(), "perception/reconstruction": reconstruction_loss.detach(),
                "perception/noisy_baseline": baseline.detach(), "perception/latent": latent_loss.detach(),
                "perception/copy_baseline": copy_loss.detach(), "perception/foothold": xy_loss.detach(),
                "perception/label_fraction": xy_valid.float().mean(), "perception/skipped": (~finite).float()}

    def _actor_core(self, seed, *, freeze_actor):
        cfg, batch = self.config, seed.history.shape[0]
        self.actor_optimizer.zero_grad(set_to_none=True)
        self.model_optimizer.zero_grad(set_to_none=True)
        self.critic_optimizer.zero_grad(set_to_none=True)
        feature_batches, target_batches, metrics = [], [], {}
        read_count = seed.read_valid.sum().clamp_min(1)
        with frozen(self.model, self.critic, self.perception):
            for start in range(0, batch, self.execution.imag_microbatch):
                end = min(start + self.execution.imag_microbatch, batch)
                weight = (end - start) / batch
                part = seed.index(slice(start, end))
                data = self.imagination.rollout(part)
                targets = self.imagination.lambda_returns(data)
                control_loss = -data.objective.mean()
                points = self.perception.points(self.actor.normalized_history(part.history)).detach()
                read_loss = self.actor.read_loss_sum(part.history, part.tokens, points,
                                                      part.read_targets, part.read_valid) / read_count
                # 主回报按样本数加权；读取监督按整批有效标签数归一化，独立于微批划分。
                loss = control_loss * weight + cfg.read_loss_coef * read_loss
                loss.backward()
                feature_batches.append(data.features[:, :-1].flatten(0, 1).detach())
                target_batches.append(targets.flatten().detach())
                values = {"control_loss": control_loss.detach() * weight, "actor_loss": loss.detach(),
                          "perception/read_loss": read_loss.detach(),
                          "imagination/survival_h": data.survival.detach().mean() * weight,
                          "imagination/discounted_reward": data.discounted_reward.detach().mean() * weight,
                          "imagination/bootstrap": data.bootstrap.detach().mean() * weight,
                          "imagination/reward": data.rewards.mean() * weight}
                for name, value in values.items():
                    metrics[name] = metrics.get(name, 0) + value
            gradient = clip_gradients(self.actor.policy_parameters(), cfg.grad_clip, cfg.grad_clip_element)
            finite = torch.isfinite(gradient) & torch.isfinite(metrics["actor_loss"])
            if not freeze_actor:
                _step_if_finite(self.actor_optimizer, finite)
        metrics["control_grad_norm"] = gradient.detach()
        features = feature_batches[0] if len(feature_batches) == 1 else torch.cat(feature_batches)
        targets = target_batches[0] if len(target_batches) == 1 else torch.cat(target_batches)
        return metrics, features, targets, finite

    def improve(self, seed, *, freeze_actor=False, timing=None):
        metrics, features, targets, finite = self._actor_core(seed, freeze_actor=freeze_actor)
        self.nonfinite_steps.add_(~finite)
        if not freeze_actor:
            self.actor_updates.add_(finite)
        value_loss, skipped, updates, minibatch = self._fit_critic(features, targets, timing=timing)
        self.critic_updates.add_(updates)
        metrics.update({"critic_loss": value_loss.detach(), "control_skipped": (~finite).float(),
                        "critic_skipped": skipped, "control_learning_rate": self.actor_optimizer.param_groups[0]["lr"],
                        "execution/imag_batch": seed.history.shape[0],
                        "execution/imag_microbatch": min(seed.history.shape[0], self.execution.imag_microbatch),
                        "execution/critic_microbatch": min(self.execution.critic_microbatch, minibatch),
                        "nonfinite_steps": self.nonfinite_steps.clone()})
        return metrics

    def state_dict(self):
        state = {"actor": self.actor_optimizer.state_dict(), "model": self.model_optimizer.state_dict(),
                 "critic": self.critic_optimizer.state_dict(), "auxiliary": self.auxiliary_optimizer.state_dict(),
                 "auxiliary_updates": int(self.auxiliary_updates.item())}
        state.update(zip(self._counter_names, self._device_counts.detach().cpu().tolist()))
        return state

    def load_optimizer_state_dict(self, name, data):
        """沿用当前融合执行选项，加载原步数、动量和其余优化器参数。"""
        optimizer = getattr(self, name + "_optimizer")
        if isinstance(optimizer, MuonAdam):
            optimizer.load_state_dict(data)
            return
        if "muon" in data:
            raise ValueError("Cannot load MuonAdam state into Adam; use matching optimizer configuration")
        # 必须在加载前覆盖：PyTorch据此把fused Adam的step张量放到参数设备。
        fused = True if self.execution.fused_adam else None
        state = {**data, "param_groups": [{**group, "fused": fused, "capturable": False,
                                          "lr": float(group["lr"])} for group in data["param_groups"]]}
        optimizer.load_state_dict(state)
        if not fused:
            # PyTorch加载时保留非捕获step原设备，跨GPU/CPU恢复须回到普通Adam的CPU计数。
            for saved in optimizer.state.values():
                saved["step"] = saved["step"].cpu()

    def load_state_dict(self, data):
        for name in ("actor", "model", "critic"):
            self.load_optimizer_state_dict(name, data[name])
        for name in ("actor_updates", "model_updates", "critic_updates"):
            setattr(self, name, data[name])
