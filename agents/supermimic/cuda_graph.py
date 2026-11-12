# Copyright (c) 2026-2027 zh
"""内容：用静态输入和可捕获优化器执行世界模型、策略与价值更新，保持原训练预算。"""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
from dataclasses import fields, is_dataclass
import gc

import torch

from .buffer import Seed
from .learner import Learner, _step_if_finite, clip_gradients, frozen
from .optim import MuonAdam


# ---------- 静态输入与实际更新预热 ----------

@contextmanager
def _capture_without_gc():
    """仅捕获期间暂停循环回收，避免析构Triton内核；始终恢复原开关。"""
    enabled = gc.isenabled()
    gc.disable()
    try:
        yield
    finally:
        if enabled:
            gc.enable()


def _clone_inputs(value):
    """为图输入保留独立存储；递归保持 Window、Seed 和 TaskState 的协议。"""
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if is_dataclass(value):
        return type(value)(**{field.name: _clone_inputs(getattr(value, field.name)) for field in fields(value)})
    if isinstance(value, tuple):
        return tuple(_clone_inputs(item) for item in value)
    raise TypeError(f"Unsupported CUDA-graph input type: {type(value).__name__}")


@torch.no_grad()
def _copy_inputs(target, source):
    """只更新已捕获输入的值，不改变图使用的内存地址。"""
    if type(target) is not type(source):
        raise ValueError("CUDA-graph input structure changed")
    if isinstance(target, torch.Tensor):
        if target.shape != source.shape or target.dtype != source.dtype or target.device != source.device:
            raise ValueError("CUDA-graph input shape, dtype or device changed")
        target.copy_(source)
    elif is_dataclass(target):
        for field in fields(target):
            _copy_inputs(getattr(target, field.name), getattr(source, field.name))
    else:
        for destination, current in zip(target, source, strict=True):
            _copy_inputs(destination, current)


class CapturedUpdate:
    """前三次真实更新用于侧流预热，第四次捕获并重放；从不额外训练。"""

    def __init__(self, function, device: torch.device, *, warmup: int = 3):
        self.function, self.device, self.warmup = function, torch.device(device), warmup
        if self.device.type != "cuda":
            raise ValueError("CUDA-graph updates require a CUDA device")
        self.calls = self.replays = 0
        self.graph = self.inputs = self.outputs = None
        self.stream = torch.cuda.Stream(device=self.device)

    def __call__(self, *inputs):
        # torch.compile、无显式device的算子和图创建均使用同一目标设备。
        with torch.cuda.device(self.device):
            if self.inputs is None:
                self.inputs = _clone_inputs(inputs)
            else:
                _copy_inputs(self.inputs, inputs)
            self.calls += 1
            current = torch.cuda.current_stream(self.device)
            if self.calls <= self.warmup:
                self.stream.wait_stream(current)
                with torch.cuda.stream(self.stream):
                    outputs = self.function(*self.inputs)
                current.wait_stream(self.stream)
                return outputs
            if self.graph is None:
                torch.cuda.synchronize(self.device)
                with _capture_without_gc():
                    self.graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(self.graph, stream=self.stream):
                        self.outputs = self.function(*self.inputs)
                # 捕获只记录 CUDA 工作；下面这次 replay 才是本次实际更新。
                print(f"[INFO] Captured CUDA update graph: {self.function.__name__}", flush=True)
            self.graph.replay()
            self.replays += 1
            return self.outputs


def _capturable_adam(parameters, rate, *, betas=(0.9, 0.999)):
    parameters = list(parameters)
    return torch.optim.Adam(parameters, lr=torch.tensor(float(rate), device=parameters[0].device),
                            betas=betas, fused=True, capturable=True)


# ---------- 可捕获更新 ----------

class CudaGraphLearner(Learner):
    """图化计算保持基线损失、微批、更新次数、速度监督和护栏接口。"""

    def __init__(self, task, actor, model, critic, config, execution=None):
        self.device = next(model.parameters()).device
        if self.device.type != "cuda":
            raise ValueError("CUDA-graph updates require a CUDA device")
        if execution is not None and not execution.fused_adam:
            raise ValueError("CUDA-graph updates require fused Adam")
        self._actor_updates = {}
        self._critic_update = None
        super().__init__(task, actor, model, critic, config, execution)
        self.model_optimizer = self._make_optimizer("model", capturable=True)
        self._model_update = CapturedUpdate(self._model_core, self.device)

    def reset_policy_optimizers(self):
        """护栏重建优化器时同步失效旧 actor／critic 图，不动世界模型图。"""
        self.actor_optimizer = self._make_optimizer("actor", capturable=True)
        self.critic_optimizer = self._make_optimizer("critic", capturable=True)
        self._actor_updates = {}
        self._critic_update = None

    def schedule(self, iteration: int):
        rates = [(group, group["lr"]) for optimizer in
                 (self.actor_optimizer, self.model_optimizer, self.critic_optimizer) for group in optimizer.param_groups]
        super().schedule(iteration)
        for group, reference in rates:
            reference.fill_(group["lr"])
            group["lr"] = reference

    def load_optimizer_state_dict(self, name, data):
        """恢复时保留设备学习率地址，避免重放使用旧常量或旧优化器状态。"""
        optimizer = getattr(self, name + "_optimizer")
        if isinstance(optimizer, MuonAdam):
            optimizer.load_state_dict(data)
        else:
            if "muon" in data:
                raise ValueError("Cannot load MuonAdam state into Adam; use matching optimizer configuration")
            references = [group["lr"] for group in optimizer.param_groups]
            state = {**data, "param_groups": [{**group, "fused": True, "capturable": True}
                                              for group in data["param_groups"]]}
            optimizer.load_state_dict(state)
            for group, reference in zip(optimizer.param_groups, references, strict=True):
                reference.copy_(torch.as_tensor(group["lr"], device=self.device))
                group["lr"] = reference
        # 加载会替换优化器状态存储；重建所有更新图，避免继续引用旧地址。
        self._model_update = CapturedUpdate(self._model_core, self.device)
        self._actor_updates = {}
        self._critic_update = None

    def _model_core(self, window):
        self.model_optimizer.zero_grad(set_to_none=True)
        loss = self.model.loss(window)
        loss.backward()
        gradient = clip_gradients(self.model.parameters(), self.config.grad_clip)
        finite = torch.isfinite(gradient)
        _step_if_finite(self.model_optimizer, finite)
        return loss.detach(), gradient.detach(), window.valid.sum(), finite

    def fit_model(self, window):
        loss, gradient, valid, finite = self._model_update(window)
        self.model_updates.add_(finite)
        self.nonfinite_steps.add_(~finite)
        return {"wm_loss": loss.clone(), "wm_grad_norm": gradient.clone(),
                "wm_valid_steps": valid.clone(), "wm_skipped": (~finite).float(),
                "cuda_graph/model_captured": float(self._model_update.graph is not None),
                "cuda_graph/model_replays": self._model_update.replays}

    def _actor_core(self, seed, *, freeze_actor):
        cfg, batch = self.config, seed.history.shape[0]
        self.actor_optimizer.zero_grad(set_to_none=True)
        self.model_optimizer.zero_grad(set_to_none=True)
        self.critic_optimizer.zero_grad(set_to_none=True)
        feature_batches, target_batches, metrics = [], [], {}
        velocity_mse_axes = seed.history.new_zeros(3)
        with frozen(self.model, self.critic):
            for start in range(0, batch, self.execution.imag_microbatch):
                end = min(start + self.execution.imag_microbatch, batch)
                weight = (end - start) / batch
                part = Seed(seed.state.index(slice(start, end)), seed.history[start:end])
                data = self.imagination.rollout(part)
                targets = self.imagination.lambda_returns(data)
                control_loss = -data.objective.mean()
                _, velocity = self.actor.encode(part.history)
                target_velocity = self.task.body_linear_velocity(part.state).detach()
                velocity_error = (velocity - target_velocity).square()
                velocity_loss = velocity_error.mean()
                loss = control_loss + cfg.velocity_loss_coef * velocity_loss
                (loss * weight).backward()
                feature_batches.append(data.features[:, :-1].flatten(0, 1).detach())
                target_batches.append(targets.flatten().detach())
                velocity_mse_axes += velocity_error.detach().mean(0) * weight
                values = {"control_loss": control_loss.detach(), "velocity_loss": velocity_loss.detach(),
                          "actor_loss": loss.detach(), "imagination/survival_h": data.survival.detach().mean(),
                          "imagination/discounted_reward": data.discounted_reward.detach().mean(),
                          "imagination/bootstrap": data.bootstrap.detach().mean(),
                          "imagination/reward": data.rewards.mean()}
                for name, value in values.items():
                    metrics[name] = metrics.get(name, 0) + value * weight
                del data, targets, loss, control_loss, velocity_loss, velocity_error, velocity, target_velocity, values, part
            metrics["velocity_rmse"] = velocity_mse_axes.mean().sqrt()
            for axis, value in zip(("x", "y", "z"), velocity_mse_axes.sqrt()):
                metrics[f"velocity_rmse_{axis}"] = value
            gradient = clip_gradients(self.actor.policy_parameters(), cfg.grad_clip, cfg.grad_clip_element)
            finite = torch.isfinite(gradient) & torch.isfinite(metrics["actor_loss"])
            if not freeze_actor:
                _step_if_finite(self.actor_optimizer, finite)
        metrics["control_grad_norm"] = gradient.detach()
        return metrics, torch.cat(feature_batches), torch.cat(target_batches), finite

    def _critic_core(self, features, targets):
        cfg, optimizer = self.config, self.critic_optimizer
        # 每次真实critic更新仅提交一次统计；断言与提交随图重放，主机不再等待actor结果。
        statistics = self.critic.normalizer.prepare_update(features)
        torch._assert_async(statistics.finite, "Normalization data contains non-finite values")
        self.critic.normalizer.apply_update(statistics._replace(
            count=torch.where(statistics.finite, statistics.count, 0)))
        minibatch = features.shape[0] // cfg.critic_minibatches
        losses, flags = [], []
        for _ in range(cfg.critic_epochs):
            indices = torch.randperm(features.shape[0], device=features.device)
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
                gradient = clip_gradients(self.critic.parameters(), cfg.grad_clip)
                finite = torch.isfinite(gradient) & statistics.finite
                _step_if_finite(optimizer, finite)
                losses.append(value_loss)
                flags.append(finite)
        return torch.stack(losses).mean(), torch.stack(flags).sum()

    def timing_setup_pending(self, *, control_enabled=False, freeze_actor=False):
        """按本轮实际使用的图判断是否仍有预热／捕获，查询不创建或重放图。"""
        if self._model_update.graph is None:
            return True
        if not control_enabled:
            return False
        actor = self._actor_updates.get(freeze_actor)
        return (actor is None or actor.graph is None or self._critic_update is None
                or self._critic_update.graph is None)

    def improve(self, seed, *, freeze_actor=False, timing=None):
        batch = seed.history.shape[0]
        if batch < 1:
            raise ValueError("Imagination seed batch must be nonempty")
        if freeze_actor not in self._actor_updates:
            self._actor_updates[freeze_actor] = CapturedUpdate(
                lambda value: self._actor_core(value, freeze_actor=freeze_actor), self.device)
        actor_update = self._actor_updates[freeze_actor]
        with timing.section("actor_update") if timing is not None else nullcontext():
            metrics, features, targets, finite = actor_update(seed)
            self.nonfinite_steps.add_(~finite)
            if not freeze_actor:
                self.actor_updates.add_(finite)
        with timing.section("critic_prepare") if timing is not None else nullcontext():
            # 这里只处理静态形状和图对象；归一化及有限性断言在critic图中先于优化执行。
            minibatch = features.shape[0] // self.config.critic_minibatches
            if minibatch == 0:
                raise ValueError("Imagined transitions must fill at least one row per critic minibatch")
            if self._critic_update is None:
                self._critic_update = CapturedUpdate(self._critic_core, self.device)
        with timing.section("critic_update") if timing is not None else nullcontext():
            value_loss, count = self._critic_update(features, targets)
        skipped = self.config.critic_epochs * self.config.critic_minibatches - count
        self.critic_updates.add_(count)
        result = {name: value.clone() for name, value in metrics.items()}
        result.update({"critic_loss": value_loss.clone(), "control_skipped": (~finite).float(),
                       "critic_skipped": skipped, "control_learning_rate": self.actor_optimizer.param_groups[0]["lr"].clone(),
                       "execution/imag_batch": batch,
                       "execution/imag_microbatch": min(batch, self.execution.imag_microbatch),
                       "execution/critic_microbatch": min(self.execution.critic_microbatch, minibatch),
                       "nonfinite_steps": self.nonfinite_steps.clone(),
                       "cuda_graph/actor_captured": float(actor_update.graph is not None),
                       "cuda_graph/actor_replays": actor_update.replays,
                       "cuda_graph/critic_captured": float(self._critic_update.graph is not None),
                       "cuda_graph/critic_replays": self._critic_update.replays})
        return result
