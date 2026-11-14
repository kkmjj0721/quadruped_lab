# Copyright (c) 2026-2027 zh
"""内容：组织视觉辅助监督、当前权重的记忆重建、已有 BPTT 学习器和独立检查点。"""

from dataclasses import asdict
from pathlib import Path

import torch

from agents.foothold import foothold_loss
from agents.perception import map_loss

from ..learner import Learner, clip_gradients
from ..policy import Mlp
from .config import AuxiliaryConfig, VisualExecutionConfig
from .imagine import VisualImagination
from .interfaces import CameraState, MODEL_ID, VisualMemory, VisualSeed
from .policy import DeploymentPolicy


class VisualLearner(Learner):
    """环境无关的视觉算法入口；真实采集、回放存储和任务事件仍由调用方负责。"""

    def __init__(self, task, actor, model, critic, perception, planner, config, auxiliary=None, execution=None):
        execution = (VisualExecutionConfig(**asdict(execution)) if execution is not None else
                     VisualExecutionConfig(cuda_graph=False, wm_compile=False, fused_adam=False,
                                           wm_bf16=True, actor_bf16=True, critic_bf16=True,
                                           perception_bf16=True, planner_bf16=True))
        if task.spec.model_id != MODEL_ID:
            raise ValueError("VisualLearner requires the visual foothold model identifier")
        if config.compile_imagination or execution.cuda_graph:
            raise ValueError("VisualLearner requires compile_imagination=False and cuda_graph=False")
        for dimensions in (perception.config, planner):
            if (dimensions.proprio_dim != task.spec.observation_dim
                    or dimensions.history_length != task.spec.history_length):
                raise ValueError("Visual modules must match task proprioception and history dimensions")
        if tuple(actor.map_shape) != tuple(perception.config.map_shape) or actor.feet != 4:
            raise ValueError("Visual actor and perception map shapes must agree")
        if planner.command_dim != task.spec.command_dim:
            raise ValueError("Planner and task command dimensions must agree")
        device = next(model.parameters()).device
        if any(next(module.parameters()).device != device for module in (actor, critic, perception, planner)):
            raise ValueError("All visual algorithm modules must share a device")
        if any((execution.wm_bf16, execution.actor_bf16, execution.critic_bf16,
                execution.perception_bf16, execution.planner_bf16)):
            if device.type != "cuda":
                raise ValueError("BF16 network execution requires a native BF16 CUDA device")
            with torch.cuda.device(device):
                if not torch.cuda.is_bf16_supported(including_emulation=False):
                    raise ValueError("BF16 network execution requires native BF16 support on the selected CUDA device")
        for method in ("update_camera", "update_visual_targets", "visual_privileged"):
            if not callable(getattr(task, method, None)):
                raise TypeError(f"Visual task must implement {method}")
        model.use_bf16 = execution.wm_bf16
        for root, enabled in ((actor, execution.actor_bf16), (critic, execution.critic_bf16)):
            for module in root.modules():
                if isinstance(module, Mlp):
                    module.use_bf16 = enabled
        perception.configure_bf16(execution.perception_bf16)
        planner.use_bf16 = execution.planner_bf16
        super().__init__(task, actor, model, critic, config, execution)
        self.perception, self.planner = perception, planner
        self.auxiliary = auxiliary or AuxiliaryConfig()
        self.perception_optimizer = torch.optim.Adam(perception.parameters(), lr=self.auxiliary.perception_lr,
                                                     weight_decay=self.auxiliary.perception_weight_decay)
        self.planner_optimizer = torch.optim.Adam(planner.parameters(), lr=self.auxiliary.planner_lr)
        self.perception_version = self.planner_updates = 0
        self.imagination = VisualImagination(task, actor, model, critic, perception, planner, config)

    # ---------- 感知序列与当前权重记忆 ----------

    def _validate_prefix(self, prefix):
        cfg = self.perception.config
        histories = prefix.histories
        if histories.ndim != 4 or histories.shape[2:] != (cfg.history_length, cfg.proprio_dim):
            raise ValueError("Prefix histories must have shape [B,T,H,D]")
        batch, steps = histories.shape[:2]
        if batch < 1 or prefix.depth_pairs.shape != (batch, steps, 2, *cfg.depth_shape):
            raise ValueError("Prefix depth pairs must have shape [B,T,2,16,16] with a nonempty batch")
        for mask in (prefix.fresh, prefix.valid):
            if mask.shape != (batch, steps) or mask.dtype != torch.bool:
                raise ValueError("Prefix masks must be boolean [B,T]")
        tensors = (histories, prefix.depth_pairs, prefix.fresh, prefix.valid)
        device = next(self.perception.parameters()).device
        if any(value.device != device for value in tensors):
            raise ValueError("Prefix tensors must be on the perception device")
        if histories.dtype != torch.float32 or prefix.depth_pairs.dtype != torch.float32:
            raise ValueError("Prefix observations must use float32")
        if steps and ((prefix.valid[:, 1:] & ~prefix.valid[:, :-1]).any()
                      or (prefix.valid[:, 0] & ~prefix.fresh[:, 0]).any()):
            raise ValueError("Prefix must start at a fresh episode frame and use trailing padding only")
        if not torch.isfinite(histories[prefix.valid]).all() or not torch.isfinite(prefix.depth_pairs[prefix.valid & prefix.fresh]).all():
            raise ValueError("Valid prefix observations must be finite")

    def _sequence(self, prefix):
        """当前权重完整重放；缓存 token 保留到下次新帧，填充行不推进记忆。"""
        cfg = self.perception.config
        batch, steps = prefix.histories.shape[:2]
        token = prefix.histories.new_zeros(batch, cfg.token_dim)
        hidden = prefix.histories.new_zeros(batch, cfg.hidden_dim)
        maps = []
        for step in range(steps):
            active = prefix.valid[:, step]
            fresh = active & prefix.fresh[:, step]
            if fresh.any():
                indices = fresh.nonzero(as_tuple=True)[0]
                encoded = self.perception.encode_depth(prefix.depth_pairs[indices, step].detach())
                token = token.index_copy(0, indices, encoded)
            history = torch.where(active[:, None, None], prefix.histories[:, step], 0).detach()
            height_map, candidate_hidden = self.perception.step(token, history, hidden)
            hidden = torch.where(active[:, None], candidate_hidden, hidden)
            maps.append(height_map)
        maps = torch.stack(maps, 1) if maps else prefix.histories.new_empty(batch, 0, *cfg.map_shape)
        return maps, token, hidden

    @torch.no_grad()
    def rebuild_seed(self, state, history, camera, prefix):
        """前缀严格结束于 state 之前；相机状态由任务提供，不能再次消费当前步。"""
        self._validate_prefix(prefix)
        batch = prefix.histories.shape[0]
        cfg = self.perception.config
        if (history.shape != (batch, cfg.history_length, cfg.proprio_dim)
                or camera.depth_pair.shape != (batch, 2, *cfg.depth_shape)
                or camera.phase.shape != (batch,) or camera.phase.dtype != torch.long):
            raise ValueError("Seed history and camera shapes must match the prefix batch")
        values = (history, camera.depth_pair, camera.phase, state.physical, state.memory, state.command, state.context)
        if any(value.device != prefix.histories.device or value.shape[0] != batch for value in values):
            raise ValueError("Seed tensors must share the prefix batch and device")
        if not all(torch.isfinite(value).all() for value in values) or (camera.phase < 0).any():
            raise ValueError("Seed tensors must be finite and camera phases nonnegative")
        _, token, hidden = self._sequence(prefix)
        memory = VisualMemory(CameraState(camera.depth_pair.detach().clone(), camera.phase.clone()), token, hidden)
        return VisualSeed(state.detached_clone(), history.detach().clone(), memory, self.perception_version)

    def fit_perception(self, batch):
        """一个真实序列批次的辅助更新，支持仅监督后缀但从回合起点恢复记忆。"""
        self._validate_prefix(batch.prefix)
        prefix, cfg = batch.prefix, self.perception.config
        batch_size, steps = prefix.valid.shape
        if (batch.learn.shape != (batch_size, steps) or batch.learn.dtype != torch.bool
                or batch.height_valid.shape != (batch_size, steps, *cfg.map_shape)
                or batch.height_valid.dtype != torch.bool
                or batch.height_metric.shape != batch.height_valid.shape
                or batch.height_normalized.shape != (batch_size, steps, 1, *cfg.map_shape)
                or batch.clean_pairs.shape != prefix.depth_pairs.shape):
            raise ValueError("Perception supervision shapes or mask dtypes are invalid")
        if any(value.device != prefix.histories.device for value in (
                batch.learn, batch.height_valid, batch.height_metric, batch.height_normalized, batch.clean_pairs)):
            raise ValueError("Perception supervision must share the prefix device")
        learn = batch.learn & prefix.valid
        frames = learn & prefix.fresh
        map_valid = batch.height_valid & learn[:, :, None, None]
        zero = prefix.histories.new_zeros(())
        self.perception_optimizer.zero_grad(set_to_none=True)
        if not (frames.any() or map_valid.any()):
            return {"perception_loss": zero, "map_loss": zero, "perception_updated": False}
        maps, _, _ = self._sequence(prefix)
        metric_loss = map_loss(maps.flatten(0, 1), batch.height_metric.flatten(0, 1), map_valid.flatten(0, 1))
        reconstruction = {"loss": zero, "depth": zero, "height": zero, "contrastive": zero}
        if frames.any():
            reconstruction = self.perception.reconstruction_loss(
                prefix.depth_pairs[frames].detach(), batch.clean_pairs[frames].detach(),
                batch.height_normalized[frames].detach())
        loss = reconstruction["loss"] + metric_loss
        loss.backward()
        gradient = clip_gradients(self.perception.parameters(), self.auxiliary.grad_clip)
        finite = bool(torch.isfinite(loss) & torch.isfinite(gradient))
        if finite:
            self.perception_optimizer.step()
            self.perception_version += 1
        return {"perception_loss": loss.detach(), "map_loss": metric_loss.detach(),
                "depth_loss": reconstruction["depth"].detach(), "height_loss": reconstruction["height"].detach(),
                "contrastive_loss": reconstruction["contrastive"].detach(), "perception_grad_norm": gradient.detach(),
                "perception_updated": finite}

    # ---------- 选点监督与策略更新 ----------

    def fit_planner(self, batch):
        self.planner_optimizer.zero_grad(set_to_none=True)
        logits = self.planner(batch.histories.detach(), batch.commands.detach(), batch.height_map.detach())
        loss = foothold_loss(logits, batch.gt_candidates, batch.contacts, batch.valid, batch.candidate_valid)
        if not (batch.valid & batch.candidate_valid.any(-1, keepdim=True)).any():
            return {"planner_loss": loss.detach(), "planner_updated": False}
        loss.backward()
        gradient = clip_gradients(self.planner.parameters(), self.auxiliary.grad_clip)
        finite = bool(torch.isfinite(loss) & torch.isfinite(gradient))
        if finite:
            self.planner_optimizer.step()
            self.planner_updates += 1
        return {"planner_loss": loss.detach(), "planner_grad_norm": gradient.detach(), "planner_updated": finite}

    def improve(self, seed, *, freeze_actor=False, timing=None):
        if not isinstance(seed, VisualSeed):
            raise TypeError("VisualLearner requires a VisualSeed")
        if seed.perception_version != self.perception_version:
            raise ValueError("Perception changed; rebuild the visual seed from its complete episode prefix")
        # 清除辅助更新留下的梯度，便于直接检查策略梯度隔离。
        self.perception_optimizer.zero_grad(set_to_none=True)
        self.planner_optimizer.zero_grad(set_to_none=True)
        return super().improve(seed, freeze_actor=freeze_actor, timing=timing)

    # ---------- 检查点与部署 ----------

    def _metadata(self):
        return {"model_id": MODEL_ID, "spec": asdict(self.task.spec), "config": asdict(self.config),
                "auxiliary": asdict(self.auxiliary), "perception": asdict(self.perception.config),
                "planner": {name: getattr(self.planner, name) for name in ("proprio_dim", "history_length", "command_dim")}}

    def save(self, path):
        """保存完整算法状态；环境、回放和随机数生命周期由任务运行器管理。"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        networks = {name: getattr(self, name).state_dict() for name in ("actor", "model", "critic", "perception", "planner")}
        torch.save({"metadata": self._metadata(), "execution": asdict(self.execution),
                    "networks": networks, "learner": super().state_dict(),
                    "perception_optimizer": self.perception_optimizer.state_dict(),
                    "planner_optimizer": self.planner_optimizer.state_dict(),
                    "perception_version": self.perception_version, "planner_updates": self.planner_updates,
                    "schedule": {name: getattr(self, name) for name in
                                 ("lr_multiplier", "ramp_start", "actor_frozen_until", "policy_start")}}, path)

    def load(self, path):
        device = next(self.model.parameters()).device
        data = torch.load(path, map_location=device, weights_only=True)
        if data["metadata"] != self._metadata():
            raise ValueError("Visual checkpoint configuration or semantic identifier does not match")
        # 在修改任何网络之前一次校验所有参数布局和数值。
        for name in ("actor", "model", "critic", "perception", "planner"):
            current = getattr(self, name).state_dict()
            saved = data["networks"][name]
            if current.keys() != saved.keys() or any(
                    current[key].shape != value.shape or not torch.isfinite(value).all()
                    or (key.endswith(".var") and (value < 0).any())
                    or (key.endswith(".count") and (value <= 0).any()) for key, value in saved.items()):
                raise ValueError(f"Visual checkpoint has invalid {name} network state")
        for name in ("actor", "model", "critic", "perception", "planner"):
            getattr(self, name).load_state_dict(data["networks"][name])
        super().load_state_dict(data["learner"])
        self.nonfinite_steps = data["learner"]["nonfinite_steps"]
        for name in ("perception", "planner"):
            optimizer = getattr(self, name + "_optimizer")
            optimizer.load_state_dict(data[name + "_optimizer"])
            # map_location会搬动Adam计数；普通辅助优化器继续使用原生CPU步数。
            for state in optimizer.state.values():
                state["step"] = state["step"].cpu()
        # 恢复可能换入相同更新次数但不同权重；所有已有视觉种子必须重建。
        self.perception_version = max(self.perception_version, data["perception_version"]) + 1
        self.planner_updates = data["planner_updates"]
        for name in ("lr_multiplier", "ramp_start", "actor_frozen_until", "policy_start"):
            setattr(self, name, data["schedule"][name])

    def deployment_policy(self):
        return DeploymentPolicy(self.perception, self.planner, self.actor)

    def export(self, path):
        """导出独立 TorchScript 模型，调用方持有相机缓存、GRU 记忆及锁存状态。"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.jit.script(self.deployment_policy().cpu()).save(str(path))
