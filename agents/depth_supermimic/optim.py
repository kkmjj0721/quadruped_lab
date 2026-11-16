# Copyright (c) 2026-2027 zh
"""内容：按 PartyMimic 的 Linear 分组组合 Muon 与 Adam，支持设备门控和更新图恢复。

来源：PartyMimic@69c53621 的 src/partymimic/optim.py，许可与适配边界见 NOTICE.md。
"""

from __future__ import annotations

import torch
from torch import nn
from torch.optim._muon import _adjust_lr, _zeropower_via_newtonschulz


class MuonAdam:
    """指定 MLP 的权重矩阵使用 Muon，其余参数使用 Adam。

    io=False 时仅选择每个 MLP 中间的 Linear.weight；io=True 时包含首尾层。
    CUDA Graph 使用来源的设备学习率更新，普通执行使用原生 Muon。
    """

    def __init__(self, model: nn.Module, mlps: list[nn.Module], lr: float, *, betas=(0.9, 0.95),
                 fused: bool | None = True, capturable: bool = False, io: bool = False, weight_decay: float = 0.1):
        selected = set()
        for mlp in mlps:
            linears = [layer for layer in mlp.modules() if isinstance(layer, nn.Linear)]
            selected.update(id(layer.weight) for layer in (linears if io else linears[1:-1]))
        muon = [parameter for parameter in model.parameters() if id(parameter) in selected]
        adam = [parameter for parameter in model.parameters() if id(parameter) not in selected and parameter.requires_grad]
        if not muon or not adam:
            raise ValueError("MuonAdam requires both Muon matrices and Adam parameters")
        self.capturable = capturable
        self.device = muon[0].device
        if capturable and (self.device.type != "cuda" or not fused):
            raise ValueError("Capturable MuonAdam requires CUDA and fused Adam")
        muon_rate = torch.tensor(float(lr), device=self.device) if capturable else lr
        adam_rate = torch.tensor(float(lr), device=self.device) if capturable else lr
        self.muon = torch.optim.Muon(muon, lr=muon_rate, weight_decay=weight_decay, adjust_lr_fn="match_rms_adamw")
        self.adam = torch.optim.Adam(adam, lr=adam_rate, betas=betas, fused=fused, capturable=capturable)

    @property
    def param_groups(self):
        """每次取得实际子组，避免加载状态后仍引用已经替换的字典。"""
        return self.muon.param_groups + self.adam.param_groups

    def zero_grad(self, set_to_none: bool = True):
        self.muon.zero_grad(set_to_none=set_to_none)
        self.adam.zero_grad(set_to_none=set_to_none)

    @torch.no_grad()
    def _muon_step_tensor_lr(self, ok: torch.Tensor | None = None):
        """保持来源的浮点算术顺序，用设备标量更新学习率与有限性门控。"""
        for group in self.muon.param_groups:
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                state = self.muon.state[parameter]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(parameter.grad, memory_format=torch.preserve_format)
                momentum = state["momentum_buffer"]
                if ok is None:
                    momentum.lerp_(parameter.grad, 1 - group["momentum"])
                else:
                    momentum.copy_(torch.where(ok, momentum.lerp(parameter.grad, 1 - group["momentum"]), momentum))
                update = parameter.grad.lerp(momentum, group["momentum"]) if group["nesterov"] else momentum
                update = _zeropower_via_newtonschulz(update, group["ns_coefficients"], group["ns_steps"], group["eps"])
                ratio = _adjust_lr(1.0, group["adjust_lr_fn"], parameter.shape)
                # 先转参数精度再乘设备 lr，避免 BF16 更新量在缩放时再次舍入。
                delta = update.to(parameter.dtype) * (group["lr"] * -ratio)
                if group["weight_decay"] != 0.0:
                    shrink = 1 - group["lr"] * group["weight_decay"]
                    parameter.mul_(shrink if ok is None else torch.where(ok, shrink, 1.0))
                parameter.add_(delta if ok is None else torch.where(ok, delta, 0.0))

    @torch.no_grad()
    def step(self, ok: torch.Tensor | None = None):
        """不接受的更新同时跳过参数、Muon 动量、Adam 动量和计数。"""
        if ok is not None:
            for group in self.param_groups:
                for parameter in group["params"]:
                    if parameter.grad is not None:
                        parameter.grad.copy_(torch.where(ok, parameter.grad, 0.0))
            if not self.capturable and (self.device.type == "cpu" or not self.adam.defaults["fused"]):
                if bool(ok):
                    self.muon.step()
                    self.adam.step()
                return
            if self.capturable:
                self._muon_step_tensor_lr(ok)
            else:
                # 普通 CUDA 执行保留原生 Muon 算术，未接受时回写参数和动量。
                parameters = [parameter for group in self.muon.param_groups for parameter in group["params"]
                              if parameter.grad is not None]
                momenta = []
                for parameter in parameters:
                    state = self.muon.state[parameter]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(parameter.grad, memory_format=torch.preserve_format)
                    momenta.append(state["momentum_buffer"])
                tensors = parameters + momenta
                previous = [tensor.clone() for tensor in tensors]
                self.muon.step()
                for tensor, old in zip(tensors, previous, strict=True):
                    tensor.copy_(torch.where(ok, tensor, old))
            self.adam.found_inf = (~ok).float()
            try:
                self.adam.step()
            finally:
                self.adam.found_inf = None
            return
        if self.capturable:
            self._muon_step_tensor_lr()
        else:
            self.muon.step()
        self.adam.step()

    def state_dict(self):
        return {"muon": self.muon.state_dict(), "adam": self.adam.state_dict()}

    def load_state_dict(self, data):
        """保留当前执行方式和设备 lr 地址，恢复两套优化器的实际学习状态。"""
        if not isinstance(data, dict) or set(data) != {"muon", "adam"}:
            raise ValueError("MuonAdam requires a Muon+Adam checkpoint; optimizer types cannot be converted")
        references = [group["lr"] for group in self.param_groups]
        muon_state = {**data["muon"], "param_groups": [{**group, "lr": float(group["lr"])}
                                                     for group in data["muon"]["param_groups"]]}
        # 加载前覆盖执行字段，使 PyTorch 为 fused／捕获模式将 Adam.step 放到参数设备。
        adam_state = {**data["adam"], "param_groups": [
            {**group, "lr": float(group["lr"]), "fused": self.adam.defaults["fused"], "capturable": self.capturable}
            for group in data["adam"]["param_groups"]]}
        self.muon.load_state_dict(muon_state)
        self.adam.load_state_dict(adam_state)
        if not self.capturable and not self.adam.defaults["fused"]:
            # PyTorch 在此模式下保留传入 step 的设备，需恢复为原生初始化使用的 CPU。
            for state in self.adam.state.values():
                state["step"] = state["step"].cpu()
        if self.capturable:
            for group, reference in zip(self.param_groups, references, strict=True):
                reference.fill_(group["lr"])
                group["lr"] = reference
