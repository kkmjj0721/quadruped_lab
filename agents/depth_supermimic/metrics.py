# Copyright (c) 2026-2027 zh
"""内容：在设备上累计有限指标，并在日志边界集中读取标量与有效掩码。"""

from functools import lru_cache
import math

import torch

from .compilation import compile_kernel


@lru_cache(maxsize=4)
def _compiled_accumulator(function, backend):
    """复用两个无状态累计核心的编译包装，逐轮构造累计器不重复配置Dynamo。"""
    return compile_kernel(function, backend)


class EpisodeWindow:
    """在设备上保留最近100个完成回合，同步结束时按环境编号排列。"""

    def __init__(self, num_envs, device):
        self.records = torch.zeros(100, 2, dtype=torch.float64, device=device)
        self.valid = torch.zeros(100, dtype=torch.bool, device=device)
        self.order = torch.arange(100 + num_envs, device=device)

    def clear(self):
        self.valid.zero_()

    @torch.no_grad()
    def append(self, returns, lengths, done):
        # 长度先转FP32，保持原cpu().tolist()之前的转换口径。
        candidates = torch.stack((returns.double(), lengths.float().double()), dim=-1)
        records = torch.cat((self.records, candidates))
        valid = torch.cat((self.valid, done))
        keys = torch.where(valid, self.order, -1)
        selected = keys.topk(100, sorted=True).indices.flip(0)
        self.records.copy_(records[selected])
        self.valid.copy_(valid[selected])

    def metrics(self, dt):
        count = self.valid.sum()
        denominator = count.clamp_min(1)
        returns = self.records[:, 0]
        mean = torch.where(self.valid, returns, 0).sum() / denominator
        centered = torch.where(self.valid, returns - mean, 0)
        lengths = torch.where(self.valid, self.records[:, 1], 0).sum() / denominator
        metrics = {
            "episode/return_mean": mean,
            "episode/return_min": torch.where(self.valid, returns, torch.inf).amin(),
            "episode/return_max": torch.where(self.valid, returns, -torch.inf).amax(),
            "episode/return_std": (centered.square().sum() / denominator).sqrt(),
            "episode/length_steps_mean": lengths,
            "episode/length_seconds_mean": lengths * dt,
            "episode/window_count": count,
        }
        populated = count > 0
        return metrics, {name: populated for name in metrics}


def _accumulate_model_metrics(values, totals, validity):
    """固定指标元组内融合逐更新过滤与累计，沿用每个输入的数值类型。"""
    updated, valid = [], []
    for value, total, accepted in zip(values, totals, validity):
        finite = torch.isfinite(value).all()
        updated.append(total + torch.where(finite, value, 0))
        valid.append(accepted | finite)
    return tuple(updated), tuple(valid)


class ModelMetrics:
    """保留逐更新过滤、总更新次数分母及全无效缺省，可编译设备累计核心。"""

    def __init__(self, *, compile_backend=None):
        self.totals = {}
        self.valid = {}
        self._accumulate = (_accumulate_model_metrics if compile_backend is None
                            else _compiled_accumulator(_accumulate_model_metrics, compile_backend))

    @torch.no_grad()
    def add(self, metrics):
        device_values = {}
        for name, value in metrics.items():
            if isinstance(value, torch.Tensor):
                device_values[name] = value.detach()
                if name not in self.totals:
                    # 与原来的0+where(finite,value,0)保持推广一致，包括bool到int64。
                    self.totals[name] = torch.zeros_like(value, dtype=torch.result_type(value, 0))
                    self.valid[name] = torch.zeros((), dtype=torch.bool, device=value.device)
            else:
                finite = math.isfinite(value)
                contribution = value if finite else 0
                self.totals[name] = self.totals.get(name, 0) + contribution
                self.valid[name] = self.valid.get(name, False) | finite
        if device_values:
            names = tuple(device_values)
            totals, valid = self._accumulate(tuple(device_values.values()), tuple(self.totals[name] for name in names),
                                             tuple(self.valid[name] for name in names))
            self.totals.update(zip(names, totals))
            self.valid.update(zip(names, valid))

    def mean(self, updates):
        return {name: value / updates for name, value in self.totals.items()}, self.valid


def _metric_reduction(name):
    """区分窗口增量、末轮快照和按真实控制步加权的采集均值。"""
    if name in {"failures", "timeouts", "reset_steps", "episode/completed", "real_steps_delta",
                "collection/control_steps"} or name.endswith("_updates_delta"):
        return "sum"
    if (name in {"iteration", "training_target_iteration", "real_steps", "nonfinite_steps",
                 "control_enabled", "model_not_ready"}
            or name.endswith(("_total", "_learning_rate"))
            or name.startswith(("replay/", "episode/", "control/", "guard/"))):
        return "last"
    if name == "real_reward" or name.startswith(("task/", "Reward_Rate/")):
        return "weighted"
    return "mean"


def _accumulate_metric_columns(values, masks, totals, counts, bad, last, weighted, optional, steps):
    """固定列布局内融合打包、有效性和累计；FP64输出避免整数先经FP32舍入。"""
    # 转换必须先于混合类型stack；只指定FP64输出时编译器可能先按FP32提升输入。
    values = torch.stack(tuple(value.to(torch.float64) for value in values))
    valid = torch.stack(masks)
    finite = torch.isfinite(values)
    valid = valid & (finite | ~optional)
    bad.logical_or_(valid & ~finite & ~optional)
    weights = torch.where(weighted, steps, 1)
    accepted = torch.where(valid, values, 0)
    totals.copy_(torch.where(last, accepted, totals + accepted * weights))
    counts.copy_(torch.where(last, valid, counts + valid * weights))


class MetricWindow:
    """设备标量按列累计，主机已有标量留在主机；仅汇总边界由调用方回读。

    可选编译固定布局的打包与累计核心，主机字典和跨轮调度不进入图。
    最近100回合等快照取末轮；缺省和可选非有限值不进入均值分母。
    同名指标在窗口中固定使用主机数值或同一设备张量，沿用训练循环的来源约定。
    """

    def __init__(self, device, *, compile_backend=None):
        self.device = torch.device(device)
        self._names = []
        self._zeros = {}
        self._mask_defaults = {}
        self._optional = set()
        self._accumulate = (_accumulate_metric_columns if compile_backend is None
                            else _compiled_accumulator(_accumulate_metric_columns, compile_backend))
        self._totals = torch.empty(0, dtype=torch.float64, device=self.device)
        self._counts = torch.empty_like(self._totals)
        self._bad = torch.empty(0, dtype=torch.bool, device=self.device)
        self._steps = torch.ones(1, dtype=torch.float64, device=self.device)
        self._true = torch.ones((), dtype=torch.bool, device=self.device)
        self._false = torch.zeros((), dtype=torch.bool, device=self.device)
        self.clear()

    def __len__(self):
        return self.iterations

    def clear(self):
        self.iterations = 0
        self.start_iteration = self.end_iteration = None
        self._host = {}
        self._totals.zero_()
        self._counts.zero_()
        self._bad.zero_()

    @torch.no_grad()
    def add(self, metrics, *, optional=(), validity=None):
        validity = validity if validity is not None else {}
        optional = set(optional)
        self.iterations += 1
        iteration = metrics.get("iteration", self.iterations)
        if isinstance(iteration, torch.Tensor):
            iteration = iteration.detach().clone()
        if self.start_iteration is None:
            self.start_iteration = iteration
        self.end_iteration = iteration
        steps = metrics.get("collection/control_steps", 1)
        device_values = {}
        for name in list(self._host):
            if name not in metrics and _metric_reduction(name) == "last" and math.isfinite(self._host[name][0]):
                del self._host[name]
        for name, value in metrics.items():
            if isinstance(value, torch.Tensor):
                device_values[name] = value.detach().reshape(())
                continue
            valid = validity.get(name, True)
            mode = _metric_reduction(name)
            if isinstance(valid, torch.Tensor) or mode == "weighted" and isinstance(steps, torch.Tensor):
                device_values[name] = torch.as_tensor(value, device=self.device).detach().reshape(())
                continue
            if name in optional:
                valid = valid and math.isfinite(value)
            weight = steps if mode == "weighted" else 1
            if not valid:
                if mode == "last" and math.isfinite(self._host.get(name, (0, 0))[0]):
                    self._host.pop(name, None)
                continue
            previous, denominator = self._host.get(name, (0, 0))
            if mode == "last":
                # 必须记录的非有限快照不能被后续合法值掩盖。
                self._host[name] = (value if math.isfinite(previous) else previous, 1)
            else:
                self._host[name] = (previous + value * weight, denominator + weight)

        new_names = [name for name in device_values if name not in self._zeros]
        new_optional = self._optional | optional
        if new_names or new_optional != self._optional:
            self._names.extend(new_names)
            self._zeros.update((name, torch.zeros_like(device_values[name])) for name in new_names)
            # 每列视图仅建立一次；对象独立以保持编译输入别名约定，底层标志只读共享。
            self._mask_defaults.update((name, (self._false.view(()), self._true.view(()))) for name in new_names)
            self._optional = new_optional
            padding = self._totals.new_zeros(len(new_names))
            self._totals = torch.cat((self._totals, padding))
            self._counts = torch.cat((self._counts, padding))
            self._bad = torch.cat((self._bad, padding.bool()))
            modes = [_metric_reduction(name) for name in self._names]
            self._last = torch.tensor([mode == "last" for mode in modes], device=self.device)
            self._mean = torch.tensor([mode in {"mean", "weighted"} for mode in modes], device=self.device)
            self._weighted = torch.tensor([mode == "weighted" for mode in modes], device=self.device)
            self._filter = torch.tensor([name in self._optional for name in self._names], device=self.device)
        if not self._names:
            return
        values, masks = [], []
        for name in self._names:
            # 缺省值沿用该列的形状和dtype，缺省模式变化不会触发布局重编译。
            values.append(device_values.get(name, self._zeros[name]))
            valid = validity.get(name, True) if name in device_values else False
            masks.append(valid.detach().reshape(()) if isinstance(valid, torch.Tensor)
                         else self._mask_defaults[name][bool(valid)])
        # 2/3步交替只更新设备值，避免Python步数参与编译值特化。
        if isinstance(steps, torch.Tensor):
            self._steps.copy_(steps)
        else:
            self._steps.fill_(steps)
        self._accumulate(tuple(values), tuple(masks), self._totals, self._counts, self._bad,
                         self._last, self._weighted, self._filter, self._steps)

    def aggregate(self):
        """返回汇总值及有效掩码，不同步设备；空窗口返回两个空映射。"""
        if not self.iterations:
            return {}, {}
        metrics = {name: total / count if _metric_reduction(name) in {"mean", "weighted"} else total
                   for name, (total, count) in self._host.items()}
        validity = {}
        if self._names:
            means = self._totals / self._counts.clamp_min(1)
            values = torch.where(self._mean, means, self._totals)
            values = torch.where(self._bad, torch.nan, values)
            valid = (self._counts > 0) | self._bad
            metrics.update(zip(self._names, values.unbind()))
            validity.update(zip(self._names, valid.unbind()))
        metrics.update({"metrics/window_start_iteration": self.start_iteration,
                        "metrics/window_end_iteration": self.end_iteration,
                        "metrics/window_iterations": self.iterations})
        return metrics, validity


def read_metrics(metrics, device, *, optional=(), validity=None):
    """一次回传张量；仅过滤学习器原本可省略的键，保留其他指标的异常。"""
    validity = validity if validity is not None else {}
    tensor_names = [name for name, value in metrics.items() if isinstance(value, torch.Tensor)]
    mask_names = list(validity)
    packed = [metrics[name].detach().to(device=device, dtype=torch.float64).reshape(()) for name in tensor_names]
    packed.extend(torch.as_tensor(validity[name], device=device, dtype=torch.float64).reshape(()) for name in mask_names)
    values = torch.stack(packed).cpu().tolist() if packed else []
    result = {name: float(value) for name, value in metrics.items() if name not in tensor_names}
    result.update(zip(tensor_names, values[:len(tensor_names)]))
    for name, valid in zip(mask_names, values[len(tensor_names):]):
        if not valid:
            result.pop(name, None)
    for name in optional:
        if name in result and not math.isfinite(result[name]):
            del result[name]
    return result
