# Copyright (c) 2026-2027 zh
"""内容：定义视觉算法的显式相机记忆、回合前缀、监督批次与任务边界。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from torch import Tensor

from ..buffer import Seed
from ..interfaces import TaskState


MODEL_ID = "supermimic_visual_foothold_v2"


@dataclass
class CameraState:
    """当前控制步刷新前的双帧与相位；图像已经过任务侧噪声及归一化。"""

    depth_pair: Tensor
    phase: Tensor

    def index(self, index):
        return CameraState(self.depth_pair[index], self.phase[index])


@dataclass
class VisualMemory:
    """感知权重对应的缓存；所有张量首轴都是批次，没有模块内部隐状态。"""

    camera: CameraState
    token: Tensor
    hidden: Tensor

    def index(self, index):
        return VisualMemory(self.camera.index(index), self.token[index], self.hidden[index])


@dataclass
class VisualSeed(Seed):
    """当前控制步感知更新前的种子；完整前缀严格排除当前控制步。"""

    visual: VisualMemory
    perception_version: int

    def index(self, index):
        return VisualSeed(self.state.index(index), self.history[index], self.visual.index(index), self.perception_version)


@dataclass
class PerceptionPrefix:
    """同回合完整前缀，时间轴为第二轴；valid 仅允许连续有效段后接填充。

    histories 为 [B,T,H,D]，depth_pairs 为 [B,T,2,16,16]。
    fresh 为新相机帧标志；首个有效步必须为新帧。零长度表示回合初始。
    任务负责保证前缀没有跨回合；这里不根据位姿跳变猜测 reset。
    """

    histories: Tensor
    depth_pairs: Tensor
    fresh: Tensor
    valid: Tensor


@dataclass
class PerceptionBatch:
    """完整前缀和监督标签；learn 标记参与损失的时间步，可排除预热前缀。"""

    prefix: PerceptionPrefix
    clean_pairs: Tensor
    height_normalized: Tensor
    height_metric: Tensor
    height_valid: Tensor
    learn: Tensor


@dataclass
class FootholdBatch:
    """调用方提供同一时刻的重建地图与 GT 标签，不在算法内生成接触事件。"""

    histories: Tensor
    commands: Tensor
    height_map: Tensor
    gt_candidates: Tensor
    contacts: Tensor
    valid: Tensor
    candidate_valid: Tensor


class VisualTask(Protocol):
    """在已有 SuperMimic 任务协议上补充的三个方法，均不保存隐式批次状态。"""

    def update_camera(self, state: TaskState, camera: CameraState) -> tuple[CameraState, Tensor]:
        """按预测位姿和相位渲染，返回新相机状态及 bool[B] 刷新标志。"""
        ...

    def update_visual_targets(self, state: TaskState, candidates: Tensor) -> tuple[TaskState, Tensor, Tensor]:
        """锁存候选，返回状态、yaw 系目标[B,4,3]和有效位[B,4]。

        任务不得 detach 原 physical；坐标转换保留对当前位姿的导数。
        触地当步的旧目标已经由上一个转移的 reward 计分，再在此释放。
        """
        ...

    def visual_privileged(self, state: TaskState, memory: VisualMemory) -> Tensor:
        """critic 输入包含同一时刻的物理、感知、相机及锁存状态。"""
        ...
