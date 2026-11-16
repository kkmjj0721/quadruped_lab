# Copyright (c) 2026-2027 zh
"""内容：定义低频深度样本、完整视觉前缀和可微循环训练的批次协议。"""

from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor

from .frame_storage import HostDepthFrames
from .tensor_state import TaskState


@dataclass
class DepthObservation:
    """当前决策时刻；图像按相机帧缓存，phase 为距上次新帧的控制步数。"""

    clean: Tensor
    noisy: Tensor
    valid: Tensor
    new_frame: Tensor
    phase: Tensor
    episode_id: Tensor
    frame_index: Tensor
    control_step: Tensor


@dataclass
class DepthSeed:
    """前缀由旧到新且包含起点最近新帧；末个有效前缀保留起点感知梯度。"""

    state: TaskState
    history: Tensor
    phase: Tensor
    images: Tensor | HostDepthFrames
    image_valid: Tensor
    clean: Tensor
    pixel_valid: Tensor

    def index(self, index):
        images = self.images.index_rows(index) if isinstance(self.images, HostDepthFrames) else self.images[index]
        return DepthSeed(self.state.index(index), self.history[index], self.phase[index], images,
                         self.image_valid[index], self.clean[index], self.pixel_valid[index])


@dataclass
class DepthWindow:
    """只含同回合的自回归窗口；结束步及其后位置均无物理或视觉监督。"""

    seed: DepthSeed
    actions: Tensor
    targets: TaskState
    valid: Tensor
    terminated: Tensor
    next_images: Tensor | HostDepthFrames
    new_frame: Tensor

    @property
    def initial(self):
        return self.seed.state

    def index(self, index):
        images = (self.next_images.index_rows(index) if isinstance(self.next_images, HostDepthFrames)
                  else self.next_images[index])
        return DepthWindow(self.seed.index(index), self.actions[index], self.targets.index(index), self.valid[index],
                           self.terminated[index], images, self.new_frame[index])


@dataclass
class PreparedDepthSeed:
    """图外重建完整旧前缀；图内从最新图像保留感知梯度，形状不随前缀变化。"""

    state: TaskState
    history: Tensor
    phase: Tensor
    prefix_hidden: Tensor
    latest_image: Tensor
    clean: Tensor
    pixel_valid: Tensor

    def index(self, index):
        return PreparedDepthSeed(self.state.index(index), self.history[index], self.phase[index],
                                 self.prefix_hidden[index], self.latest_image[index], self.clean[index],
                                 self.pixel_valid[index])


@dataclass
class PreparedDepthWindow:
    """模型图仅接收固定物理窗口、当前重建记忆及停止梯度的真实视觉标签。"""

    initial: TaskState
    actions: Tensor
    targets: TaskState
    valid: Tensor
    new_frame: Tensor
    phase: Tensor
    hidden: Tensor
    latent: Tensor
    labels: Tensor
