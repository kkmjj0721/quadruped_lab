# Copyright (c) 2026-2027 zh
"""内容：定义同步深度视觉算法的任务、观测、真实转移和回放窗口协议。"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import hashlib
import json
from typing import Protocol

import torch
from torch import Tensor


@dataclass(frozen=True)
class AlgorithmSpec:
    """语义标识应包含坐标系、动作映射、观测顺序及深度处理约定。"""

    state_dim: int
    delta_dim: int
    action_dim: int
    observation_dim: int
    privileged_dim: int
    model_input_dim: int
    semantic_id: str
    depth_semantic_id: str
    context_dim: int = 0
    memory_dim: int = 0
    command_dim: int = 0
    command_indices: tuple[int, ...] = ()
    contact_dim: int = 0
    history_length: int = 5
    depth_shape: tuple[int, int, int] = (1, 64, 64)
    image_height_shape: tuple[int, ...] = ()
    visual_interval: int = 1
    visual_phase_index: int = -1
    model_id: str = "perceptive_supermimic_visual_clock_v4"

    def __post_init__(self):
        for name in ("state_dim", "delta_dim", "action_dim", "observation_dim", "privileged_dim", "model_input_dim", "history_length"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("context_dim", "memory_dim", "command_dim", "contact_dim"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        shape = tuple(self.depth_shape)
        if len(shape) != 3 or any(isinstance(x, bool) or not isinstance(x, int) for x in shape) or shape[0] < 1 or min(shape[1:]) < 8:
            raise ValueError("Depth shape must be (channels, height, width), with channels >= 1 and spatial dimensions >= 8")
        object.__setattr__(self, "depth_shape", shape)
        height_shape = tuple(self.image_height_shape)
        if height_shape and (len(height_shape) != 3 or height_shape[0] != 1 or any(type(x) is not int or x < 1 for x in height_shape)):
            raise ValueError("Image height shape must be empty or (1, height, width)")
        object.__setattr__(self, "image_height_shape", height_shape)
        if type(self.visual_interval) is not int or self.visual_interval < 1:
            raise ValueError("Visual interval must be a positive integer")
        if type(self.visual_phase_index) is not int or (self.visual_interval > 1 and not 0 <= self.visual_phase_index < self.memory_dim):
            raise ValueError("A held visual observation requires a valid phase index in task memory")
        indices = tuple(self.command_indices)
        if (len(indices) != self.command_dim or len(set(indices)) != len(indices)
                or any(isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < self.observation_dim for index in indices)
                or len(indices) >= self.observation_dim):
            raise ValueError("Command indices must uniquely locate every command component inside the observation")
        object.__setattr__(self, "command_indices", indices)
        if not self.semantic_id or not self.depth_semantic_id or self.model_id != "perceptive_supermimic_visual_clock_v4":
            raise ValueError("Missing task/depth semantics or unsupported algorithm format")

    @property
    def hash(self) -> str:
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()


@dataclass
class TaskState:
    """所有浮点状态位于同一设备；末轴分别对应声明的物理量和内部状态。"""

    physical: Tensor
    memory: Tensor
    command: Tensor
    context: Tensor

    def map(self, function) -> TaskState:
        return TaskState(**{field.name: function(getattr(self, field.name)) for field in fields(self)})

    def index(self, index) -> TaskState:
        return self.map(lambda value: value[index])

    def detached_clone(self) -> TaskState:
        return self.map(lambda value: value.detach().clone())


@dataclass
class VisualObservation:
    """frame_id是控制步；相机版本仅采集时增加，高度标签随图像保持。"""

    proprio: Tensor
    depth: Tensor
    depth_clean: Tensor
    episode_id: Tensor
    frame_id: Tensor
    camera_frame_id: Tensor | None = None
    image_height: Tensor | None = None

    def index(self, index) -> VisualObservation:
        return VisualObservation(**{field.name: None if getattr(self, field.name) is None else getattr(self, field.name)[index]
                                    for field in fields(self)})

    def detached_clone(self) -> VisualObservation:
        return VisualObservation(**{field.name: None if getattr(self, field.name) is None else getattr(self, field.name).detach().clone()
                                    for field in fields(self)})


@dataclass
class Decision:
    """上下文版本标记随机化或外生动力学改变，不能在回放中串联不同版本。"""

    state: TaskState
    observation: VisualObservation
    context_version: Tensor

    def index(self, index) -> Decision:
        return Decision(self.state.index(index), self.observation.index(index), self.context_version[index])

    def detached_clone(self) -> Decision:
        return Decision(self.state.detached_clone(), self.observation.detached_clone(), self.context_version.detach().clone())


@dataclass
class EnvStep:
    """final为重置前最终测量，next为下一次决策；contacts标记final的接触。"""

    final: Decision
    next: Decision
    reward: Tensor
    terminated: Tensor
    truncated: Tensor
    contacts: Tensor


@dataclass
class ModelPrediction:
    state: TaskState
    latent: Tensor
    contact_logits: Tensor


@dataclass
class VisualSeed:
    state: TaskState
    history: Tensor
    depth: Tensor

    def index(self, index) -> VisualSeed:
        return VisualSeed(self.state.index(index), self.history[index], self.depth[index])


@dataclass
class VisualWindow:
    """批次优先；终止当步可监督final，随后步无效；depth是窗口首帧。"""

    states: TaskState
    actions: Tensor
    targets: TaskState
    contacts: Tensor
    valid: Tensor
    depth: Tensor
    depth_next: Tensor
    depth_next_clean: Tensor
    image_keys: Tensor | None = None
    depth_next_height: Tensor | None = None

    @property
    def initial(self) -> TaskState:
        return self.states.index((slice(None), 0))

    def index(self, index) -> VisualWindow:
        return VisualWindow(self.states.index(index), self.actions[index], self.targets.index(index),
                            self.contacts[index], self.valid[index], self.depth[index],
                            self.depth_next[index], self.depth_next_clean[index],
                            None if self.image_keys is None else self.image_keys[index],
                            None if self.depth_next_height is None else self.depth_next_height[index])


class Task(Protocol):
    """任务负责物理积分、几何查询与奖励；训练算法不依赖具体机器人。"""

    spec: AlgorithmSpec
    dt: float

    def model_input(self, state: TaskState, action: Tensor) -> Tensor: ...
    def model_target(self, state: TaskState, target: TaskState) -> Tensor: ...
    def model_error(self, recorded: TaskState, predicted: TaskState, target: TaskState) -> Tensor: ...
    def integrate(self, state: TaskState, action: Tensor, delta: Tensor) -> TaskState: ...
    def observe(self, state: TaskState, *, noisy: bool) -> Tensor: ...
    def body_linear_velocity(self, state: TaskState) -> Tensor:
        """返回同一时刻的机身系xyz线速度[B,3]，float32、m/s且与状态同设备。"""
        ...
    def terrain_height(self, state: TaskState) -> Tensor:
        """仅启用高度监督时需要；返回同终态、同设备float32的[B,1,H,W]，归一化至[-1,1]。"""
        ...
    def privileged(self, state: TaskState, latent: Tensor) -> Tensor: ...
    def soft_termination(self, prediction: ModelPrediction) -> Tensor: ...
    def reward(self, state: TaskState, action: Tensor, prediction: ModelPrediction, failure: Tensor) -> Tensor: ...


class Environment(Protocol):
    task: Task
    num_envs: int
    device: torch.device | str

    def reset(self) -> Decision: ...
    def step(self, action: Tensor) -> EnvStep: ...


def append_history(history: Tensor, frame: Tensor) -> Tensor:
    """使用非原地拼接，在想象中保留本体历史的跨步导数。"""
    return torch.cat((history[:, 1:], frame.unsqueeze(1)), dim=1)


def validate_decision(value: Decision, spec: AlgorithmSpec, batch: int, device: torch.device) -> None:
    """在环境输出边界统一校验张量协议，内部学习链路沿用该约定。"""
    entries = [(name, getattr(value.state, name), (batch, width), torch.float32)
               for name, width in (("physical", spec.state_dim), ("memory", spec.memory_dim),
                                   ("command", spec.command_dim), ("context", spec.context_dim))]
    obs = value.observation
    if spec.visual_interval > 1 and obs.camera_frame_id is None:
        raise ValueError("Held images require camera frame IDs")
    if spec.image_height_shape and obs.image_height is None:
        raise ValueError("Image acquisition height labels are required by the task specification")
    if obs.camera_frame_id is not None:
        entries.append(("camera_frame_id", obs.camera_frame_id, (batch,), torch.int64))
    if obs.image_height is not None:
        entries.append(("image_height", obs.image_height, (batch, *spec.image_height_shape), torch.float32))
    entries.extend((("proprio", obs.proprio, (batch, spec.observation_dim), torch.float32),
                    ("depth", obs.depth, (batch, *spec.depth_shape), torch.float32),
                    ("depth_clean", obs.depth_clean, (batch, *spec.depth_shape), torch.float32),
                    ("episode_id", obs.episode_id, (batch,), torch.int64),
                    ("frame_id", obs.frame_id, (batch,), torch.int64),
                    ("context_version", value.context_version, (batch,), torch.int64)))
    for name, tensor, shape, dtype in entries:
        if tensor.shape != shape or tensor.dtype != dtype or tensor.device != device:
            raise ValueError(f"Invalid {name}: expected shape {shape}, dtype {dtype}, device {device}")
        if tensor.is_floating_point() and not torch.isfinite(tensor).all():
            raise FloatingPointError(f"Environment returned non-finite {name}")
    if ((obs.depth < 0) | (obs.depth > 1) | (obs.depth_clean < 0) | (obs.depth_clean > 1)).any():
        raise ValueError("Processed depth and clean targets must be within [0,1]")

    if obs.image_height is not None and ((obs.image_height < -1) | (obs.image_height > 1)).any():
        raise ValueError("Image height labels must be within [-1,1]")
    if spec.visual_interval > 1:
        phase = value.state.memory[:, spec.visual_phase_index]
        if ((phase < 0) | (phase >= spec.visual_interval) | (phase != phase.round())).any():
            raise ValueError("Visual phase must be an integer within the acquisition interval")
