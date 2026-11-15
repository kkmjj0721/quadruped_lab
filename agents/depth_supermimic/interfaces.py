# Copyright (c) 2026-2027 zh
"""
内容：
    约定双帧视觉算法的尺寸、真实终止前转移、GRU 起点和多步监督窗口。
    TaskState 仅作为张量载体复用；本算法不消费 Go2 策略或动力学实现。
    任务必须提供 semantic_configuration() 纯 Python 字典；检查点据此核对任务定义，
    其中包含奖励、控制、相机、地形等语义设置，不包含种子、环境数量和执行预算。
    实际环境数量由检查点独立保存和约束，恢复逐环境课程时要求数量不变。
    failure_features(predicted, recorded_target, command) 只用于失败头监督，
    两路共享记录的下一状态接触计时和实际执行指令，其余预测状态条件由任务保留。
    standing_mask(state) 返回任务站立判据，仅用于真实采集的额外动作噪声屏蔽。
"""

from dataclasses import dataclass, fields

from torch import Tensor

from agents.supermimic.interfaces import TaskState


@dataclass(frozen=True)
class DepthSpec:
    """任务负责坐标和物理布局，算法只读取明确的尺寸与语义标识。"""

    state_dim: int = 129
    delta_dim: int = 36
    auxiliary_dim: int = 40
    contact_dim: int = 4
    action_dim: int = 12
    observation_dim: int = 45
    privileged_dim: int = 0
    context_dim: int = 26
    memory_dim: int = 42
    command_dim: int = 3
    model_context_dim: int = 28
    scan_dim: int = 187
    history_length: int = 5
    latent_dim: int = 64
    depth_shape: tuple[int, int, int] = (2, 32, 32)
    observation_term_dims: tuple[int, ...] = (3, 3, 3, 12, 12, 12)
    semantic_id: str = "go1_depth_mgdp_gru_v3"

    @property
    def model_input_dim(self):
        return self.delta_dim + self.action_dim + self.model_context_dim + self.scan_dim


@dataclass
class DepthStep:
    """final_* 必须来自 reset 前；next_* 是下一次真实决策的输入。"""

    final_state: TaskState
    next_state: TaskState
    reward: Tensor
    terminated: Tensor
    truncated: Tensor
    final_depth: Tensor
    next_depth: Tensor
    executed_action: Tensor
    final_clean_depth: Tensor
    next_observation: Tensor


@dataclass
class DepthSeed:
    state: TaskState
    history: Tensor
    depth: Tensor
    time_left: Tensor
    env: Tensor
    hidden: Tensor

    def index(self, index):
        return DepthSeed(self.state.index(index), self.history[index], self.depth[index], self.time_left[index], self.env[index], self.hidden[index])


@dataclass
class DepthWindow:
    """批次、时间轴顺序固定；valid 包含终止当步，屏蔽其后的新回合。"""

    states: TaskState
    targets: TaskState
    actions: Tensor
    contact: Tensor
    depth: Tensor
    next_depth: Tensor
    valid: Tensor
    terminated: Tensor
    weight: Tensor
    env: Tensor
    next_clean_depth: Tensor

    def index(self, index):
        return DepthWindow(**{field.name: getattr(self, field.name).index(index) if field.name in ("states", "targets")
                              else getattr(self, field.name)[index] for field in fields(self)})


def flatten_state(state: TaskState) -> TaskState:
    """只合并批次与时间轴，沿用任务自身的每个字段宽度。"""
    return state.map(lambda value: value.flatten(0, 1))
