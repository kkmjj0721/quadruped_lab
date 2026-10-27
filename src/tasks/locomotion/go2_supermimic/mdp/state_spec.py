# Copyright (c) 2026-2027 zh
"""内容：定义Go2关节顺序、任务记忆和真实上下文槽位；物理状态见reference_state。"""

JOINT_NAMES = tuple(f"{leg}_{joint}_joint" for leg in ("FL", "FR", "RL", "RR") for joint in ("hip", "thigh", "calf"))
FOOT_NAMES = ("FL", "FR", "RL", "RR")
ACTION_DIM = len(JOINT_NAMES)
PREVIOUS_ACTION = slice(0, 12)
ELAPSED_TIME = 12
COMMAND_TIMER = 13
CONTEXT_FRICTION = 0
CONTEXT_ENCODER_BIAS = slice(1, 13)
CONTEXT_ORIGIN = slice(16, 19)
CONTEXT_RANGES = slice(19, 25)
CONTEXT_ZERO_PROBABILITY = 25
CONTEXT_DIM = 26
