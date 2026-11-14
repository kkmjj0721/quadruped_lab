# Copyright (c) 2026-2027 zh
"""内容：定义固定 MGDP 感知结构和本项目时序地图头的输入尺寸。"""

from dataclasses import dataclass
from typing import ClassVar


@dataclass(frozen=True)
class PerceptionConfig:
    """本体尺寸由任务声明，其余结构沿用所选感知基准与地图设计。"""

    proprio_dim: int
    history_length: int = 5
    depth_shape: ClassVar[tuple[int, int]] = (16, 16)
    map_shape: ClassVar[tuple[int, int]] = (17, 11)
    token_dim: ClassVar[int] = 64
    hidden_dim: ClassVar[int] = 128

    def __post_init__(self):
        for name in ("proprio_dim", "history_length"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
