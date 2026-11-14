# Copyright (c) 2026-2027 zh
"""内容：声明四足注意力选点模块的本体历史和指令维度。"""

from dataclasses import dataclass


@dataclass(frozen=True)
class FootholdConfig:
    """观测单位和布局由调用任务规定；网格及网络宽度遵循已定方案。"""

    proprio_dim: int
    history_length: int = 5
    command_dim: int = 3

    def __post_init__(self):
        for name in ("proprio_dim", "history_length", "command_dim"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
