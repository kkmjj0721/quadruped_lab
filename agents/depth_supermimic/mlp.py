# Copyright (c) 2026-2027 zh
"""内容：提供隐藏层使用 LayerNorm 的 ELU MLP 及局部 BF16 计算边界。"""

import torch
from torch import Tensor, nn


class Mlp(nn.Module):
    """隐藏层依次执行线性映射、逐样本归一化和ELU，输出保持线性。"""

    def __init__(self, dims, last_activation=False):
        super().__init__()
        layers, width = [], dims[0]
        for hidden_width in dims[1:-1]:
            layers.extend((nn.Linear(width, hidden_width), nn.LayerNorm(hidden_width), nn.ELU()))
            width = hidden_width
        layers.append(nn.Linear(width, dims[-1]))
        if last_activation:
            layers.append(nn.ELU())
        self.network = nn.Sequential(*layers)
        self.use_bf16 = False

    def forward(self, values: Tensor) -> Tensor:
        if self.use_bf16:
            # 恢复 FP32 后再进入归一化、循环记忆或物理任务计算。
            with torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
                outputs = self.network(values)
            return outputs.float()
        return self.network(values)


