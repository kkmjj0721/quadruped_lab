# Copyright (c) 2026-2027 zh
"""内容：导出只包含深度视觉更新与单 Actor 的有状态 TorchScript 策略。"""

from copy import deepcopy
import json
import os
from pathlib import Path

import torch
from torch import Tensor, nn


class DepthDeployment(nn.Module):
    """主机维护相机时钟与回合重置，图中只接收部署可获取的四个输入。"""

    def __init__(self, actor):
        super().__init__()
        self.actor = deepcopy(actor).cpu().float().eval()
        self.actor.actor.use_bf16 = False

    def forward(self, history: Tensor, image: Tensor, hidden: Tensor, new_frame: Tensor) -> tuple[Tensor, Tensor]:
        hidden, latent = self.actor.encode(image, hidden, new_frame)
        return self.actor(history, latent), hidden


def export_depth_policy(actor, path, *, metadata):
    path = Path(path)
    module = torch.jit.script(DepthDeployment(actor))
    pending = path.with_suffix(path.suffix + ".tmp")
    torch.jit.save(module, str(pending), _extra_files={"metadata.json": json.dumps(metadata, ensure_ascii=False)})
    os.replace(pending, path)
    return path
