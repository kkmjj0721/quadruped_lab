# Copyright (c) 2026-2027 zh
"""内容：固定Go2接触logit的输出尺度，保留通用世界模型的前向、自回归损失和优化。"""

import torch

from agents.supermimic.world_model import WorldModel

from ..mdp.reference_state import CONTACT_FOUND


class Go2WorldModel(WorldModel):
    """连续瞬时量按标签归一化；接触logit使用单位尺度并以概率残差监督。"""

    @torch.no_grad()
    def apply_statistics(self, statistics):
        super().apply_statistics(statistics)
        active = statistics[1].count > 0
        mean, variance = self.output_norm.mean[CONTACT_FOUND], self.output_norm.var[CONTACT_FOUND]
        mean.copy_(torch.where(active, 0, mean))
        variance.copy_(torch.where(active, 1, variance))
