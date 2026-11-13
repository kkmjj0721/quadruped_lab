# Copyright (c) 2026-2027 zh
"""
内容：
    查询 MGDP 原版监督高程和脚下边缘；无需依赖碰撞射线最高交点。
"""

import torch

from .generator import MgdpTerrainData


class MgdpTerrainQuery:
    """无噪声硬查询；想象状态所需可微代理由任务层单独定义。"""

    def __init__(self, data: MgdpTerrainData, device: str | torch.device):
        self.data = data
        self.height_fields = tuple(torch.as_tensor(f.label_height_raw, device=device) for f in data.families)
        self.edge_fields = tuple(torch.as_tensor(f.edge_mask, device=device) for f in data.families)
        self.world_offsets = tuple(torch.as_tensor(f.world_offset[:2], dtype=torch.float32, device=device) for f in data.families)
        self.family_ids = tuple(0 if f.name == "mix" else 1 for f in data.families)

    @staticmethod
    def _family_mask(family, xy, family_id):
        family = torch.as_tensor(family, device=xy.device)
        while family.ndim < xy.ndim - 1:
            family = family.unsqueeze(-1)
        return family == family_id

    def _indices(self, xy, family_id):
        family = self.data.families[family_id]
        # 原版 .long() 为朝零截断；边界保留一格给 min 三点查询。
        indices = ((xy - self.world_offsets[family_id] + family.border_size) / family.horizontal_scale).long()
        nx, ny = family.label_height_raw.shape
        return indices[..., 0].clamp(0, nx - 2), indices[..., 1].clamp(0, ny - 2)

    def height(self, xy: torch.Tensor, family: torch.Tensor | int) -> torch.Tensor:
        """xy[...,2] 为世界坐标；family 支持标量或批次前缀，如 [B] 配 [B,P,2]。"""
        result = torch.zeros(xy.shape[:-1], device=xy.device, dtype=xy.dtype)
        for index, heights in enumerate(self.height_fields):
            x, y = self._indices(xy, index)
            value = torch.minimum(torch.minimum(heights[x, y], heights[x + 1, y]), heights[x, y + 1])
            value = value.to(xy.dtype) * self.data.families[index].vertical_scale
            result = torch.where(self._family_mask(family, xy, self.family_ids[index]), value, result)
        return result

    def foot_edges(self, xy: torch.Tensor, family: torch.Tensor | int) -> torch.Tensor:
        """保持原版 x_edge_mask 硬索引，不按悬空标签重新生成边缘。"""
        result = torch.zeros(xy.shape[:-1], device=xy.device, dtype=torch.bool)
        for index, edges in enumerate(self.edge_fields):
            cfg = self.data.families[index]
            indices = ((xy - self.world_offsets[index] + cfg.border_size) / cfg.horizontal_scale).round().long()
            x = indices[..., 0].clamp(0, edges.shape[0] - 1)
            y = indices[..., 1].clamp(0, edges.shape[1] - 1)
            result = torch.where(self._family_mask(family, xy, self.family_ids[index]), edges[x, y], result)
        return result
