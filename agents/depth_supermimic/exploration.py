# Copyright (c) 2026-2027 zh
"""内容：按地形分组分配不加额外动作噪声的采集环境。"""

import torch
from torch import Tensor


def stratified_collection_mask(group_ids: Tensor, fraction: float) -> Tensor:
    """按已校验的分组与比例分配无动作噪声环境，不消耗随机数。

    总名额按round取整；各组按最大余数分配，组内等距取中点。
    标识相同的环境无需连续排列，零名额的小组遵循整数配额。
    """
    mask = torch.zeros_like(group_ids, dtype=torch.bool)
    groups, counts = torch.unique(group_ids, sorted=True, return_counts=True)
    quotas = counts.to(torch.float64) * fraction
    allocations = quotas.floor().long()
    remaining = round(group_ids.numel() * fraction) - int(allocations.sum())
    order = torch.argsort(quotas - allocations, descending=True, stable=True)
    allocations[order[:remaining]] += 1
    for group, count in zip(groups.tolist(), allocations.tolist()):
        if count:
            indices = (group_ids == group).nonzero(as_tuple=True)[0]
            positions = ((2 * torch.arange(count, device=group_ids.device) + 1) * indices.numel()) // (2 * count)
            mask[indices[positions]] = True
    return mask

