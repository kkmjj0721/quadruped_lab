# Copyright (c) 2026-2027 zh
"""内容：根据实际触点构造三维高斯软标签，并计算有效足目标的交叉熵。"""

import math

import torch
from torch import Tensor


def foothold_loss(logits: Tensor, gt_candidates: Tensor, target_contacts: Tensor, valid: Tensor,
                  candidate_valid: Tensor, bandwidth: float = 0.1) -> Tensor:
    """有效位由任务的触地标签给出；GT 坐标和触点不接受反向传播。

    candidate_valid 只标记训练侧几何标签，不进入部署选点。无效坐标和
    足目标先替换再计算，避免 NaN／Inf 进入 softmax；没有有效候选的行
    不参与有效足归约，全无效批次返回连通 logits 的零损失和零梯度。
    """
    if logits.dim() != 3:
        raise ValueError("logits must have shape [B, feet, candidates]")
    batch, feet, candidates = logits.shape
    if gt_candidates.shape != (batch, candidates, 3) or target_contacts.shape != (batch, feet, 3):
        raise ValueError("Ground-truth candidate or contact dimensions do not match logits")
    if valid.shape != (batch, feet) or valid.dtype != torch.bool:
        raise ValueError("valid must be a boolean tensor of shape [B, feet]")
    if candidate_valid.shape != (batch, candidates) or candidate_valid.dtype != torch.bool:
        raise ValueError("candidate_valid must be a boolean tensor of shape [B, candidates]")
    if not math.isfinite(bandwidth) or bandwidth <= 0:
        raise ValueError("bandwidth must be finite and positive")
    has_candidates = candidate_valid.any(dim=-1)
    effective_valid = valid & has_candidates.unsqueeze(-1)
    with torch.no_grad():
        if not torch.isfinite(gt_candidates[candidate_valid]).all():
            raise ValueError("Valid candidate coordinates must be finite")
        if not torch.isfinite(target_contacts[effective_valid]).all():
            raise ValueError("Valid target contact coordinates must be finite")
        candidates_safe = torch.where(candidate_valid.unsqueeze(-1), gt_candidates, 0.0)
        contacts_safe = torch.where(effective_valid.unsqueeze(-1), target_contacts, 0.0)
        distance_squared = (candidates_safe[:, None] - contacts_safe.unsqueeze(-2)).square().sum(dim=-1)
        label_logits = (-distance_squared / (2 * bandwidth * bandwidth)).masked_fill(
            ~candidate_valid.unsqueeze(1), -torch.inf)
        # 全无候选行先消除全负无穷 softmax，再将其所有标签置零。
        label_logits = torch.where(has_candidates[:, None, None], label_logits, 0.0)
        labels = torch.where(candidate_valid[:, None], label_logits.softmax(dim=-1), 0.0)
    logits_safe = torch.where(effective_valid.unsqueeze(-1), logits, 0.0)
    losses = -(labels * logits_safe.log_softmax(dim=-1)).sum(dim=-1)
    return torch.where(effective_valid, losses, 0.0).sum() / effective_valid.sum().clamp_min(1)
