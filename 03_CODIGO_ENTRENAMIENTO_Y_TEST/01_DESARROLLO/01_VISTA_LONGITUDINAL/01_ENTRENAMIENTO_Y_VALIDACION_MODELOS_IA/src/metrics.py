"""Metricas binarias para segmentacion."""

from __future__ import annotations

import torch


def logits_to_mask(logits: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """Convierte logits a mascara binaria."""
    return (torch.sigmoid(logits) >= threshold).float()


def binary_stats(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-7) -> dict[str, torch.Tensor]:
    """Calcula TP, FP, FN y metricas por batch."""
    pred = pred.float()
    target = target.float()
    dims = tuple(range(1, pred.ndim))

    tp = torch.sum(pred * target, dim=dims)
    fp = torch.sum(pred * (1.0 - target), dim=dims)
    fn = torch.sum((1.0 - pred) * target, dim=dims)

    dice = (2.0 * tp + eps) / (2.0 * tp + fp + fn + eps)
    iou = (tp + eps) / (tp + fp + fn + eps)
    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)
    f1 = dice

    return {
        "dice": dice.mean(),
        "iou": iou.mean(),
        "precision": precision.mean(),
        "recall": recall.mean(),
        "f1": f1.mean(),
    }


def dice_loss_from_logits(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """Dice loss suave desde logits."""
    probs = torch.sigmoid(logits)
    dims = tuple(range(1, probs.ndim))
    intersection = torch.sum(probs * target, dim=dims)
    denominator = torch.sum(probs + target, dim=dims)
    dice = (2.0 * intersection + eps) / (denominator + eps)
    return 1.0 - dice.mean()
