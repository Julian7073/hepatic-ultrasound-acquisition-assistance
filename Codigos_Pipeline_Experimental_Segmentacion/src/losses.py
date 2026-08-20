"""Funciones de perdida para segmentacion binaria."""

import torch
import torch.nn as nn


def dice_loss(logits: torch.Tensor, targets: torch.Tensor, epsilon: float = 1e-7) -> torch.Tensor:
    """Dice loss suave calculada desde logits."""
    probabilities = torch.sigmoid(logits)
    dimensions = tuple(range(1, probabilities.ndim))
    intersection = torch.sum(probabilities * targets, dim=dimensions)
    denominator = torch.sum(probabilities + targets, dim=dimensions)
    dice = (2.0 * intersection + epsilon) / (denominator + epsilon)
    return 1.0 - dice.mean()


def combined_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Combina BCE y Dice para equilibrar pixeles y solapamiento."""
    return nn.functional.binary_cross_entropy_with_logits(logits, targets) + dice_loss(logits, targets)
