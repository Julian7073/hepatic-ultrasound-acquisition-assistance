"""Metricas por imagen para segmentacion binaria y clases vacias."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch


def logits_to_mask(logits: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """Convierte logits en mascaras 0/1."""
    return (torch.sigmoid(logits) >= threshold).float()


def per_image_rows(predictions: torch.Tensor, targets: torch.Tensor, filenames: list[str]) -> list[dict]:
    """Calcula metricas por imagen sin ocultar los casos vacios."""
    pred = predictions.reshape(predictions.shape[0], -1)
    target = targets.reshape(targets.shape[0], -1)
    tp = (pred * target).sum(dim=1)
    fp = (pred * (1.0 - target)).sum(dim=1)
    fn = ((1.0 - pred) * target).sum(dim=1)
    epsilon = 1e-7
    rows = []
    for index, filename in enumerate(filenames):
        gt_area = float(target[index].sum().item())
        pred_area = float(pred[index].sum().item())
        rows.append({
            "filename": filename,
            "gt_positive": gt_area > 0,
            "gt_area_px": gt_area,
            "pred_area_px": pred_area,
            "dice": float(((2 * tp[index] + epsilon) / (2 * tp[index] + fp[index] + fn[index] + epsilon)).item()),
            "iou": float(((tp[index] + epsilon) / (tp[index] + fp[index] + fn[index] + epsilon)).item()),
            "precision": float(((tp[index] + epsilon) / (tp[index] + fp[index] + epsilon)).item()),
            "recall": float(((tp[index] + epsilon) / (tp[index] + fn[index] + epsilon)).item()),
        })
    return rows


def summarize_rows(rows: list[dict], prefix: str) -> dict:
    """Resume metricas globales, positivas y falsos positivos en GT vacio."""
    frame = pd.DataFrame(rows)
    if frame.empty:
        return {}
    positive = frame[frame["gt_positive"]]
    empty = frame[~frame["gt_positive"]]
    false_positive_empty = empty[empty["pred_area_px"] > 0]
    summary = {
        f"{prefix}_images": len(frame),
        f"{prefix}_dice": frame["dice"].mean(),
        f"{prefix}_iou": frame["iou"].mean(),
        f"{prefix}_precision": frame["precision"].mean(),
        f"{prefix}_recall": frame["recall"].mean(),
        f"{prefix}_f1": frame["dice"].mean(),
        f"{prefix}_mean_gt_area_px": frame["gt_area_px"].mean(),
        f"{prefix}_mean_pred_area_px": frame["pred_area_px"].mean(),
        f"{prefix}_positive_images": len(positive),
        f"{prefix}_empty_gt_images": len(empty),
        f"{prefix}_positive_dice": positive["dice"].mean() if len(positive) else np.nan,
        f"{prefix}_positive_iou": positive["iou"].mean() if len(positive) else np.nan,
        f"{prefix}_positive_precision": positive["precision"].mean() if len(positive) else np.nan,
        f"{prefix}_positive_recall": positive["recall"].mean() if len(positive) else np.nan,
        f"{prefix}_empty_gt_false_positive_images": len(false_positive_empty),
        f"{prefix}_empty_gt_false_positive_rate": len(false_positive_empty) / max(len(empty), 1),
        f"{prefix}_empty_gt_mean_pred_area_px": empty["pred_area_px"].mean() if len(empty) else 0.0,
        f"{prefix}_positive_mean_gt_area_px": positive["gt_area_px"].mean() if len(positive) else np.nan,
        f"{prefix}_positive_mean_pred_area_px": positive["pred_area_px"].mean() if len(positive) else np.nan,
    }
    summary[f"{prefix}_combined_la_score"] = (
        summary[f"{prefix}_positive_dice"] - summary[f"{prefix}_empty_gt_false_positive_rate"]
        if len(positive) else np.nan
    )
    return summary
