"""Visualizacion de predicciones de segmentacion."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch


def tensor_to_rgb(image_tensor: torch.Tensor) -> np.ndarray:
    """Convierte tensor [3,H,W] a RGB uint8."""
    image = image_tensor.detach().cpu().numpy().transpose(1, 2, 0)
    return np.clip(image * 255.0, 0, 255).astype(np.uint8)


def mask_to_uint8(mask_tensor: torch.Tensor) -> np.ndarray:
    """Convierte mascara [1,H,W] a uint8 0/255."""
    mask = mask_tensor.detach().cpu().numpy().squeeze()
    return ((mask > 0.5).astype(np.uint8) * 255)


def create_overlay(image_rgb: np.ndarray, mask: np.ndarray, color=(255, 40, 40), alpha: float = 0.35) -> np.ndarray:
    """Crea overlay simple de mascara sobre imagen."""
    overlay = image_rgb.copy().astype(np.float32)
    active = mask > 0
    overlay[active] = (1 - alpha) * overlay[active] + alpha * np.array(color, dtype=np.float32)
    return np.clip(overlay, 0, 255).astype(np.uint8)


def save_prediction_panel(
    image_tensor: torch.Tensor,
    gt_mask_tensor: torch.Tensor,
    pred_mask_tensor: torch.Tensor,
    output_path: Path,
) -> None:
    """Guarda panel: original, GT, prediccion y overlay."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    image = tensor_to_rgb(image_tensor)
    gt = mask_to_uint8(gt_mask_tensor)
    pred = mask_to_uint8(pred_mask_tensor)
    overlay = create_overlay(image, pred)

    gt_rgb = cv2.cvtColor(gt, cv2.COLOR_GRAY2RGB)
    pred_rgb = cv2.cvtColor(pred, cv2.COLOR_GRAY2RGB)
    panel = np.concatenate([image, gt_rgb, pred_rgb, overlay], axis=1)
    panel_bgr = cv2.cvtColor(panel, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(output_path), panel_bgr)
