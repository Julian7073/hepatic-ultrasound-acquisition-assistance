"""Paneles cualitativos de imagen, GT, prediccion y overlay."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch

from src.metrics import logits_to_mask


def tensor_rgb(tensor: torch.Tensor) -> np.ndarray:
    """Convierte tensor RGB CHW a uint8 HWC."""
    image = tensor.detach().cpu().numpy().transpose(1, 2, 0)
    return np.clip(image * 255.0, 0, 255).astype(np.uint8)


def tensor_mask(tensor: torch.Tensor) -> np.ndarray:
    """Convierte mascara tensorial en 0/255."""
    return ((tensor.detach().cpu().numpy().squeeze() > 0.5).astype(np.uint8) * 255)


def save_panel(image, ground_truth, prediction, output_path: Path) -> None:
    """Guarda original, GT, prediccion y overlay en un solo PNG."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rgb = tensor_rgb(image)
    gt = tensor_mask(ground_truth)
    pred = tensor_mask(prediction)
    overlay = rgb.copy().astype(np.float32)
    active = pred > 0
    overlay[active] = 0.60 * overlay[active] + 0.40 * np.array([255, 40, 40], dtype=np.float32)
    gt_rgb = cv2.cvtColor(gt, cv2.COLOR_GRAY2RGB)
    pred_rgb = cv2.cvtColor(pred, cv2.COLOR_GRAY2RGB)
    panel = np.concatenate([rgb, gt_rgb, pred_rgb, overlay.astype(np.uint8)], axis=1)
    cv2.imwrite(str(output_path), cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))


@torch.no_grad()
def save_selected_overlays(model, loader, device, output_dir: Path, per_image_rows: list[dict], max_samples: int) -> None:
    """Guarda mejores positivos y peores casos segun Dice."""
    if max_samples <= 0:
        return
    ordered = sorted(per_image_rows, key=lambda row: row["dice"])
    worst = ordered[: min(3, max_samples)]
    positives = sorted((row for row in ordered if row["gt_positive"]), key=lambda row: row["dice"], reverse=True)
    selected = worst + positives[: max(0, max_samples - len(worst))]
    selected_names = {row["filename"] for row in selected}
    saved = 0
    model.eval()
    for batch in loader:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)
        predictions = logits_to_mask(model(images))
        for index, filename in enumerate(batch["filename"]):
            if filename not in selected_names:
                continue
            label = "worst" if any(row["filename"] == filename for row in worst) else "positive"
            safe_name = Path(filename).stem[:100]
            save_panel(images[index], masks[index], predictions[index], output_dir / f"{label}_{safe_name}.png")
            saved += 1
            if saved >= len(selected_names):
                return
