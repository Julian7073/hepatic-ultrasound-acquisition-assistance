"""Preprocesamiento no destructivo para aislar el campo ecografico."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import cv2
import numpy as np
from PIL import Image


@dataclass(frozen=True)
class FanCropResult:
    image: np.ndarray
    mask: np.ndarray
    bbox_x: int
    bbox_y: int
    bbox_width: int
    bbox_height: int
    detected: int
    mask_fraction: float

    def metadata(self) -> dict:
        values = asdict(self)
        values.pop("image")
        values.pop("mask")
        return values


def _largest_component(binary: np.ndarray) -> np.ndarray:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    if count <= 1:
        return np.zeros_like(binary)
    candidates = stats[1:, cv2.CC_STAT_AREA]
    label = int(np.argmax(candidates)) + 1
    return (labels == label).astype(np.uint8)


def isolate_ultrasound_fan(rgb: np.ndarray, margin: int = 8) -> FanCropResult:
    """Detecta el componente ecografico central y devuelve su recorte enmascarado."""
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("Se esperaba una imagen RGB de tres canales.")
    height, width = rgb.shape[:2]
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    work = np.zeros_like(gray, dtype=np.uint8)
    y0 = int(round(height * 0.10))
    x0 = int(round(width * 0.07))
    x1 = int(round(width * 0.93))
    border_samples = np.concatenate((
        gray[: max(1, y0 // 2), :].ravel(),
        gray[:, : max(1, x0 // 2)].ravel(),
        gray[:, min(width - 1, x1 + (width - x1) // 2):].ravel(),
    ))
    background_level = float(np.median(border_samples))
    intensity_threshold = max(24.0, background_level + 8.0)
    work[y0:, x0:x1] = (
        gray[y0:, x0:x1] > intensity_threshold
    ).astype(np.uint8)

    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))
    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    work = cv2.morphologyEx(work, cv2.MORPH_CLOSE, close_kernel, iterations=2)
    work = cv2.morphologyEx(work, cv2.MORPH_OPEN, open_kernel, iterations=1)
    component = _largest_component(work)

    ys, xs = np.where(component > 0)
    plausible = len(xs) >= int(0.08 * width * height)
    if plausible:
        bx0 = max(0, int(xs.min()) - margin)
        by0 = max(y0, int(ys.min()) - margin)
        bx1 = min(width, int(xs.max()) + margin + 1)
        by1 = min(height, int(ys.max()) + margin + 1)
        bbox_width = bx1 - bx0
        bbox_height = by1 - by0
        plausible = (
            bbox_width >= int(width * 0.35)
            and bbox_height >= int(height * 0.35)
            and bbox_width <= int(width * 0.92)
        )

    if not plausible:
        mask = np.ones((height, width), dtype=np.uint8)
        return FanCropResult(
            image=rgb.copy(), mask=mask, bbox_x=0, bbox_y=0,
            bbox_width=width, bbox_height=height, detected=0,
            mask_fraction=1.0,
        )

    mask = component.astype(np.uint8)
    masked = rgb.copy()
    masked[mask == 0] = 0
    cropped = masked[by0:by1, bx0:bx1]
    return FanCropResult(
        image=cropped, mask=mask, bbox_x=bx0, bbox_y=by0,
        bbox_width=bbox_width, bbox_height=bbox_height, detected=1,
        mask_fraction=float(mask.mean()),
    )


def fan_crop_pil(image: Image.Image) -> Image.Image:
    rgb = np.asarray(image.convert("RGB"))
    result = isolate_ultrasound_fan(rgb)
    return Image.fromarray(result.image)