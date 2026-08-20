"""Base de inferencia frame por frame para la futura GUI."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch

from config_segmentation import FINAL_MODELS_ROOT
from src.models import create_model


MIN_LA_AREA_PX = 874
MAX_LA_STD_INTENSITY = 43.3302
MAX_GLCM_ENTROPY = 6.1257


@dataclass
class FrameDecision:
    """Resultado resumido para un frame."""

    decision: str
    message: str
    has_roi: int
    has_higado: int
    has_la: int
    area_roi_px: int
    area_higado_px: int
    area_la_px: int
    higado_roi_ratio: float
    la_std_intensity: float | None
    glcm_entropy: float | None
    glcm_contrast: float | None
    glcm_homogeneity: float | None
    glcm_energy: float | None


def load_binary_model(checkpoint_path: Path, device: torch.device):
    """Carga un checkpoint final de segmentacion binaria."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = create_model(checkpoint["architecture"], image_size=int(checkpoint.get("image_size", 512))).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, int(checkpoint.get("image_size", 512))


def preprocess_frame(frame_bgr: np.ndarray, image_size: int) -> torch.Tensor:
    """Preprocesa frame BGR de OpenCV a tensor RGB normalizado."""
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(frame_rgb, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
    tensor = torch.from_numpy(resized.astype(np.float32).transpose(2, 0, 1) / 255.0).unsqueeze(0)
    return tensor


@torch.no_grad()
def predict_mask(model, frame_bgr: np.ndarray, image_size: int, device: torch.device, threshold: float = 0.5) -> np.ndarray:
    """Predice mascara binaria y la devuelve al tamano original del frame."""
    tensor = preprocess_frame(frame_bgr, image_size).to(device)
    logits = model(tensor)
    pred = (torch.sigmoid(logits)[0, 0].detach().cpu().numpy() >= threshold).astype(np.uint8)
    original_h, original_w = frame_bgr.shape[:2]
    return cv2.resize(pred, (original_w, original_h), interpolation=cv2.INTER_NEAREST)


def mask_area(mask: np.ndarray) -> int:
    """Cuenta pixeles activos."""
    return int(np.sum(mask > 0))


def intensity_std(frame_bgr: np.ndarray, mask: np.ndarray) -> float | None:
    """Desviacion estandar de intensidad dentro de una mascara."""
    if mask_area(mask) == 0:
        return None
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    return float(np.std(gray[mask > 0]))


def quantize_gray(gray: np.ndarray, levels: int = 32) -> np.ndarray:
    """Cuantiza intensidades 0-255 para construir GLCM de forma estable."""
    clipped = np.clip(gray.astype(np.float32), 0, 255)
    quantized = np.floor(clipped * levels / 256.0).astype(np.int32)
    return np.clip(quantized, 0, levels - 1)


def glcm_features_for_mask(
    frame_bgr: np.ndarray,
    mask: np.ndarray,
    levels: int = 32,
    offsets: tuple[tuple[int, int], ...] = ((0, 1), (1, 0), (1, 1), (-1, 1)),
) -> dict[str, float | None]:
    """Calcula textura GLCM dentro de la mascara sin modificar contraste.

    Solo se usan pares de pixeles donde ambos puntos caen dentro de la mascara.
    Esto evita contaminar la textura con fondo negro externo o interfaz del ecografo.
    """
    if mask_area(mask) < 2:
        return {"contrast": None, "entropy": None, "homogeneity": None, "energy": None}

    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    gray_q = quantize_gray(gray, levels=levels)
    mask_bool = mask > 0
    ii, jj = np.indices((levels, levels))
    feature_rows = []

    height, width = gray_q.shape
    for dy, dx in offsets:
        y0_start = max(0, -dy)
        y0_end = min(height, height - dy)
        x0_start = max(0, -dx)
        x0_end = min(width, width - dx)
        y1_start = y0_start + dy
        y1_end = y0_end + dy
        x1_start = x0_start + dx
        x1_end = x0_end + dx

        valid = mask_bool[y0_start:y0_end, x0_start:x0_end] & mask_bool[y1_start:y1_end, x1_start:x1_end]
        if not np.any(valid):
            continue

        i_values = gray_q[y0_start:y0_end, x0_start:x0_end][valid]
        j_values = gray_q[y1_start:y1_end, x1_start:x1_end][valid]
        matrix = np.zeros((levels, levels), dtype=np.float64)
        np.add.at(matrix, (i_values, j_values), 1)
        np.add.at(matrix, (j_values, i_values), 1)

        probability = matrix / matrix.sum()
        nonzero = probability[probability > 0]
        contrast = float(np.sum(((ii - jj) ** 2) * probability))
        entropy = float(-np.sum(nonzero * np.log2(nonzero)))
        homogeneity = float(np.sum(probability / (1.0 + np.abs(ii - jj))))
        energy = float(math.sqrt(np.sum(probability**2)))
        feature_rows.append((contrast, entropy, homogeneity, energy))

    if not feature_rows:
        return {"contrast": None, "entropy": None, "homogeneity": None, "energy": None}

    features = np.array(feature_rows, dtype=np.float64)
    return {
        "contrast": float(np.mean(features[:, 0])),
        "entropy": float(np.mean(features[:, 1])),
        "homogeneity": float(np.mean(features[:, 2])),
        "energy": float(np.mean(features[:, 3])),
    }


def decide_longitudinal_frame(
    frame_bgr: np.ndarray,
    roi_mask: np.ndarray,
    higado_mask: np.ndarray,
    la_mask: np.ndarray,
    min_roi_area: int = 1000,
    min_higado_roi_ratio: float = 0.15,
) -> FrameDecision:
    """Aplica reglas iniciales y devuelve mensaje para usuario no experto."""
    area_roi = mask_area(roi_mask)
    area_higado = mask_area(higado_mask)
    area_la = mask_area(la_mask)
    has_roi = int(area_roi >= min_roi_area)
    has_higado = int(area_higado > 0)
    has_la = int(area_la > 0)
    higado_roi_ratio = area_higado / area_roi if area_roi > 0 else 0.0
    la_std = intensity_std(frame_bgr, la_mask)
    texture = glcm_features_for_mask(frame_bgr, la_mask)
    entropy = texture["entropy"]

    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    mostly_black = float(np.mean(gray < 15)) > 0.80

    if not has_roi or mostly_black:
        return FrameDecision(
            "no_structure",
            "Ninguna estructura visible. Asegurese de que la sonda tenga contacto adecuado con la piel y que haya suficiente gel ecografico.",
            has_roi,
            has_higado,
            has_la,
            area_roi,
            area_higado,
            area_la,
            higado_roi_ratio,
            la_std,
            entropy,
            texture["contrast"],
            texture["homogeneity"],
            texture["energy"],
        )

    if not has_higado or higado_roi_ratio < min_higado_roi_ratio:
        return FrameDecision(
            "partial_liver",
            "Higado parcialmente visible. Mueva la sonda ligeramente hacia arriba, hacia abajo, hacia la izquierda y hacia la derecha hasta centrar mejor la estructura hepatica.",
            has_roi,
            has_higado,
            has_la,
            area_roi,
            area_higado,
            area_la,
            higado_roi_ratio,
            la_std,
            entropy,
            texture["contrast"],
            texture["homogeneity"],
            texture["energy"],
        )

    la_ok = (
        has_la == 1
        and area_la >= MIN_LA_AREA_PX
        and la_std is not None
        and entropy is not None
        and la_std <= MAX_LA_STD_INTENSITY
        and entropy <= MAX_GLCM_ENTROPY
    )
    if not la_ok:
        return FrameDecision(
            "liver_without_reference",
            "Higado visible, pero la referencia anatomica aun no es suficiente. Ajuste ligeramente la inclinacion o posicion de la sonda hasta mejorar la visualizacion interna.",
            has_roi,
            has_higado,
            has_la,
            area_roi,
            area_higado,
            area_la,
            higado_roi_ratio,
            la_std,
            entropy,
            texture["contrast"],
            texture["homogeneity"],
            texture["energy"],
        )

    return FrameDecision(
        "capture",
        "Higado visible. Mantengase en esta posicion y capture la imagen.",
        has_roi,
        has_higado,
        has_la,
        area_roi,
        area_higado,
        area_la,
        higado_roi_ratio,
        la_std,
        entropy,
        texture["contrast"],
        texture["homogeneity"],
        texture["energy"],
    )


def expected_final_model_paths() -> dict[str, Path]:
    """Rutas esperadas para inferencia con los modelos seleccionados."""
    return {
        "ROI": FINAL_MODELS_ROOT / "best_roi_model.pth",
        "Higado": FINAL_MODELS_ROOT / "best_higado_model.pth",
        "LA": FINAL_MODELS_ROOT / "best_la_model.pth",
    }
