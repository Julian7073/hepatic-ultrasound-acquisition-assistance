"""Reglas unificadas de calidad para inferencia longitudinal."""

from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np


PIPELINE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = (
    PIPELINE_ROOT / "configs" / "longitudinal_decision_config.json"
)
GLCM_REGION_MODES = {"la_mask", "la_dilated", "roi_and_la_dilated"}
FINAL_RULE_MODES = {
    "regla_base",
    "regla_base_plus_border",
    "regla_lumen_or_border",
}

DECISION_MESSAGES = {
    "no_structure": "Ninguna estructura visible. Revisar contacto y gel.",
    "partial_liver": "Hígado parcialmente visible. Mueva la sonda para centrarlo.",
    "liver_without_reference": (
        "Referencia anatómica insuficiente. Ajustar inclinación o posición."
    ),
    "capture": "Hígado visible. Mantener posición y capturar imagen.",
}


@lru_cache(maxsize=8)
def _load_config_cached(path_text: str) -> dict:
    path = Path(path_text)
    if not path.exists():
        raise FileNotFoundError(f"No existe la configuración longitudinal: {path}")
    config = json.loads(path.read_text(encoding="utf-8"))
    validate_decision_config(config)
    config["config_path"] = str(path.resolve())
    return config


def load_decision_config(path: str | Path | None = None) -> dict:
    """Carga y valida una configuración sin modificar el archivo original."""
    resolved = Path(path or DEFAULT_CONFIG_PATH).expanduser().resolve()
    return dict(_load_config_cached(str(resolved)))


def validate_decision_config(config: dict) -> None:
    """Valida modos y parámetros que cambian la región de medición."""
    if config.get("glcm_region_mode") not in GLCM_REGION_MODES:
        raise ValueError(
            f"glcm_region_mode inválido: {config.get('glcm_region_mode')}"
        )
    if config.get("final_rule_mode") not in FINAL_RULE_MODES:
        raise ValueError(
            f"final_rule_mode inválido: {config.get('final_rule_mode')}"
        )
    for key in ("la_dilation_kernel", "border_ring_kernel"):
        value = int(config[key])
        if value < 1 or value % 2 == 0:
            raise ValueError(f"{key} debe ser impar y >=1; recibido {value}")
    if int(config["reference_area"]) <= 0:
        raise ValueError("reference_area debe ser positivo.")


def dilate_mask(mask: np.ndarray, kernel_size: int = 15) -> np.ndarray:
    """Dilata una máscara binaria con kernel cuadrado, equivalente a MaxFilter."""
    binary = (mask > 0).astype(np.uint8)
    kernel_size = int(kernel_size)
    if kernel_size < 1:
        return binary
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    return cv2.dilate(binary, kernel, iterations=1)


def compute_valid_lumen_region(
    roi_mask: np.ndarray,
    la_mask: np.ndarray,
    mode: str,
    dilation_kernel: int = 15,
) -> np.ndarray:
    """Construye la región usada para GLCM según la configuración."""
    if mode not in GLCM_REGION_MODES:
        raise ValueError(f"Modo GLCM no soportado: {mode}")
    roi = roi_mask > 0
    la = (la_mask > 0).astype(np.uint8)
    if mode == "la_mask":
        return la
    dilated = dilate_mask(la, dilation_kernel)
    if mode == "la_dilated":
        return dilated
    return (roi & (dilated > 0)).astype(np.uint8)


def _quantize_gray(gray: np.ndarray, levels: int) -> np.ndarray:
    values = np.clip(gray.astype(np.float32), 0, 255)
    quantized = np.floor(values * levels / 256.0).astype(np.int32)
    return np.clip(quantized, 0, levels - 1)


def compute_glcm_features(
    gray: np.ndarray,
    mask: np.ndarray,
    levels: int = 32,
    offsets: tuple[tuple[int, int], ...] = (
        (0, 1),
        (1, 0),
        (1, 1),
        (-1, 1),
    ),
) -> dict[str, float | int | None]:
    """Calcula GLCM simétrica usando solo pares dentro de la máscara."""
    if int(np.count_nonzero(mask)) < 2:
        return {
            "glcm_contrast": None,
            "glcm_entropy": None,
            "glcm_homogeneity": None,
            "glcm_energy": None,
            "glcm_valid_pairs": 0,
        }

    gray_q = _quantize_gray(gray, levels)
    mask_bool = mask > 0
    ii, jj = np.indices((levels, levels))
    feature_rows = []
    valid_pairs = 0
    height, width = gray.shape

    for dy, dx in offsets:
        y0, y1 = max(0, -dy), min(height, height - dy)
        x0, x1 = max(0, -dx), min(width, width - dx)
        valid = (
            mask_bool[y0:y1, x0:x1]
            & mask_bool[y0 + dy:y1 + dy, x0 + dx:x1 + dx]
        )
        if not np.any(valid):
            continue
        first = gray_q[y0:y1, x0:x1][valid]
        second = gray_q[y0 + dy:y1 + dy, x0 + dx:x1 + dx][valid]
        valid_pairs += int(first.size)
        matrix = np.zeros((levels, levels), dtype=np.float64)
        np.add.at(matrix, (first, second), 1)
        np.add.at(matrix, (second, first), 1)
        probability = matrix / matrix.sum()
        nonzero = probability[probability > 0]
        feature_rows.append((
            float(np.sum(((ii - jj) ** 2) * probability)),
            float(-np.sum(nonzero * np.log2(nonzero))),
            float(np.sum(probability / (1.0 + np.abs(ii - jj)))),
            float(math.sqrt(np.sum(probability**2))),
        ))

    if not feature_rows:
        return {
            "glcm_contrast": None,
            "glcm_entropy": None,
            "glcm_homogeneity": None,
            "glcm_energy": None,
            "glcm_valid_pairs": valid_pairs,
        }
    values = np.asarray(feature_rows, dtype=np.float64)
    return {
        "glcm_contrast": float(np.mean(values[:, 0])),
        "glcm_entropy": float(np.mean(values[:, 1])),
        "glcm_homogeneity": float(np.mean(values[:, 2])),
        "glcm_energy": float(np.mean(values[:, 3])),
        "glcm_valid_pairs": valid_pairs,
    }


def _gradient_magnitude(gray: np.ndarray) -> np.ndarray:
    gray_float = gray.astype(np.float32)
    gx = np.zeros_like(gray_float)
    gy = np.zeros_like(gray_float)
    gx[:, 1:-1] = (gray_float[:, 2:] - gray_float[:, :-2]) / 2.0
    gy[1:-1, :] = (gray_float[2:, :] - gray_float[:-2, :]) / 2.0
    return np.sqrt(gx * gx + gy * gy)


def evaluate_border_rule(
    gray: np.ndarray,
    roi_mask: np.ndarray,
    la_mask: np.ndarray,
    config: dict,
) -> dict[str, float | int | str | None]:
    """Calcula el anillo V2 y sus tres ramas de evidencia."""
    roi = roi_mask > 0
    la = la_mask > 0
    if not np.any(la):
        return _empty_border_metrics("sin_la")

    dilated = dilate_mask(la_mask, int(config["border_ring_kernel"]))
    ring = (dilated > 0) & (~la) & roi
    if not np.any(ring):
        return _empty_border_metrics("anillo_vacio")

    la_values = gray[la].astype(np.float32)
    ring_values = gray[ring].astype(np.float32)
    gradient_values = _gradient_magnitude(gray)[ring]
    la_median = float(np.median(la_values))
    la_p10 = float(np.percentile(la_values, 10))
    ring_p90 = float(np.percentile(ring_values, 90))
    bright_delta = float(ring_p90 - la_median)
    gradient_p75 = float(np.percentile(gradient_values, 75))
    bright_ratio = float(np.mean(ring_values >= la_median + 10.0))
    local_delta = float(ring_p90 - la_p10)

    bright_ok = int(
        bright_delta >= float(config["min_border_p90_minus_la_median"])
        and bright_ratio >= float(config["min_border_high_ratio_10"])
    )
    gradient_ok = int(
        gradient_p75 >= float(config["min_border_gradient_p75"])
    )
    local_ok = int(
        local_delta >= float(config["min_border_p90_minus_la_p10"])
    )
    return {
        "border_status": "ok",
        "border_ring_area_px": int(np.count_nonzero(ring)),
        "border_bright_delta": bright_delta,
        "border_gradient_p75": gradient_p75,
        "border_ring_bright_ratio": bright_ratio,
        "border_p90_minus_la_p10": local_delta,
        "bright_border_ok": bright_ok,
        "gradient_border_ok": gradient_ok,
        "local_contrast_ok": local_ok,
        "border_evidence": int(bool(bright_ok or gradient_ok or local_ok)),
        "border_all_evidence": int(bool(bright_ok and gradient_ok and local_ok)),
    }


def _empty_border_metrics(status: str) -> dict:
    return {
        "border_status": status,
        "border_ring_area_px": 0,
        "border_bright_delta": None,
        "border_gradient_p75": None,
        "border_ring_bright_ratio": None,
        "border_p90_minus_la_p10": None,
        "bright_border_ok": 0,
        "gradient_border_ok": 0,
        "local_contrast_ok": 0,
        "border_evidence": 0,
        "border_all_evidence": 0,
    }


def compute_lumen_quality_metrics(
    frame_rgb: np.ndarray,
    roi_mask: np.ndarray,
    la_mask: np.ndarray,
    config: dict,
) -> dict:
    """Mide LA original, región GLCM configurada y evidencia de borde."""
    validate_decision_config(config)
    gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
    roi = (roi_mask > 0).astype(np.uint8)
    la = (la_mask > 0).astype(np.uint8)
    frame_area = int(gray.shape[0] * gray.shape[1])
    area_scale = frame_area / float(config["reference_area"])
    la_area = int(np.count_nonzero(la))
    roi_area = int(np.count_nonzero(roi))
    la_values = gray[la > 0]
    valid_region = compute_valid_lumen_region(
        roi,
        la,
        str(config["glcm_region_mode"]),
        int(config["la_dilation_kernel"]),
    )
    offsets = tuple(tuple(int(v) for v in pair) for pair in config["glcm_offsets"])
    glcm = compute_glcm_features(
        gray,
        valid_region,
        levels=int(config["glcm_levels"]),
        offsets=offsets,
    )
    metrics = {
        "la_area_px": la_area,
        "la_roi_ratio": la_area / max(roi_area, 1),
        "valid_lumen_area_px": int(np.count_nonzero(valid_region)),
        "la_area_threshold_scaled": max(
            1, round(float(config["min_la_area_px_ref"]) * area_scale)
        ),
        "la_presence_threshold_scaled": max(
            1, round(float(config["min_la_presence_px_ref"]) * area_scale)
        ),
        "la_mean": float(np.mean(la_values)) if la_values.size else None,
        "la_std": float(np.std(la_values)) if la_values.size else None,
        "glcm_region_mode": str(config["glcm_region_mode"]),
        "la_dilation_kernel": int(config["la_dilation_kernel"]),
        **glcm,
    }
    metrics.update(evaluate_border_rule(gray, roi, la, config))
    return metrics


def evaluate_lumen_rule(metrics: dict, config: dict) -> dict[str, int | str]:
    """Evalúa regla base y variante final configurada."""
    la_present = int(
        int(metrics.get("la_area_px", 0))
        >= int(metrics.get("la_presence_threshold_scaled", 1))
    )
    area_ok = int(
        la_present == 1
        and int(metrics.get("la_area_px", 0))
        >= int(metrics.get("la_area_threshold_scaled", 1))
    )
    la_std = metrics.get("la_std")
    entropy = metrics.get("glcm_entropy")
    std_ok = int(
        la_std is not None and float(la_std) <= float(config["max_la_std"])
    )
    entropy_ok = int(
        entropy is not None
        and float(entropy) <= float(config["max_glcm_entropy"])
    )
    base_rule = int(bool(la_present and area_ok and std_ok and entropy_ok))

    mode = str(config["final_rule_mode"])
    use_border = bool(config.get("use_border_rule", True))
    if not use_border or mode == "regla_base":
        final_rule = base_rule
    elif mode == "regla_base_plus_border":
        final_rule = int(bool(base_rule and metrics.get("border_all_evidence", 0)))
    else:
        final_rule = int(bool(base_rule and metrics.get("border_evidence", 0)))
    return {
        "la_present": la_present,
        "la_area_ok": area_ok,
        "la_std_ok": std_ok,
        "la_entropy_ok": entropy_ok,
        "base_lumen_rule": base_rule,
        "final_lumen_rule": final_rule,
        "rule_mode": mode,
        "use_border_rule": int(use_border),
    }


def final_longitudinal_decision(metrics: dict, config: dict) -> dict:
    """Devuelve una decisión y razón únicas para script y GUI."""
    rule = evaluate_lumen_rule(metrics, config)
    roi_present = int(metrics.get("roi_present", 0))
    liver_present = int(metrics.get("liver_present", 0))
    liver_ratio = float(metrics.get("liver_roi_ratio", 0.0))

    if not roi_present:
        decision = "no_structure"
        reason = "roi_absent_or_image_mostly_black"
    elif not liver_present or liver_ratio < float(config["liver_min_roi_ratio"]):
        decision = "partial_liver"
        reason = "liver_absent_small_or_incomplete"
    elif not rule["final_lumen_rule"]:
        decision = "liver_without_reference"
        failures = []
        if not rule["la_area_ok"]:
            failures.append("la_absent_or_small")
        if not rule["la_std_ok"]:
            failures.append("la_std_out_of_range")
        if not rule["la_entropy_ok"]:
            failures.append("la_entropy_out_of_range")
        if (
            rule["base_lumen_rule"]
            and rule["use_border_rule"]
            and not metrics.get("border_evidence", 0)
        ):
            failures.append("border_evidence_absent")
        reason = ";".join(failures) or "final_lumen_rule_failed"
    else:
        decision = "capture"
        reason = "roi_liver_and_final_lumen_rule_acceptable"

    return {
        **rule,
        "decision": decision,
        "decision_reason": reason,
        "message": DECISION_MESSAGES[decision],
    }
