"""Metricas de intensidad y textura GLCM para la vista longitudinal."""

from __future__ import annotations

import math

import numpy as np


def region_intensity_stats(gray: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    """Calcula media y desviacion estandar dentro de una mascara."""
    values = gray[mask > 0]
    if values.size == 0:
        return {"mean": float("nan"), "std": float("nan")}
    return {"mean": float(np.mean(values)), "std": float(np.std(values))}


def quantize_gray(gray: np.ndarray, levels: int) -> np.ndarray:
    """Reduce la imagen de 0-255 a pocos niveles para GLCM."""
    clipped = np.clip(gray.astype(np.float32), 0, 255)
    quantized = np.floor(clipped * levels / 256.0).astype(np.int32)
    return np.clip(quantized, 0, levels - 1)


def glcm_for_offset(
    gray_q: np.ndarray,
    mask: np.ndarray,
    levels: int,
    dy: int,
    dx: int,
) -> tuple[np.ndarray, int]:
    """Construye una GLCM usando solo pares de pixeles dentro de la mascara."""
    height, width = gray_q.shape

    y0_start = max(0, -dy)
    y0_end = min(height, height - dy)
    x0_start = max(0, -dx)
    x0_end = min(width, width - dx)

    y1_start = y0_start + dy
    y1_end = y0_end + dy
    x1_start = x0_start + dx
    x1_end = x0_end + dx

    base_mask = mask[y0_start:y0_end, x0_start:x0_end] > 0
    neighbor_mask = mask[y1_start:y1_end, x1_start:x1_end] > 0
    valid = base_mask & neighbor_mask

    if not np.any(valid):
        return np.zeros((levels, levels), dtype=np.float64), 0

    i_values = gray_q[y0_start:y0_end, x0_start:x0_end][valid]
    j_values = gray_q[y1_start:y1_end, x1_start:x1_end][valid]

    matrix = np.zeros((levels, levels), dtype=np.float64)
    np.add.at(matrix, (i_values, j_values), 1)
    np.add.at(matrix, (j_values, i_values), 1)
    return matrix, int(i_values.size)


def glcm_properties(
    gray: np.ndarray,
    mask: np.ndarray,
    levels: int,
    offsets: list[tuple[int, int]],
) -> dict[str, float | int]:
    """Calcula contraste, entropia, homogeneidad y energia promediando offsets."""
    if int(np.sum(mask > 0)) < 2:
        return {
            "contrast": float("nan"),
            "entropy": float("nan"),
            "homogeneity": float("nan"),
            "energy": float("nan"),
            "valid_pairs": 0,
        }

    gray_q = quantize_gray(gray, levels)
    ii, jj = np.indices((levels, levels))
    values = []
    total_pairs = 0

    for dy, dx in offsets:
        matrix, n_pairs = glcm_for_offset(gray_q, mask, levels, dy, dx)
        total_pairs += n_pairs
        if n_pairs == 0 or matrix.sum() == 0:
            continue

        probability = matrix / matrix.sum()
        nonzero = probability[probability > 0]
        contrast = float(np.sum(((ii - jj) ** 2) * probability))
        entropy = float(-np.sum(nonzero * np.log2(nonzero)))
        homogeneity = float(np.sum(probability / (1.0 + np.abs(ii - jj))))
        energy = float(math.sqrt(np.sum(probability**2)))
        values.append((contrast, entropy, homogeneity, energy))

    if not values:
        return {
            "contrast": float("nan"),
            "entropy": float("nan"),
            "homogeneity": float("nan"),
            "energy": float("nan"),
            "valid_pairs": total_pairs,
        }

    arr = np.array(values, dtype=np.float64)
    return {
        "contrast": float(np.mean(arr[:, 0])),
        "entropy": float(np.mean(arr[:, 1])),
        "homogeneity": float(np.mean(arr[:, 2])),
        "energy": float(np.mean(arr[:, 3])),
        "valid_pairs": int(total_pairs),
    }
