"""Utilidades para crear, guardar, cargar y visualizar mascaras binarias."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


def empty_mask(width: int, height: int) -> np.ndarray:
    """Crea mascara vacia uint8 con valores 0/1."""
    return np.zeros((height, width), dtype=np.uint8)


def polygon_to_mask(segmentation, width: int, height: int) -> np.ndarray:
    """Convierte segmentacion COCO poligonal a mascara."""
    mask_image = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask_image)

    if isinstance(segmentation, dict):
        return empty_mask(width, height)

    polygons = segmentation if isinstance(segmentation, list) else []
    for polygon in polygons:
        if not isinstance(polygon, list) or len(polygon) < 6:
            continue
        points = [(float(polygon[i]), float(polygon[i + 1])) for i in range(0, len(polygon) - 1, 2)]
        draw.polygon(points, outline=1, fill=1)

    return np.array(mask_image, dtype=np.uint8)


def annotations_to_mask(annotations: list[dict], width: int, height: int) -> np.ndarray:
    """Une todas las anotaciones de una clase en una sola mascara binaria."""
    output = empty_mask(width, height)
    for annotation in annotations:
        mask = polygon_to_mask(annotation.get("segmentation", []), width, height)
        output = np.maximum(output, mask)
    return output.astype(np.uint8)


def save_binary_mask(mask: np.ndarray, path: Path) -> None:
    """Guarda mascara 0/1 como PNG 0/255."""
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((mask.astype(np.uint8) > 0).astype(np.uint8) * 255).save(path)


def load_binary_mask(path: Path, expected_size: tuple[int, int] | None = None) -> np.ndarray:
    """Carga mascara PNG y la devuelve como arreglo 0/1."""
    if not path.exists():
        raise FileNotFoundError(f"No existe mascara: {path}")
    with Image.open(path) as image:
        image = image.convert("L")
        if expected_size is not None and image.size != expected_size:
            raise ValueError(f"Tamano de mascara no coincide: {path} {image.size} != {expected_size}")
        return (np.array(image) > 0).astype(np.uint8)


def mask_area(mask: np.ndarray) -> int:
    """Cuenta pixeles activos."""
    return int(np.sum(mask > 0))


def dilate_mask(mask: np.ndarray, kernel_size: int = 15, iterations: int = 1) -> np.ndarray:
    """Dilata una mascara usando MaxFilter de Pillow."""
    if kernel_size < 1:
        return (mask > 0).astype(np.uint8)
    if kernel_size % 2 == 0:
        kernel_size += 1

    image = Image.fromarray((mask > 0).astype(np.uint8) * 255)
    for _ in range(max(1, iterations)):
        image = image.filter(ImageFilter.MaxFilter(kernel_size))
    return (np.array(image) > 0).astype(np.uint8)


def mask_edges(mask: np.ndarray) -> np.ndarray:
    """Calcula borde simple de mascara para control visual."""
    binary = (mask > 0).astype(np.uint8)
    padded = np.pad(binary, 1, mode="constant")
    neighbors = (
        padded[:-2, 1:-1]
        + padded[2:, 1:-1]
        + padded[1:-1, :-2]
        + padded[1:-1, 2:]
    )
    return ((binary == 1) & (neighbors < 4)).astype(np.uint8)


def overlay_masks(
    image_path: Path,
    masks: dict[str, np.ndarray],
    colors: dict[str, tuple[int, int, int]],
    output_path: Path,
    alpha: float = 0.35,
) -> None:
    """Superpone mascaras y contornos sobre una imagen original."""
    with Image.open(image_path) as image:
        base = image.convert("RGB")

    base_arr = np.array(base).astype(np.float32)
    overlay_arr = base_arr.copy()

    for class_name, mask in masks.items():
        color = np.array(colors[class_name], dtype=np.float32)
        active = mask > 0
        overlay_arr[active] = (1 - alpha) * overlay_arr[active] + alpha * color
        overlay_arr[mask_edges(mask) > 0] = color

    output = Image.fromarray(np.clip(overlay_arr, 0, 255).astype(np.uint8))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.save(output_path)
