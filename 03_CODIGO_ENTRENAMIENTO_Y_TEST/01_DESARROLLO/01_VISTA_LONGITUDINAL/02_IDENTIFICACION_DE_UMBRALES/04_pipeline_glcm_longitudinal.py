"""Pipeline longitudinal: mascaras, dilatacion, AND y metricas GLCM."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from config import (
    GLCM_LEVELS,
    GLCM_OFFSETS,
    LA_DILATION_KERNEL_SIZE,
    MASKS_ROOT,
    METRICS_ROOT,
    MIN_LA_AREA_PX,
    ROBOFLOW_COCO_ROOT,
    SPLITS,
    annotations_path,
    ensure_output_dirs,
)
from utils.coco_utils import load_coco_json, split_image_path
from utils.mask_utils import dilate_mask, load_binary_mask, mask_area, save_binary_mask
from utils.metadata_utils import mask_filename_for_image, metadata_from_filename
from utils.texture_utils import glcm_properties, region_intensity_stats


def safe_nan(value: float) -> float:
    """Normaliza NaN para escritura en CSV."""
    try:
        if math.isnan(float(value)):
            return float("nan")
    except Exception:
        return float("nan")
    return float(value)


def load_gray_image(path: Path) -> np.ndarray:
    """Carga imagen original en escala de grises sin modificar contraste."""
    with Image.open(path) as image:
        return np.array(image.convert("L"))


def process_image(split: str, image_info: dict) -> dict:
    """Procesa una imagen y devuelve metricas."""
    filename = image_info["file_name"]
    metadata = metadata_from_filename(filename)
    image_path = split_image_path(ROBOFLOW_COCO_ROOT, split, filename)
    mask_name = mask_filename_for_image(filename)
    expected_size = (int(image_info["width"]), int(image_info["height"]))
    notes = []

    if not image_path.exists():
        raise FileNotFoundError(f"No existe imagen: {image_path}")

    gray = load_gray_image(image_path)
    if gray.shape[::-1] != expected_size:
        notes.append("tamano_imagen_distinto_a_coco")
        expected_size = gray.shape[::-1]

    masks = {}
    for class_name in ["ROI", "Higado", "LA"]:
        mask_path = MASKS_ROOT / split / class_name / mask_name
        if mask_path.exists():
            masks[class_name] = load_binary_mask(mask_path, expected_size=expected_size)
        else:
            notes.append(f"falta_mascara_{class_name}")
            masks[class_name] = np.zeros_like(gray, dtype=np.uint8)

    roi_mask = masks["ROI"]
    higado_mask = masks["Higado"]
    la_mask = masks["LA"]
    la_dilated = dilate_mask(la_mask, kernel_size=LA_DILATION_KERNEL_SIZE)
    and_mask = ((roi_mask > 0) & (la_dilated > 0)).astype(np.uint8)

    save_binary_mask(la_dilated, MASKS_ROOT / split / "LA_dilated" / mask_name)
    save_binary_mask(and_mask, MASKS_ROOT / split / "ROI_AND_LA_dilated" / mask_name)

    roi_area = mask_area(roi_mask)
    higado_area = mask_area(higado_mask)
    la_area = mask_area(la_mask)
    la_dilated_area = mask_area(la_dilated)
    and_area = mask_area(and_mask)

    la_stats = region_intensity_stats(gray, la_mask)
    and_stats = region_intensity_stats(gray, and_mask)
    glcm = glcm_properties(gray, and_mask, levels=GLCM_LEVELS, offsets=GLCM_OFFSETS)

    has_roi = int(roi_area > 0)
    has_higado = int(higado_area > 0)
    has_la = int(la_area > 0)

    acceptable_initial = int(has_roi == 1 and has_la == 1 and la_area >= MIN_LA_AREA_PX and and_area >= MIN_LA_AREA_PX)
    if has_la == 0:
        notes.append("sin_LA_anotada")
    if 0 < la_area < MIN_LA_AREA_PX:
        notes.append("LA_area_muy_pequena")
    if acceptable_initial == 1:
        notes.append("regla_inicial_area_ok")

    return {
        "split": split,
        "filename": filename,
        "patient": metadata["patient"],
        "view": metadata["view"],
        "quality": metadata["quality"],
        "has_roi": has_roi,
        "has_higado": has_higado,
        "has_la": has_la,
        "roi_area_px": roi_area,
        "higado_area_px": higado_area,
        "la_area_px": la_area,
        "la_dilated_area_px": la_dilated_area,
        "and_area_px": and_area,
        "lumen_ratio_roi": float(and_area / roi_area) if roi_area > 0 else float("nan"),
        "la_mean_intensity": safe_nan(la_stats["mean"]),
        "la_std_intensity": safe_nan(la_stats["std"]),
        "and_mean_intensity": safe_nan(and_stats["mean"]),
        "and_std_intensity": safe_nan(and_stats["std"]),
        "glcm_contrast": safe_nan(glcm["contrast"]),
        "glcm_entropy": safe_nan(glcm["entropy"]),
        "glcm_homogeneity": safe_nan(glcm["homogeneity"]),
        "glcm_energy": safe_nan(glcm["energy"]),
        "glcm_valid_pairs": int(glcm["valid_pairs"]),
        "acceptable_lumen_rule": acceptable_initial,
        "notes": ";".join(notes),
    }


def main() -> None:
    """Ejecuta el pipeline GLCM para train/valid/test de Roboflow."""
    ensure_output_dirs()
    rows = []

    for split in SPLITS:
        json_path = annotations_path(split)
        if not json_path.exists():
            print(f"ADVERTENCIA: no existe {json_path}")
            continue

        coco = load_coco_json(json_path)
        for image_info in tqdm(coco.get("images", []), desc=f"GLCM {split}"):
            try:
                rows.append(process_image(split, image_info))
            except Exception as exc:
                rows.append(
                    {
                        "split": split,
                        "filename": image_info.get("file_name", "unknown"),
                        "patient": "unknown",
                        "view": "unknown",
                        "quality": "unknown",
                        "has_roi": 0,
                        "has_higado": 0,
                        "has_la": 0,
                        "roi_area_px": 0,
                        "higado_area_px": 0,
                        "la_area_px": 0,
                        "la_dilated_area_px": 0,
                        "and_area_px": 0,
                        "lumen_ratio_roi": float("nan"),
                        "la_mean_intensity": float("nan"),
                        "la_std_intensity": float("nan"),
                        "and_mean_intensity": float("nan"),
                        "and_std_intensity": float("nan"),
                        "glcm_contrast": float("nan"),
                        "glcm_entropy": float("nan"),
                        "glcm_homogeneity": float("nan"),
                        "glcm_energy": float("nan"),
                        "glcm_valid_pairs": 0,
                        "acceptable_lumen_rule": 0,
                        "notes": f"error:{exc}",
                    }
                )

    output_csv = METRICS_ROOT / "glcm_longitudinal_metrics.csv"
    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"Metricas guardadas: {output_csv}")
    print(f"Imagenes procesadas: {len(df)}")


if __name__ == "__main__":
    main()
