"""Audita y visualiza el recorte automatico del campo ecografico."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config_dino import BINARY_FIGURES_ROOT, BINARY_REPORTS_ROOT, REPORTS_ROOT, ensure_directories
from src.ultrasound_preprocessing import isolate_ultrasound_fan


def table_markdown(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_Sin datos._"
    display = frame.copy().fillna("")
    lines = [
        "| " + " | ".join(display.columns) + " |",
        "| " + " | ".join(["---"] * len(display.columns)) + " |",
    ]
    for row in display.astype(str).itertuples(index=False, name=None):
        lines.append("| " + " | ".join(value.replace("|", "/") for value in row) + " |")
    return "\n".join(lines)


def main() -> None:
    ensure_directories()
    index_path = REPORTS_ROOT / "02_dino_embedding_index_stride5.csv"
    if not index_path.exists():
        raise FileNotFoundError(f"Falta el indice {index_path}")
    index = pd.read_csv(index_path)

    rows = []
    previews = []
    grouped = index.groupby(["view", "quality"], sort=True)
    for group_number, ((view, quality), group) in enumerate(grouped):
        preview_row = group[group["role"] == "development"].iloc[len(group[group["role"] == "development"]) // 2]
        previews.append(preview_row)
        for row in group.itertuples(index=False):
            with Image.open(row.image_path) as source:
                rgb = np.asarray(source.convert("RGB"))
            result = isolate_ultrasound_fan(rgb)
            rows.append({
                "patient": row.patient,
                "role": row.role,
                "view": view,
                "quality": quality,
                "video_id": row.video_id,
                "filename": row.filename,
                **result.metadata(),
            })

    audit = pd.DataFrame(rows)
    audit_path = BINARY_REPORTS_ROOT / "06_fan_crop_audit.csv"
    audit.to_csv(audit_path, index=False, encoding="utf-8-sig")
    summary = (
        audit.groupby(["role", "view", "quality"], as_index=False)
        .agg(
            images=("filename", "size"),
            detection_rate=("detected", "mean"),
            median_mask_fraction=("mask_fraction", "median"),
            median_bbox_width=("bbox_width", "median"),
            median_bbox_height=("bbox_height", "median"),
        )
    )

    figure, axes = plt.subplots(len(previews), 3, figsize=(12, 3.2 * len(previews)), dpi=130)
    for row_index, row in enumerate(previews):
        with Image.open(row.image_path) as source:
            rgb = np.asarray(source.convert("RGB"))
        result = isolate_ultrasound_fan(rgb)
        masked = rgb.copy()
        if result.detected:
            masked[result.mask == 0] = 0
        axes[row_index, 0].imshow(rgb)
        axes[row_index, 0].set_title(f"{row['view']} / {row['quality']} - original")
        axes[row_index, 1].imshow(masked)
        axes[row_index, 1].set_title("campo detectado")
        axes[row_index, 2].imshow(result.image)
        axes[row_index, 2].set_title(f"recorte (detected={result.detected})")
        for axis in axes[row_index]:
            axis.axis("off")
    figure.tight_layout()
    figure_path = BINARY_FIGURES_ROOT / "06_fan_crop_qc.png"
    figure.savefig(figure_path)
    plt.close(figure)

    report = [
        "# Auditoria del recorte del campo ecografico", "",
        "El recorte es determinista y no modifica las imagenes originales. Se ignoran la "
        "cabecera y los bordes laterales, se conserva el mayor componente ecografico y "
        "se usa la imagen completa como fallback si la geometria no es plausible.", "",
        "## Resumen", "", table_markdown(summary), "",
        f"- Detalle por imagen: {audit_path}",
        f"- Control visual: {figure_path}",
    ]
    report_path = BINARY_REPORTS_ROOT / "06_fan_crop_audit.md"
    report_path.write_text("\n".join(report), encoding="utf-8")
    print(summary.to_string(index=False))
    print(f"Reporte: {report_path}")
    print(f"Figura: {figure_path}")


if __name__ == "__main__":
    main()