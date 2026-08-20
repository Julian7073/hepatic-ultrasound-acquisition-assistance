"""Genera un anexo visual con casos representativos de vista longitudinal.

Este script no modifica imagenes originales. Solo crea:
- un CSV con la interpretacion visual acordada;
- una figura tipo anexo para incluir en el reporte tecnico.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from config import FIGURES_ROOT, REPORTS_ROOT, ensure_output_dirs


DATASET_ROOT = Path(os.environ.get("THESIS_PROJECT_ROOT", Path(__file__).resolve().parents[1])) / "Dataset_Frames_Processed"

VISUAL_CASES = [
    {
        "case_id": "P001_clear_LA_visible",
        "patient": "P001",
        "view": "longitudinal",
        "quality": "clear",
        "visual_category": "longitudinal bien adquirida",
        "interpretation": "LA visible, con lumen anecoico y borde hiperecogenico claro. Caso aceptable.",
        "doubt_criterion": "No aplica; el borde blanco permite validar el lumen.",
        "image_path": DATASET_ROOT / "P001" / "longitudinal" / "clear" / "P001_longitudinal_clear_20260501_122220_PACIENTE 001_202605020003-converted_frame_00008.png",
    },
    {
        "case_id": "P002_clear_LA_no_confiable",
        "patient": "P002",
        "view": "longitudinal",
        "quality": "clear",
        "visual_category": "longitudinal nominal, LA no claramente visible",
        "interpretation": "La imagen pertenece nominalmente a longitudinal, pero LA no se observa con claridad suficiente.",
        "doubt_criterion": "Si existe un circulo oscuro sin borde blanco claro, se considera dudoso y no aceptable.",
        "image_path": DATASET_ROOT / "P002" / "longitudinal" / "clear" / "P002_longitudinal_clear_20260501_134001_PACIENTE 002_202605020003-converted_frame_00100.png",
    },
    {
        "case_id": "P003_clear_LA_no_confiable",
        "patient": "P003",
        "view": "longitudinal",
        "quality": "clear",
        "visual_category": "longitudinal nominal, LA no claramente visible",
        "interpretation": "La vista es longitudinal, pero el lumen anecoico no queda claramente validado por sus bordes.",
        "doubt_criterion": "Circulo negro sin pared hiperecogenica definida: caso dudoso para LA.",
        "image_path": DATASET_ROOT / "P003" / "longitudinal" / "clear" / "P003_longitudinal_clear_20260501_135548_PACIENTE 003_202605020003-converted_frame_00070.png",
    },
    {
        "case_id": "P003_blurry_rechazado",
        "patient": "P003",
        "view": "longitudinal",
        "quality": "blurry",
        "visual_category": "blurry rechazado",
        "interpretation": "Imagen borrosa; no es defendible validar LA para analisis posterior.",
        "doubt_criterion": "Baja definicion global y falta de borde anatomico claro.",
        "image_path": DATASET_ROOT / "P003" / "longitudinal" / "blurry" / "P003_longitudinal_blurry_20260501_135548_PACIENTE 003_202605020005-converted_frame_00049.png",
    },
]


def save_cases_csv(cases: list[dict]) -> Path:
    """Guarda una tabla reproducible con los casos visuales definidos."""
    rows = []
    for case in cases:
        row = case.copy()
        row["image_path"] = str(row["image_path"])
        row["file_exists"] = Path(row["image_path"]).exists()
        rows.append(row)

    df = pd.DataFrame(rows)
    output_path = REPORTS_ROOT / "longitudinal_visual_case_examples.csv"
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    return output_path


def _load_font(size: int) -> ImageFont.ImageFont:
    """Carga una fuente simple compatible con Windows y fallback interno."""
    candidates = [
        Path(r"C:\Windows\Fonts\arial.ttf"),
        Path(r"C:\Windows\Fonts\calibri.ttf"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default()


def save_visual_grid(cases: list[dict]) -> Path:
    """Crea una figura 2x2 con los casos representativos usando Pillow."""
    output_path = FIGURES_ROOT / "casos_representativos_longitudinal.png"
    thumb_width = 520
    thumb_height = 390
    label_height = 92
    margin = 26
    title_height = 64
    canvas_width = margin * 3 + thumb_width * 2
    canvas_height = title_height + margin * 3 + (thumb_height + label_height) * 2

    canvas = Image.new("RGB", (canvas_width, canvas_height), "white")
    draw = ImageDraw.Draw(canvas)
    title_font = _load_font(22)
    label_font = _load_font(16)
    small_font = _load_font(12)

    title = "Casos representativos para interpretacion visual de la vista longitudinal"
    draw.text((margin, 18), title, fill=(20, 20, 20), font=title_font)

    for index, case in enumerate(cases):
        row = index // 2
        col = index % 2
        x = margin + col * (thumb_width + margin)
        y = title_height + margin + row * (thumb_height + label_height + margin)
        image_path = case["image_path"]

        if not image_path.exists():
            draw.rectangle((x, y, x + thumb_width, y + thumb_height), fill=(235, 235, 235), outline=(180, 180, 180))
            draw.text((x + 20, y + 170), f"No encontrada:\n{image_path.name}", fill=(120, 0, 0), font=small_font)
            continue

        image = Image.open(image_path).convert("RGB")
        image.thumbnail((thumb_width, thumb_height), Image.Resampling.LANCZOS)
        image_x = x + (thumb_width - image.width) // 2
        image_y = y + (thumb_height - image.height) // 2
        canvas.paste(image, (image_x, image_y))
        draw.rectangle((x, y, x + thumb_width, y + thumb_height), outline=(80, 80, 80), width=1)

        label_y = y + thumb_height + 8
        draw.text((x, label_y), f"{case['patient']} | {case['quality']}", fill=(20, 20, 20), font=label_font)
        draw.text((x, label_y + 24), case["visual_category"], fill=(55, 55, 55), font=small_font)
        draw.text((x, label_y + 44), case["interpretation"][:86], fill=(80, 80, 80), font=small_font)
        draw.text((x, label_y + 62), case["doubt_criterion"][:86], fill=(110, 70, 0), font=small_font)

    canvas.save(output_path)
    return output_path


def main() -> None:
    """Ejecuta la generacion del anexo visual."""
    ensure_output_dirs()
    csv_path = save_cases_csv(VISUAL_CASES)
    figure_path = save_visual_grid(VISUAL_CASES)

    missing = [case["image_path"] for case in VISUAL_CASES if not case["image_path"].exists()]
    if missing:
        print("Advertencia: algunas imagenes no existen:")
        for path in missing:
            print(f"- {path}")

    print(f"CSV generado: {csv_path}")
    print(f"Figura generada: {figure_path}")


if __name__ == "__main__":
    main()
