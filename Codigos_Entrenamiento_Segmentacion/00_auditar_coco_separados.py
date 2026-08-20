"""Audita los datasets COCO separados: ROI, Higado y LA.

El script no modifica datos originales. Genera:
- outputs/segmentation_training/reports/coco_separated_audit.csv
- outputs/segmentation_training/reports/coco_separated_audit_summary.md
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
from PIL import Image

from config_segmentation import (
    DATASET_ROOTS,
    IMAGE_EXTENSIONS,
    REPORTS_ROOT,
    ROBOFLOW_LONGITUDINAL_ROOT,
    SPLITS,
    TARGET_CLASSES,
    annotations_path,
    ensure_output_dirs,
)


def load_json(path: Path) -> dict:
    """Carga un JSON COCO."""
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def normalize_class_name(value: str) -> str:
    """Normaliza nombres para comparacion tolerante."""
    return str(value).strip().lower().replace("í", "i").replace("ı", "i")


def image_size_if_readable(path: Path) -> tuple[int | None, int | None, bool]:
    """Intenta leer dimensiones de una imagen sin modificarla."""
    try:
        with Image.open(path) as image:
            return int(image.width), int(image.height), True
    except Exception:
        return None, None, False


def audit_split(class_name: str, split: str) -> dict:
    """Audita un split de un dataset COCO separado."""
    dataset_root = DATASET_ROOTS[class_name]
    split_dir = dataset_root / split
    json_path = annotations_path(class_name, split)

    row = {
        "dataset_class_expected": class_name,
        "dataset_root": str(dataset_root),
        "split": split,
        "split_dir_exists": split_dir.exists(),
        "annotations_json": str(json_path),
        "annotations_json_exists": json_path.exists(),
        "image_count_coco": 0,
        "image_files_in_folder": 0,
        "missing_image_files": 0,
        "unreadable_image_files": 0,
        "annotation_count_total": 0,
        "annotation_count_target": 0,
        "annotation_count_other": 0,
        "images_with_target_annotation": 0,
        "images_without_target_annotation": 0,
        "category_names": "",
        "detected_target_category": "",
        "single_target_class_ok": False,
        "has_only_expected_annotation_class": False,
        "width_min": None,
        "width_max": None,
        "height_min": None,
        "height_max": None,
        "status": "pending",
        "notes": "",
    }
    notes = []

    if not split_dir.exists():
        row["status"] = "error"
        row["notes"] = "no_existe_split_dir"
        return row

    image_files = [path for path in split_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS]
    row["image_files_in_folder"] = len(image_files)

    if not json_path.exists():
        row["status"] = "error"
        row["notes"] = "no_existe_annotations_json"
        return row

    coco = load_json(json_path)
    images = coco.get("images", [])
    annotations = coco.get("annotations", [])
    categories = coco.get("categories", [])

    category_by_id = {category.get("id"): str(category.get("name", "")) for category in categories}
    normalized_expected = normalize_class_name(class_name)
    target_category_ids = [
        category_id for category_id, name in category_by_id.items() if normalize_class_name(name) == normalized_expected
    ]

    row["image_count_coco"] = len(images)
    row["annotation_count_total"] = len(annotations)
    row["category_names"] = ";".join(category_by_id.values())
    row["detected_target_category"] = ";".join(category_by_id[category_id] for category_id in target_category_ids)
    row["single_target_class_ok"] = len(target_category_ids) == 1

    if len(target_category_ids) != 1:
        notes.append(f"categoria_objetivo_no_unica:{target_category_ids}")

    target_id_set = set(target_category_ids)
    annotation_counts_by_category = Counter(annotation.get("category_id") for annotation in annotations)
    row["annotation_count_target"] = int(sum(annotation_counts_by_category[category_id] for category_id in target_id_set))
    row["annotation_count_other"] = int(
        sum(count for category_id, count in annotation_counts_by_category.items() if category_id not in target_id_set)
    )
    row["has_only_expected_annotation_class"] = row["annotation_count_other"] == 0 and row["single_target_class_ok"]

    image_ids_with_target = {
        annotation.get("image_id") for annotation in annotations if annotation.get("category_id") in target_id_set
    }
    row["images_with_target_annotation"] = len(image_ids_with_target)
    row["images_without_target_annotation"] = max(0, len(images) - len(image_ids_with_target))

    filename_set = {path.name for path in image_files}
    widths = []
    heights = []
    missing = 0
    unreadable = 0
    for image_info in images:
        filename = image_info.get("file_name", "")
        image_path = split_dir / filename
        if filename not in filename_set and not image_path.exists():
            missing += 1
            continue
        width, height, readable = image_size_if_readable(image_path)
        if not readable:
            unreadable += 1
            continue
        widths.append(width)
        heights.append(height)

    row["missing_image_files"] = missing
    row["unreadable_image_files"] = unreadable
    if widths:
        row["width_min"] = min(widths)
        row["width_max"] = max(widths)
    if heights:
        row["height_min"] = min(heights)
        row["height_max"] = max(heights)

    if row["image_count_coco"] != row["image_files_in_folder"]:
        notes.append("conteo_imagenes_coco_vs_carpeta_distinto")
    if missing > 0:
        notes.append("faltan_imagenes_referenciadas")
    if unreadable > 0:
        notes.append("imagenes_no_legibles")
    if row["annotation_count_target"] == 0:
        notes.append("sin_anotaciones_objetivo")
    if row["annotation_count_other"] > 0:
        notes.append("hay_anotaciones_de_otras_clases")

    row["status"] = "ok" if not notes else "warning"
    row["notes"] = ";".join(notes)
    return row


def audit_zip_status() -> list[dict]:
    """Reporta ZIPs presentes para dejar claro que se priorizan carpetas."""
    rows = []
    for class_name in TARGET_CLASSES + ["V2"]:
        zip_path = ROBOFLOW_LONGITUDINAL_ROOT / f"{class_name}_COCO.zip"
        folder_path = ROBOFLOW_LONGITUDINAL_ROOT / f"{class_name}_COCO"
        rows.append(
            {
                "dataset": f"{class_name}_COCO",
                "folder_exists": folder_path.exists(),
                "zip_exists": zip_path.exists(),
                "zip_path": str(zip_path) if zip_path.exists() else "",
                "priority": "folder" if folder_path.exists() else "zip_or_missing",
            }
        )
    return rows


def write_markdown(audit_df: pd.DataFrame, zip_rows: list[dict]) -> Path:
    """Escribe un reporte Markdown legible para tesis."""
    output_path = REPORTS_ROOT / "coco_separated_audit_summary.md"
    lines = [
        "# Auditoria de datasets COCO separados",
        "",
        "Esta auditoria verifica los datasets exportados desde Roboflow para entrenar ROI, Higado y LA por separado.",
        "",
        "## Regla de uso",
        "",
        "Si existe carpeta descomprimida y archivo ZIP equivalente, se prioriza la carpeta descomprimida. Los ZIP se conservan como respaldo.",
        "",
        "## ZIPs y carpetas detectadas",
        "",
        "| Dataset | Carpeta | ZIP | Prioridad |",
        "| --- | --- | --- | --- |",
    ]
    for row in zip_rows:
        lines.append(
            f"| {row['dataset']} | {row['folder_exists']} | {row['zip_exists']} | {row['priority']} |"
        )

    lines.extend(
        [
            "",
            "## Resumen por dataset y split",
            "",
            "| Clase | Split | Imagenes COCO | Imagenes carpeta | Anotaciones objetivo | Imagenes con anotacion | Otras anotaciones | Estado | Notas |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |",
        ]
    )
    for _, row in audit_df.iterrows():
        lines.append(
            "| "
            f"{row['dataset_class_expected']} | {row['split']} | {row['image_count_coco']} | "
            f"{row['image_files_in_folder']} | {row['annotation_count_target']} | "
            f"{row['images_with_target_annotation']} | {row['annotation_count_other']} | "
            f"{row['status']} | {row['notes']} |"
        )

    problems = audit_df[audit_df["status"] != "ok"]
    lines.extend(["", "## Diagnostico", ""])
    if problems.empty:
        lines.append("No se detectaron errores bloqueantes. Los datasets separados estan listos para preparar entrenamiento local.")
    else:
        lines.append("Se detectaron advertencias o errores que deben revisarse antes de entrenar:")
        lines.append("")
        for _, row in problems.iterrows():
            lines.append(f"- {row['dataset_class_expected']} {row['split']}: {row['notes']}")

    lines.extend(
        [
            "",
            "## Comando recomendado para piloto",
            "",
            "Despues de instalar dependencias de entrenamiento, iniciar con U-Net ROI:",
            "",
            "```powershell",
            'python ".\\Codigos_Entrenamiento_Segmentacion\\02_entrenar_unet.py" --class_name ROI --epochs 5 --batch_size 2',
            "```",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


def main() -> None:
    """Ejecuta auditoria de ROI_COCO, Higado_COCO y LA_COCO."""
    ensure_output_dirs()

    rows = []
    for class_name in TARGET_CLASSES:
        for split in SPLITS:
            rows.append(audit_split(class_name, split))

    audit_df = pd.DataFrame(rows)
    csv_path = REPORTS_ROOT / "coco_separated_audit.csv"
    audit_df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    zip_rows = audit_zip_status()
    zip_csv_path = REPORTS_ROOT / "coco_zip_folder_status.csv"
    pd.DataFrame(zip_rows).to_csv(zip_csv_path, index=False, encoding="utf-8-sig")

    markdown_path = write_markdown(audit_df, zip_rows)

    print("Auditoria COCO separados")
    print("=" * 72)
    print(audit_df[
        [
            "dataset_class_expected",
            "split",
            "image_count_coco",
            "image_files_in_folder",
            "annotation_count_target",
            "images_with_target_annotation",
            "annotation_count_other",
            "category_names",
            "status",
            "notes",
        ]
    ].to_string(index=False))
    print("=" * 72)
    print(f"CSV auditoria: {csv_path}")
    print(f"CSV ZIP/carpetas: {zip_csv_path}")
    print(f"Reporte Markdown: {markdown_path}")

    if (audit_df["status"] == "error").any():
        raise SystemExit("Hay errores bloqueantes en la auditoria. Revisar reporte antes de entrenar.")


if __name__ == "__main__":
    main()
