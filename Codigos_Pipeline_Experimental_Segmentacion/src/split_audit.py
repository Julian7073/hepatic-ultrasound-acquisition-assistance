"""Auditoria de COCO y posible fuga entre splits por paciente o video."""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

from config_experimental import CLASS_NAMES, DATASET_ROOTS, REPORTS_ROOT, SPLITS


def normalize_class_name(value: str) -> str:
    """Normaliza nombres de clases conservando las etiquetas oficiales."""
    key = str(value).strip().lower().replace("í", "i")
    aliases = {"roi": "ROI", "higado": "Higado", "la": "LA"}
    return aliases.get(key, str(value).strip())


def source_key(filename: str) -> str:
    """Elimina el hash agregado por Roboflow para recuperar el nombre fuente."""
    name = Path(filename).name
    return re.sub(r"\.rf\.[0-9a-f]+\.[^.]+$", "", name, flags=re.IGNORECASE)


def infer_metadata(filename: str) -> dict[str, str]:
    """Infiere paciente, calidad, vista y video desde el nombre real del frame."""
    key = source_key(filename)
    patient_match = re.search(r"(?:^|[_-])(P\d{3})(?:[_-]|$)", key, re.IGNORECASE)
    patient = patient_match.group(1).upper() if patient_match else "unknown"
    lower = key.lower()
    quality = next((item for item in ("clear", "medium", "blurry") if item in lower), "unknown")
    view = "longitudinal" if "longitudinal" in lower else "unknown"
    video_id = re.sub(r"[_-]frame[_-]?\d+.*$", "", key, flags=re.IGNORECASE)
    video_id = re.sub(r"_png$", "", video_id, flags=re.IGNORECASE)
    return {
        "source_key": key,
        "patient": patient,
        "quality": quality,
        "view": view,
        "video_id": video_id,
    }


def load_json(path: Path) -> dict:
    """Carga un COCO JSON sin modificarlo."""
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def audit_datasets() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Audita conteos, clases y grupos compartidos entre train/valid/test."""
    audit_rows: list[dict] = []
    groups: dict[tuple[str, str, str], dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for expected_class in CLASS_NAMES:
        root = DATASET_ROOTS[expected_class]
        for split in SPLITS:
            split_dir = root / split
            coco_path = split_dir / "_annotations.coco.json"
            if not coco_path.exists():
                audit_rows.append({
                    "class_name": expected_class, "split": split, "dataset_path": str(root),
                    "coco_exists": False, "images": 0, "annotations": 0,
                    "positive_images": 0, "empty_images": 0, "categories": "", "active_categories": "", "unused_categories": "",
                    "single_expected_class": False, "missing_image_files": 0,
                    "detected_patients": "", "detected_videos": 0,
                })
                continue

            coco = load_json(coco_path)
            images = coco.get("images", [])
            annotations = coco.get("annotations", [])
            categories = [normalize_class_name(item.get("name", "")) for item in coco.get("categories", [])]
            category_counts = Counter(item.get("category_id") for item in annotations)
            active_categories = [normalize_class_name(item.get("name", "")) for item in coco.get("categories", []) if category_counts.get(item.get("id"), 0) > 0]
            unused_categories = [normalize_class_name(item.get("name", "")) for item in coco.get("categories", []) if category_counts.get(item.get("id"), 0) == 0]
            valid_category_ids = {
                item.get("id") for item in coco.get("categories", [])
                if normalize_class_name(item.get("name", "")) == expected_class
            }
            positive_ids = {
                item.get("image_id") for item in annotations
                if item.get("category_id") in valid_category_ids
            }
            missing_files = sum(not (split_dir / item.get("file_name", "")).exists() for item in images)
            patients = set()
            videos = set()
            for image in images:
                metadata = infer_metadata(image.get("file_name", ""))
                patients.add(metadata["patient"])
                videos.add(metadata["video_id"])
                if metadata["patient"] != "unknown":
                    groups[(expected_class, "patient", metadata["patient"])][split] += 1
                if metadata["video_id"]:
                    groups[(expected_class, "video", metadata["video_id"])][split] += 1

            audit_rows.append({
                "class_name": expected_class,
                "split": split,
                "dataset_path": str(root),
                "coco_exists": True,
                "images": len(images),
                "annotations": sum(item.get("category_id") in valid_category_ids for item in annotations),
                "positive_images": len(positive_ids),
                "empty_images": len(images) - len(positive_ids),
                "categories": " | ".join(categories),
                "active_categories": " | ".join(active_categories),
                "unused_categories": " | ".join(unused_categories),
                "single_expected_class": len(active_categories) == 1 and active_categories[0] == expected_class,
                "missing_image_files": missing_files,
                "detected_patients": " | ".join(sorted(patients)),
                "detected_videos": len(videos),
            })

    leakage_rows = []
    for (class_name, group_type, group_id), split_counts in sorted(groups.items()):
        present_splits = [split for split in SPLITS if split_counts.get(split, 0) > 0]
        leakage_rows.append({
            "class_name": class_name,
            "group_type": group_type,
            "group_id": group_id,
            "train_images": split_counts.get("train", 0),
            "valid_images": split_counts.get("valid", 0),
            "test_images": split_counts.get("test", 0),
            "splits_present": " | ".join(present_splits),
            "split_count": len(present_splits),
            "possible_leakage": len(present_splits) > 1,
        })
    return pd.DataFrame(audit_rows), pd.DataFrame(leakage_rows)



def markdown_table(frame: pd.DataFrame) -> str:
    """Genera tabla Markdown sin dependencias opcionales."""
    if frame.empty:
        return "_Sin datos disponibles._"
    columns = [str(column) for column in frame.columns]
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in frame.fillna("").astype(str).itertuples(index=False, name=None):
        values = [value.replace("|", "/").replace("\n", " ") for value in row]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)

def write_audit_reports(audit_df: pd.DataFrame, leakage_df: pd.DataFrame) -> None:
    """Guarda CSV y Markdown con advertencias metodologicas explicitas."""
    REPORTS_ROOT.mkdir(parents=True, exist_ok=True)
    audit_df.to_csv(REPORTS_ROOT / "00_dataset_audit.csv", index=False, encoding="utf-8-sig")
    leakage_df.to_csv(REPORTS_ROOT / "00_split_leakage_audit.csv", index=False, encoding="utf-8-sig")

    issues = audit_df[(~audit_df["coco_exists"]) | (~audit_df["single_expected_class"]) | (audit_df["missing_image_files"] > 0)]
    with (REPORTS_ROOT / "00_dataset_audit.md").open("w", encoding="utf-8") as file:
        file.write("# Auditoria de datasets COCO separados\n\n")
        file.write("Este reporte se genera sin modificar los datasets oficiales.\n\n")
        file.write(markdown_table(audit_df))
        file.write("\n\n")
        if issues.empty:
            file.write("**Resultado:** no se detectaron JSON ausentes, clases activas inesperadas ni imagenes faltantes. Las categorias auxiliares sin anotaciones se conservan como metadato de Roboflow.\n")
        else:
            file.write("**Advertencia:** se detectaron inconsistencias que deben resolverse antes de interpretar resultados.\n")

    leaking = leakage_df[leakage_df["possible_leakage"]] if not leakage_df.empty else leakage_df
    by_type = Counter(leaking["group_type"]) if not leaking.empty else Counter()
    with (REPORTS_ROOT / "00_split_leakage_audit.md").open("w", encoding="utf-8") as file:
        file.write("# Auditoria de posible fuga entre splits\n\n")
        file.write(
            "Se considera riesgo de fuga cuando frames del mismo paciente o video aparecen en mas de un split. "
            "Esto puede inflar las metricas porque frames cercanos de un video suelen ser visualmente muy parecidos.\n\n"
        )
        unique_patients = leaking.loc[leaking["group_type"] == "patient", "group_id"].nunique()
        unique_videos = leaking.loc[leaking["group_type"] == "video", "group_id"].nunique()
        file.write(f"- Pacientes unicos compartidos: {unique_patients}\n")
        file.write(f"- Videos unicos compartidos: {unique_videos}\n")
        file.write(f"- Registros clase-paciente compartidos: {by_type.get('patient', 0)}\n")
        file.write(f"- Registros clase-video compartidos: {by_type.get('video', 0)}\n")
        file.write(f"- Filas totales con posible leakage: {len(leaking)}\n\n")
        if leaking.empty:
            file.write("No se detectaron grupos compartidos con los metadatos inferibles del nombre.\n")
        else:
            file.write("**Advertencia metodologica:** los splits COCO actuales no permiten afirmar generalizacion fuerte si comparten pacientes o videos.\n\n")
            file.write(markdown_table(leaking))
            file.write(
                "\n\nEl pipeline ofrece `--split_strategy group_video` y `--split_strategy group_patient` para experimentos controlados. "
                "Con pocos pacientes, el split por paciente puede ser inestable y debe reportarse como limitacion.\n"
            )


def run_audit() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Ejecuta y persiste la auditoria completa."""
    audit_df, leakage_df = audit_datasets()
    write_audit_reports(audit_df, leakage_df)
    return audit_df, leakage_df

