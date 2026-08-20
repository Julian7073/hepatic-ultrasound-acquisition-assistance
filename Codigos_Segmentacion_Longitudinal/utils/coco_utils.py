"""Lectura y validacion basica de archivos COCO Segmentation."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from config import CLASS_ALIASES, EXPECTED_CLASSES, SPLITS, annotations_path


def canonical_class_name(name: str) -> str | None:
    """Normaliza nombres de clases relevantes; ignora clases no esperadas."""
    return CLASS_ALIASES.get(str(name).strip(), CLASS_ALIASES.get(str(name).strip().lower()))


def load_coco_json(path: Path) -> dict:
    """Carga un JSON COCO."""
    if not path.exists():
        raise FileNotFoundError(f"No existe el archivo COCO: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def category_id_to_name(coco: dict) -> dict[int, str]:
    """Mapea category_id a clase canonica ROI/Higado/LA."""
    mapping: dict[int, str] = {}
    for category in coco.get("categories", []):
        canonical = canonical_class_name(category.get("name", ""))
        if canonical in EXPECTED_CLASSES:
            mapping[int(category["id"])] = canonical
    return mapping


def validate_expected_classes(coco: dict) -> tuple[bool, list[str]]:
    """Verifica que existan ROI, Higado y LA."""
    found = set(category_id_to_name(coco).values())
    missing = sorted(set(EXPECTED_CLASSES) - found)
    return len(missing) == 0, missing


def annotations_by_image_and_class(coco: dict) -> dict[int, dict[str, list[dict]]]:
    """Agrupa anotaciones por imagen y clase canonica."""
    id_to_class = category_id_to_name(coco)
    grouped: dict[int, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for annotation in coco.get("annotations", []):
        class_name = id_to_class.get(int(annotation.get("category_id", -1)))
        if class_name is None:
            continue
        grouped[int(annotation["image_id"])][class_name].append(annotation)
    return grouped


def image_records(coco: dict) -> list[dict]:
    """Devuelve imagenes COCO ordenadas por nombre."""
    return sorted(coco.get("images", []), key=lambda item: item.get("file_name", ""))


def split_image_path(coco_root: Path, split: str, file_name: str) -> Path:
    """Ruta absoluta de una imagen dentro de un split."""
    return coco_root / split / file_name


def available_splits() -> list[str]:
    """Splits configurados que tienen JSON COCO."""
    return [split for split in SPLITS if annotations_path(split).exists()]
