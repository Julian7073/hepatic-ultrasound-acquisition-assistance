"""Funciones para extraer metadatos desde nombres de archivo."""

from __future__ import annotations

import re
from pathlib import Path


VALID_VIEWS = ["transversal", "longitudinal", "oblicua", "hepatorrenal"]
VALID_QUALITIES = ["clear", "medium", "blurry"]


def normalize_text(value: str) -> str:
    """Normaliza texto para busquedas robustas."""
    value = str(value).strip().lower()
    replacements = {
        "á": "a",
        "é": "e",
        "í": "i",
        "ó": "o",
        "ú": "u",
        ".": " ",
        "_": " ",
        "-": " ",
    }
    for old, new in replacements.items():
        value = value.replace(old, new)
    return " ".join(value.split())


def extract_patient(value: str) -> str:
    """Extrae paciente como P001, P002, etc."""
    text = str(value)
    match = re.search(r"\bP0*(\d+)\b|PACIENTE[\s_-]*0*(\d+)", text, flags=re.IGNORECASE)
    if not match:
        return "unknown"
    number = match.group(1) or match.group(2)
    return f"P{int(number):03d}"


def extract_view(value: str) -> str:
    """Extrae vista normalizada desde un nombre o ruta."""
    text = normalize_text(value)
    for view in VALID_VIEWS:
        if view in text:
            return view
    return "unknown"


def extract_quality(value: str) -> str:
    """Extrae calidad normalizada desde un nombre o ruta."""
    text = normalize_text(value)
    for quality in VALID_QUALITIES:
        if quality in text:
            return quality
    return "unknown"


def metadata_from_filename(filename: str) -> dict[str, str]:
    """Devuelve paciente, vista y calidad desde el nombre de imagen."""
    return {
        "patient": extract_patient(filename),
        "view": extract_view(filename),
        "quality": extract_quality(filename),
    }


def mask_filename_for_image(filename: str) -> str:
    """Convierte nombre de imagen COCO a nombre de mascara PNG."""
    return f"{Path(filename).stem}.png"
