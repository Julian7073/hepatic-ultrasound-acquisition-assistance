"""Utilidades reproducibles y de registro del entorno."""

from __future__ import annotations

import json
import platform
import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch


def parse_bool(value) -> bool:
    """Convierte argumentos de terminal true/false a booleanos."""
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes", "si", "s"}:
        return True
    if normalized in {"false", "0", "no", "n"}:
        return False
    raise ValueError(f"Valor booleano invalido: {value}")


def seed_everything(seed: int) -> None:
    """Fija semillas y opciones deterministas razonables."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def unique_directory(base_path: Path) -> Path:
    """Evita sobrescribir experimentos existentes mediante sufijos numericos."""
    if not base_path.exists():
        base_path.mkdir(parents=True, exist_ok=False)
        return base_path
    index = 2
    while True:
        candidate = base_path.parent / f"{base_path.name}_v{index}"
        if not candidate.exists():
            candidate.mkdir(parents=True, exist_ok=False)
            return candidate
        index += 1


def environment_info() -> dict:
    """Registra hardware y versiones relevantes para reproducibilidad."""
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No disponible"
    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.processor() or "No reportado",
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "gpu": gpu,
    }


def save_json(data: dict, path: Path) -> None:
    """Guarda JSON legible y compatible con caracteres en espanol."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2, default=str)

