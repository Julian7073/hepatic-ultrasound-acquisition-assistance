"""Configuracion central para entrenamiento local de segmentacion.

Esta fase usa los COCO separados exportados desde Roboflow como fuente limpia
de anotaciones. Resize y augmentations se aplican localmente durante training.
"""

from __future__ import annotations

import os
from pathlib import Path


TESIS_ROOT = Path(os.environ.get("THESIS_PROJECT_ROOT", Path(__file__).resolve().parents[1]))

ROBOFLOW_LONGITUDINAL_ROOT = TESIS_ROOT / "Dataset_Roboflow_Longitudinal"
DATASET_ROOTS = {
    "ROI": ROBOFLOW_LONGITUDINAL_ROOT / "ROI_COCO",
    "Higado": ROBOFLOW_LONGITUDINAL_ROOT / "Higado_COCO",
    "LA": ROBOFLOW_LONGITUDINAL_ROOT / "LA_COCO",
}

OUTPUT_ROOT = TESIS_ROOT / "outputs" / "segmentation_training"
CHECKPOINTS_ROOT = OUTPUT_ROOT / "checkpoints"
FINAL_MODELS_ROOT = OUTPUT_ROOT / "final_models"
METRICS_ROOT = OUTPUT_ROOT / "metrics"
FIGURES_ROOT = OUTPUT_ROOT / "figures"
OVERLAYS_ROOT = OUTPUT_ROOT / "overlays"
REPORTS_ROOT = OUTPUT_ROOT / "reports"
LOGS_ROOT = OUTPUT_ROOT / "logs"

SPLITS = ["train", "valid", "test"]
TARGET_CLASSES = ["ROI", "Higado", "LA"]
ARCHITECTURES = ["unet", "deeplabv3", "segformer"]

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

DEFAULT_IMAGE_SIZE = 512
DEFAULT_BATCH_SIZE = 4
DEFAULT_EPOCHS = 30
DEFAULT_LR = 1e-4
DEFAULT_NUM_WORKERS = 0  # Windows-friendly.
DEFAULT_SEED = 42


def ensure_output_dirs() -> None:
    """Crea carpetas de salida sin borrar resultados previos."""
    for path in [
        OUTPUT_ROOT,
        CHECKPOINTS_ROOT,
        FINAL_MODELS_ROOT,
        METRICS_ROOT,
        FIGURES_ROOT,
        OVERLAYS_ROOT,
        REPORTS_ROOT,
        LOGS_ROOT,
    ]:
        path.mkdir(parents=True, exist_ok=True)

    for class_name in TARGET_CLASSES:
        for root in [CHECKPOINTS_ROOT, METRICS_ROOT, FIGURES_ROOT, OVERLAYS_ROOT, LOGS_ROOT]:
            (root / class_name).mkdir(parents=True, exist_ok=True)


def annotations_path(class_name: str, split: str) -> Path:
    """Ruta del JSON COCO de una clase y split."""
    return DATASET_ROOTS[class_name] / split / "_annotations.coco.json"
