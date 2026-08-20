"""Configuracion central del pipeline longitudinal basado en COCO."""

import os
from pathlib import Path


TESIS_ROOT = Path(os.environ.get("THESIS_PROJECT_ROOT", Path(__file__).resolve().parents[1]))

ROBOFLOW_COCO_ROOT = TESIS_ROOT / "Dataset_Roboflow_Longitudinal" / "V2_COCO"
FRAMES_PROCESSED_ROOT = TESIS_ROOT / "Dataset_Frames_Processed"

OUTPUTS_ROOT = TESIS_ROOT / "outputs"
MASKS_ROOT = OUTPUTS_ROOT / "masks"
METRICS_ROOT = OUTPUTS_ROOT / "metrics"
REPORTS_ROOT = OUTPUTS_ROOT / "reports"
FIGURES_ROOT = OUTPUTS_ROOT / "figures"
QC_MASKS_ROOT = OUTPUTS_ROOT / "qc_masks"
LOGS_ROOT = OUTPUTS_ROOT / "logs"

SPLITS = ["train", "valid", "test"]
EXPECTED_CLASSES = ["ROI", "Higado", "LA"]

CLASS_ALIASES = {
    "roi": "ROI",
    "ROI": "ROI",
    "higado": "Higado",
    "Higado": "Higado",
    "la": "LA",
    "LA": "LA",
}

CLASS_COLORS_RGB = {
    "ROI": (255, 210, 0),
    "Higado": (0, 220, 70),
    "LA": (255, 60, 60),
}

DEVELOPMENT_PATIENTS = ["P001", "P002", "P003"]
EXTERNAL_VALIDATION_PATIENTS = ["P005"]
EXCLUDED_PATIENTS = ["P004"]

# Parametros iniciales simples; se ajustan con 05_definir_umbral_aceptabilidad.py.
LA_DILATION_KERNEL_SIZE = 15
MIN_LA_AREA_PX = 20
GLCM_LEVELS = 32
GLCM_OFFSETS = [(0, 1), (1, 0), (1, 1), (-1, 1)]


def ensure_output_dirs() -> None:
    """Crea carpetas de salida sin tocar datos originales."""
    for path in [
        OUTPUTS_ROOT,
        MASKS_ROOT,
        METRICS_ROOT,
        REPORTS_ROOT,
        FIGURES_ROOT,
        QC_MASKS_ROOT,
        LOGS_ROOT,
    ]:
        path.mkdir(parents=True, exist_ok=True)


def annotations_path(split: str) -> Path:
    """Ruta esperada del JSON COCO de un split."""
    return ROBOFLOW_COCO_ROOT / split / "_annotations.coco.json"
