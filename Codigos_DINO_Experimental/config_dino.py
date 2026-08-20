"""Configuracion central de la fase experimental DINOv2."""

import os
from pathlib import Path


TESIS_ROOT = Path(os.environ.get("THESIS_PROJECT_ROOT", Path(__file__).resolve().parents[1]))
PIPELINE_ROOT = TESIS_ROOT / "Codigos_DINO_Experimental"
FRAMES_ROOT = TESIS_ROOT / "Dataset_Frames_Processed"
OUTPUT_ROOT = TESIS_ROOT / "outputs" / "dino_experimental"
REPORTS_ROOT = OUTPUT_ROOT / "reports"
FIGURES_ROOT = OUTPUT_ROOT / "figures"
EMBEDDINGS_ROOT = OUTPUT_ROOT / "embeddings"
MODELS_ROOT = OUTPUT_ROOT / "models"
BINARY_ROOT = OUTPUT_ROOT / "binary_improvement"
BINARY_EMBEDDINGS_ROOT = BINARY_ROOT / "embeddings"
BINARY_REPORTS_ROOT = BINARY_ROOT / "reports"
BINARY_FIGURES_ROOT = BINARY_ROOT / "figures"
BINARY_MODELS_ROOT = TESIS_ROOT / "models" / "classifiers"

PATIENTS_DEVELOPMENT = ("P001", "P002", "P003")
PATIENT_EXTERNAL = "P005"
VIEWS = ("transversal", "oblicua", "hepatorrenal")
QUALITIES = ("clear", "medium", "blurry")
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
DINOV2_MODEL_ID = "facebook/dinov2-small"
DINOV2_MODEL_IDS = {
    "small": "facebook/dinov2-small",
    "base": "facebook/dinov2-base",
}

def ensure_directories() -> None:
    for path in (
        OUTPUT_ROOT,
        REPORTS_ROOT,
        FIGURES_ROOT,
        EMBEDDINGS_ROOT,
        MODELS_ROOT,
        BINARY_ROOT,
        BINARY_EMBEDDINGS_ROOT,
        BINARY_REPORTS_ROOT,
        BINARY_FIGURES_ROOT,
        BINARY_MODELS_ROOT,
    ):
        path.mkdir(parents=True, exist_ok=True)
