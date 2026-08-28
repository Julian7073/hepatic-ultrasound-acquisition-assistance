"""Configuracion central del pipeline experimental de segmentacion."""

import os
from pathlib import Path


TESIS_ROOT = Path(os.environ.get("THESIS_PROJECT_ROOT", Path(__file__).resolve().parents[1]))
PIPELINE_ROOT = TESIS_ROOT / "Codigos_Pipeline_Experimental_Segmentacion"
DATASETS_ROOT = TESIS_ROOT / "Dataset_Roboflow_Longitudinal"
OUTPUT_ROOT = TESIS_ROOT / "outputs" / "experimental_segmentation_pipeline"

DATASET_ROOTS = {
    "ROI": DATASETS_ROOT / "ROI_COCO",
    "Higado": DATASETS_ROOT / "Higado_COCO",
    "LA": DATASETS_ROOT / "LA_COCO",
}

EXPERIMENTS_ROOT = OUTPUT_ROOT / "experiments"
REPORTS_ROOT = OUTPUT_ROOT / "reports"
FIGURES_ROOT = OUTPUT_ROOT / "figures"
OVERLAYS_ROOT = OUTPUT_ROOT / "overlays"
FINAL_MODELS_ROOT = TESIS_ROOT / "models" / "segmentation_checkpoints"
EXTERNAL_P005_ROOT = OUTPUT_ROOT / "external_validation_p005"
VIDEO_INFERENCE_ROOT = OUTPUT_ROOT / "video_inference"
GUI_OUTPUT_ROOT = OUTPUT_ROOT / "gui_sessions"
UNIFIED_GUI_OUTPUT_ROOT = TESIS_ROOT / "outputs" / "unified_gui_sessions"
P005_FRAMES_ROOT = TESIS_ROOT / "Dataset_Frames_Processed" / "P005" / "longitudinal"

SPLITS = ("train", "valid", "test")
CLASS_NAMES = ("ROI", "Higado", "LA")
ARCHITECTURES = ("unet", "deeplabv3", "segformer")
RESIZE_MODES = ("full_resize", "roi_crop_resize", "original_or_padding")


def ensure_directories() -> None:
    """Crea solamente carpetas de salida; nunca altera datasets originales."""
    for path in (
        OUTPUT_ROOT,
        EXPERIMENTS_ROOT,
        REPORTS_ROOT,
        FIGURES_ROOT,
        OVERLAYS_ROOT,
        FINAL_MODELS_ROOT,
        EXTERNAL_P005_ROOT,
        VIDEO_INFERENCE_ROOT,
        GUI_OUTPUT_ROOT,
        UNIFIED_GUI_OUTPUT_ROOT,
    ):
        path.mkdir(parents=True, exist_ok=True)
