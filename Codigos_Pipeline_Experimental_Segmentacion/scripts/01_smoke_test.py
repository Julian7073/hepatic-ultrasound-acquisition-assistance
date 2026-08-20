"""Lanza el smoke test oficial U-Net/ROI de dos epocas."""
import subprocess
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
command = [
    sys.executable, str(ROOT / "run_segmentation_experiment.py"),
    "--class_name", "ROI", "--architecture", "unet", "--epochs", "2",
    "--batch_size", "2", "--image_size", "128", "--resize_mode", "full_resize",
    "--augmentation", "none", "--pretrained", "false",
    "--experiment_name", "smoke_unet_roi_2ep",
    "--run_test", "true", "--run_benchmark", "true", "--save_overlays",
]
raise SystemExit(subprocess.call(command))
