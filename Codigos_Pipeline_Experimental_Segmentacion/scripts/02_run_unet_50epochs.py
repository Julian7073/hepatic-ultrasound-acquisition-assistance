"""Muestra comandos U-Net de 50 epocas; use --execute solo tras autorizarlos."""
import argparse
import subprocess
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument("--execute", action="store_true")
args = parser.parse_args()
for class_name in ("ROI", "Higado", "LA"):
    command = [
        sys.executable, str(ROOT / "run_segmentation_experiment.py"),
        "--class_name", class_name, "--architecture", "unet", "--epochs", "50",
        "--batch_size", "2", "--image_size", "512", "--resize_mode", "full_resize",
        "--augmentation", "none", "--pretrained", "false", "--split_strategy", "group_video", "--checkpoint_metric", "auto",
        "--experiment_name", f"unet_{class_name.lower()}_50ep_base",
        "--run_test", "true", "--run_benchmark", "true", "--save_overlays",
    ]
    print(" ".join(command))
    if args.execute:
        subprocess.run(command, check=True)
