"""Prepara el experimento U-Net/LA con x4 solo para imagenes positivas."""

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=50)
parser.add_argument("--execute", action="store_true")
args = parser.parse_args()

command = [
    sys.executable,
    str(ROOT / "run_segmentation_experiment.py"),
    "--class_name", "LA",
    "--architecture", "unet",
    "--epochs", str(args.epochs),
    "--batch_size", "2",
    "--image_size", "512",
    "--resize_mode", "full_resize",
    "--augmentation", "positive_x4",
    "--sampling_strategy", "natural",
    "--pretrained", "false",
    "--checkpoint_metric", "combined_la_score",
    "--split_strategy", "group_video",
    "--early_stopping_patience", "15",
    "--checkpoint_min_delta", "0.0001",
    "--experiment_name", f"unet_la_positive_x4_{args.epochs}ep_group_video_natural",
    "--run_test", "true",
    "--run_benchmark", "true",
    "--save_overlays",
]
print(" ".join(command))
if args.execute:
    subprocess.run(command, check=True)
