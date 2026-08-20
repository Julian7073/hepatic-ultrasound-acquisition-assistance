"""Compara U-Net sin augmentation y augmentation x4."""
import argparse
import subprocess
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument("--class_name", choices=("ROI", "Higado", "LA"), default="Higado")
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--execute", action="store_true")
args = parser.parse_args()
for augmentation in ("none", "x4"):
    name = f"unet_{args.class_name.lower()}_{augmentation}_{args.epochs}ep"
    command = [
        sys.executable, str(ROOT / "run_segmentation_experiment.py"),
        "--class_name", args.class_name, "--architecture", "unet",
        "--epochs", str(args.epochs), "--batch_size", "2", "--image_size", "512",
        "--resize_mode", "full_resize", "--augmentation", augmentation,
        "--pretrained", "false", "--split_strategy", "group_video", "--experiment_name", name,
        "--run_test", "true", "--run_benchmark", "true", "--save_overlays",
    ]
    print(" ".join(command))
    if args.execute:
        subprocess.run(command, check=True)
