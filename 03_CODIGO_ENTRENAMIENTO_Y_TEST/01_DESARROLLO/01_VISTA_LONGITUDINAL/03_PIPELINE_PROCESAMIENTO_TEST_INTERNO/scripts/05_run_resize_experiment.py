"""Compara resize completo, recorte ROI 128 y letterbox."""
import argparse
import subprocess
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument("--class_name", choices=("ROI", "Higado", "LA"), default="LA")
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--execute", action="store_true")
args = parser.parse_args()
settings = (("full_resize", 512), ("roi_crop_resize", 128), ("original_or_padding", 512))
for resize_mode, image_size in settings:
    name = f"unet_{args.class_name.lower()}_{resize_mode}_{image_size}_{args.epochs}ep"
    command = [
        sys.executable, str(ROOT / "run_segmentation_experiment.py"),
        "--class_name", args.class_name, "--architecture", "unet",
        "--epochs", str(args.epochs), "--batch_size", "2", "--image_size", str(image_size),
        "--resize_mode", resize_mode, "--augmentation", "none",
        "--pretrained", "false", "--split_strategy", "group_video", "--experiment_name", name,
        "--run_test", "true", "--run_benchmark", "true", "--save_overlays",
    ]
    print(" ".join(command))
    if args.execute:
        subprocess.run(command, check=True)
