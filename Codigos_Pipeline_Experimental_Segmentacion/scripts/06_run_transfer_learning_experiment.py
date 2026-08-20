"""Compara inicializacion aleatoria y pesos preentrenados."""
import argparse
import subprocess
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument("--architecture", choices=("unet", "deeplabv3", "segformer"), default="unet")
parser.add_argument("--class_name", choices=("ROI", "Higado", "LA"), default="ROI")
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--execute", action="store_true")
args = parser.parse_args()
for pretrained in ("false", "true"):
    name = f"{args.architecture}_{args.class_name.lower()}_pretrained_{pretrained}_{args.epochs}ep"
    command = [
        sys.executable, str(ROOT / "run_segmentation_experiment.py"),
        "--class_name", args.class_name, "--architecture", args.architecture,
        "--epochs", str(args.epochs), "--batch_size", "2", "--image_size", "512",
        "--resize_mode", "full_resize", "--augmentation", "none",
        "--pretrained", pretrained, "--split_strategy", "group_video", "--experiment_name", name,
        "--run_test", "true", "--run_benchmark", "true", "--save_overlays",
    ]
    print(" ".join(command))
    if args.execute:
        subprocess.run(command, check=True)
