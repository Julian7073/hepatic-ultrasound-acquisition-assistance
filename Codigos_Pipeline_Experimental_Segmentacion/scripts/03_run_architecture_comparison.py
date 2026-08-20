"""Muestra la matriz DeepLabV3+/SegFormer; use --execute para entrenar."""
import argparse
import subprocess
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument("--execute", action="store_true")
args = parser.parse_args()
for architecture in ("deeplabv3", "segformer"):
    for class_name in ("ROI", "Higado", "LA"):
        sampling_strategy = "balanced_la" if class_name == "LA" else "natural"
        patience = "15" if class_name == "LA" else "10"
        command = [
            sys.executable, str(ROOT / "run_segmentation_experiment.py"),
            "--class_name", class_name, "--architecture", architecture, "--epochs", "50",
            "--batch_size", "2", "--image_size", "512", "--resize_mode", "full_resize",
            "--augmentation", "none", "--sampling_strategy", sampling_strategy,
            "--pretrained", "false", "--split_strategy", "group_video", "--checkpoint_metric", "auto",
            "--early_stopping_patience", patience, "--checkpoint_min_delta", "0.0001",
            "--experiment_name", f"{architecture}_{class_name.lower()}_50ep_base",
            "--run_test", "true", "--run_benchmark", "true", "--save_overlays",
        ]
        print(" ".join(command))
        if args.execute:
            subprocess.run(command, check=True)
