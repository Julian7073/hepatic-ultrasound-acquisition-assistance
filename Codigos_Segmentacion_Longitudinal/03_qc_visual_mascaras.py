"""Genera overlays de control visual para revisar mascaras COCO."""

from __future__ import annotations

import argparse
import random

from config import CLASS_COLORS_RGB, EXPECTED_CLASSES, MASKS_ROOT, QC_MASKS_ROOT, ROBOFLOW_COCO_ROOT, SPLITS, annotations_path, ensure_output_dirs
from utils.coco_utils import load_coco_json, split_image_path
from utils.mask_utils import load_binary_mask, overlay_masks
from utils.metadata_utils import mask_filename_for_image


def parse_args() -> argparse.Namespace:
    """Argumentos de terminal."""
    parser = argparse.ArgumentParser(description="Genera overlays aleatorios de mascaras.")
    parser.add_argument("--samples-per-split", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def process_split(split: str, samples_per_split: int, rng: random.Random) -> int:
    """Genera overlays de un split."""
    json_path = annotations_path(split)
    if not json_path.exists():
        print(f"ADVERTENCIA: no existe {json_path}")
        return 0

    coco = load_coco_json(json_path)
    images = list(coco.get("images", []))
    if not images:
        return 0

    selected = rng.sample(images, min(samples_per_split, len(images)))
    generated = 0

    for image_info in selected:
        filename = image_info["file_name"]
        image_path = split_image_path(ROBOFLOW_COCO_ROOT, split, filename)
        mask_name = mask_filename_for_image(filename)
        expected_size = (int(image_info["width"]), int(image_info["height"]))

        masks = {}
        for class_name in EXPECTED_CLASSES:
            mask_path = MASKS_ROOT / split / class_name / mask_name
            if not mask_path.exists():
                print(f"ADVERTENCIA: falta mascara {mask_path}")
                continue
            masks[class_name] = load_binary_mask(mask_path, expected_size=expected_size)

        if not image_path.exists() or not masks:
            continue

        output_path = QC_MASKS_ROOT / split / f"{mask_name.replace('.png', '')}_overlay.png"
        overlay_masks(image_path, masks, CLASS_COLORS_RGB, output_path)
        generated += 1

    return generated


def main() -> None:
    """Punto de entrada."""
    args = parse_args()
    ensure_output_dirs()
    rng = random.Random(args.seed)

    total = 0
    for split in SPLITS:
        total += process_split(split, args.samples_per_split, rng)

    print(f"Overlays generados: {total}")
    print(f"Carpeta: {QC_MASKS_ROOT}")


if __name__ == "__main__":
    main()
