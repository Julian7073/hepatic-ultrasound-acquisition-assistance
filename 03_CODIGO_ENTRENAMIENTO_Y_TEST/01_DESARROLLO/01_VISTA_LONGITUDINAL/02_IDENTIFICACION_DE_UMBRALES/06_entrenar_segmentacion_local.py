"""Preparacion para entrenamiento local de segmentacion longitudinal.

Este script NO entrena todavia. Deja documentada y verificable la estructura
que luego se usara para comparar U-Net, DeepLabV3+ y SegFormer.
"""

from __future__ import annotations

import pandas as pd

from config import MASKS_ROOT, REPORTS_ROOT, ROBOFLOW_COCO_ROOT, SPLITS, annotations_path, ensure_output_dirs
from utils.coco_utils import load_coco_json, split_image_path
from utils.metadata_utils import mask_filename_for_image, metadata_from_filename


def main() -> None:
    """Crea un indice local imagen/mascara para entrenamiento futuro."""
    ensure_output_dirs()
    rows = []

    for split in SPLITS:
        json_path = annotations_path(split)
        if not json_path.exists():
            continue
        coco = load_coco_json(json_path)
        for image_info in coco.get("images", []):
            filename = image_info["file_name"]
            metadata = metadata_from_filename(filename)
            mask_name = mask_filename_for_image(filename)
            rows.append(
                {
                    "split": split,
                    "filename": filename,
                    "patient": metadata["patient"],
                    "quality": metadata["quality"],
                    "image_path": str(split_image_path(ROBOFLOW_COCO_ROOT, split, filename)),
                    "mask_roi_path": str(MASKS_ROOT / split / "ROI" / mask_name),
                    "mask_higado_path": str(MASKS_ROOT / split / "Higado" / mask_name),
                    "mask_la_path": str(MASKS_ROOT / split / "LA" / mask_name),
                    "priority_class": "LA",
                    "planned_models": "UNet_ligera;DeepLabV3Plus;SegFormer_small",
                    "status": "dataset_index_prepared_no_training_run",
                }
            )

    output_csv = REPORTS_ROOT / "local_segmentation_training_index.csv"
    pd.DataFrame(rows).to_csv(output_csv, index=False, encoding="utf-8-sig")

    print("Indice preparado para entrenamiento local posterior.")
    print(f"CSV: {output_csv}")
    print("Nota: no se entreno ningun modelo en esta fase.")


if __name__ == "__main__":
    main()
