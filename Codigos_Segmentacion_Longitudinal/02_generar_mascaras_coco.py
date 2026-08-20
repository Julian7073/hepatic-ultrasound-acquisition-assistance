"""Convierte anotaciones COCO de Roboflow a mascaras PNG por clase."""

from __future__ import annotations

import pandas as pd
from PIL import Image
from tqdm import tqdm

from config import EXPECTED_CLASSES, MASKS_ROOT, REPORTS_ROOT, ROBOFLOW_COCO_ROOT, SPLITS, annotations_path, ensure_output_dirs
from utils.coco_utils import annotations_by_image_and_class, load_coco_json, split_image_path
from utils.mask_utils import annotations_to_mask, mask_area, save_binary_mask
from utils.metadata_utils import mask_filename_for_image, metadata_from_filename


def process_split(split: str) -> list[dict]:
    """Genera mascaras para un split."""
    json_path = annotations_path(split)
    if not json_path.exists():
        print(f"ADVERTENCIA: no existe {json_path}")
        return []

    coco = load_coco_json(json_path)
    grouped_annotations = annotations_by_image_and_class(coco)
    rows = []

    for image_info in tqdm(coco.get("images", []), desc=f"Mascaras {split}"):
        filename = image_info["file_name"]
        image_path = split_image_path(ROBOFLOW_COCO_ROOT, split, filename)
        metadata = metadata_from_filename(filename)
        notes = []

        width = int(image_info["width"])
        height = int(image_info["height"])

        if image_path.exists():
            with Image.open(image_path) as image:
                if image.size != (width, height):
                    notes.append(f"tamano_coco_distinto_a_imagen:{width}x{height}_vs_{image.size[0]}x{image.size[1]}")
                    width, height = image.size
        else:
            notes.append("imagen_no_encontrada")

        image_annotations = grouped_annotations.get(int(image_info["id"]), {})
        mask_name = mask_filename_for_image(filename)

        for class_name in EXPECTED_CLASSES:
            annotations = image_annotations.get(class_name, [])
            mask = annotations_to_mask(annotations, width, height)
            output_path = MASKS_ROOT / split / class_name / mask_name
            save_binary_mask(mask, output_path)

            rows.append(
                {
                    "split": split,
                    "filename": filename,
                    "patient": metadata["patient"],
                    "view": metadata["view"],
                    "quality": metadata["quality"],
                    "class_name": class_name,
                    "has_annotation": int(len(annotations) > 0),
                    "annotation_count": len(annotations),
                    "mask_area_px": mask_area(mask),
                    "mask_path": str(output_path),
                    "notes": ";".join(notes),
                }
            )

    return rows


def main() -> None:
    """Ejecuta conversion de COCO a mascaras."""
    ensure_output_dirs()
    all_rows = []

    for split in SPLITS:
        all_rows.extend(process_split(split))

    report_path = REPORTS_ROOT / "mask_generation_report.csv"
    pd.DataFrame(all_rows).to_csv(report_path, index=False, encoding="utf-8-sig")

    print(f"\nReporte generado: {report_path}")
    if all_rows:
        df = pd.DataFrame(all_rows)
        print("\nMascaras sin anotacion por clase:")
        missing = df[df["has_annotation"] == 0].groupby(["split", "class_name"]).size()
        print(missing.to_string())


if __name__ == "__main__":
    main()
