"""Audita el dataset COCO exportado desde Roboflow."""

from __future__ import annotations

from collections import Counter, defaultdict

import pandas as pd

from config import EXPECTED_CLASSES, REPORTS_ROOT, ROBOFLOW_COCO_ROOT, SPLITS, annotations_path, ensure_output_dirs
from utils.coco_utils import annotations_by_image_and_class, category_id_to_name, load_coco_json, validate_expected_classes
from utils.metadata_utils import metadata_from_filename


def audit_split(split: str) -> list[dict]:
    """Audita un split COCO y devuelve filas por paciente/vista/calidad."""
    json_path = annotations_path(split)
    if not json_path.exists():
        print(f"ADVERTENCIA: no existe {json_path}")
        return []

    coco = load_coco_json(json_path)
    ok_classes, missing = validate_expected_classes(coco)
    categories = category_id_to_name(coco)

    print(f"\nSplit: {split}")
    print(f"  JSON: {json_path}")
    print(f"  Imagenes COCO: {len(coco.get('images', []))}")
    print(f"  Anotaciones COCO: {len(coco.get('annotations', []))}")
    print(f"  Clases detectadas relevantes: {sorted(set(categories.values()))}")
    if not ok_classes:
        print(f"  ADVERTENCIA: faltan clases esperadas: {missing}")

    grouped_annotations = annotations_by_image_and_class(coco)
    rows_by_group: dict[tuple[str, str, str], dict] = {}
    annotation_counter = Counter()
    images_with_class = defaultdict(set)

    for image in coco.get("images", []):
        filename = image.get("file_name", "")
        metadata = metadata_from_filename(filename)
        key = (metadata["patient"], metadata["view"], metadata["quality"])
        if key not in rows_by_group:
            rows_by_group[key] = {
                "split": split,
                "patient": metadata["patient"],
                "view": metadata["view"],
                "quality": metadata["quality"],
                "image_count": 0,
                "annotations_ROI": 0,
                "annotations_Higado": 0,
                "annotations_LA": 0,
                "images_with_ROI": 0,
                "images_with_Higado": 0,
                "images_with_LA": 0,
            }
        rows_by_group[key]["image_count"] += 1

        image_annotations = grouped_annotations.get(int(image["id"]), {})
        for class_name in EXPECTED_CLASSES:
            count = len(image_annotations.get(class_name, []))
            rows_by_group[key][f"annotations_{class_name}"] += count
            annotation_counter[class_name] += count
            if count > 0:
                images_with_class[class_name].add(int(image["id"]))

    for row in rows_by_group.values():
        # Se recalculan por grupo para que el CSV sea util por paciente/calidad.
        row["classes_ok"] = int(ok_classes)
        row["missing_classes"] = ",".join(missing)

    print("  Anotaciones por clase:")
    for class_name in EXPECTED_CLASSES:
        print(f"    {class_name}: {annotation_counter[class_name]}")

    return list(rows_by_group.values())


def main() -> None:
    """Ejecuta auditoria completa."""
    ensure_output_dirs()

    if not ROBOFLOW_COCO_ROOT.exists():
        raise FileNotFoundError(f"No existe ROBOFLOW_COCO_ROOT: {ROBOFLOW_COCO_ROOT}")

    all_rows = []
    for split in SPLITS:
        all_rows.extend(audit_split(split))

    output_csv = REPORTS_ROOT / "roboflow_dataset_audit.csv"
    df = pd.DataFrame(all_rows)
    df.to_csv(output_csv, index=False, encoding="utf-8-sig")

    print(f"\nCSV generado: {output_csv}")
    if not df.empty:
        print("\nResumen por split:")
        print(df.groupby("split")["image_count"].sum().to_string())
        print("\nResumen por paciente/calidad:")
        print(df.groupby(["patient", "quality"])["image_count"].sum().to_string())


if __name__ == "__main__":
    main()
