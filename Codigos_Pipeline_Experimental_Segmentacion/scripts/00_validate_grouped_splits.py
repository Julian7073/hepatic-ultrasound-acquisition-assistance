"""Reporta tamano y positivos de los splits agrupados sin entrenar."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from config_experimental import CLASS_NAMES, DATASET_ROOTS, REPORTS_ROOT
from src.dataset_coco import assign_grouped_splits, load_records


if __name__ == "__main__":
    rows = []
    for class_name in CLASS_NAMES:
        records = load_records(DATASET_ROOTS[class_name], class_name)
        grouped = assign_grouped_splits(records, "group_video", seed=42)
        print(f"\nClase: {class_name}")
        for split, items in grouped.items():
            positives = sum(bool(item.annotations) for item in items)
            patients = sorted({item.patient for item in items})
            videos = sorted({item.video_id for item in items})
            print(
                f"{split}: images={len(items)} positives={positives} "
                f"patients={patients} videos={len(videos)}"
            )
            rows.append({
                "class_name": class_name, "split": split, "images": len(items),
                "positive_images": positives, "patients": " | ".join(patients),
                "video_count": len(videos), "seed": 42, "strategy": "group_video",
            })


    frame = pd.DataFrame(rows)
    REPORTS_ROOT.mkdir(parents=True, exist_ok=True)
    frame.to_csv(REPORTS_ROOT / "00_grouped_split_preview.csv", index=False, encoding="utf-8-sig")
    lines = [
        "# Vista previa del split agrupado por video", "",
        "Esta distribucion evita que un mismo video aparezca en varios splits, pero no separa pacientes.",
        "Con seed 42, test contiene un solo video de P001; por tanto sigue siendo una validacion interna limitada y no reemplaza P005.", "",
        frame.to_string(index=False), "",
    ]
    (REPORTS_ROOT / "00_grouped_split_preview.md").write_text("\n".join(lines), encoding="utf-8")
