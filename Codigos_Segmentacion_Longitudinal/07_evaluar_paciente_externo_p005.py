"""Prepara validacion externa de P005 sin mezclarlo con Roboflow train/valid/test."""

from __future__ import annotations

import pandas as pd

from config import FRAMES_PROCESSED_ROOT, REPORTS_ROOT, ensure_output_dirs


def main() -> None:
    """Busca imagenes longitudinales de P005 y genera reporte base."""
    ensure_output_dirs()
    p005_root = FRAMES_PROCESSED_ROOT / "P005" / "longitudinal"
    rows = []

    if not p005_root.exists():
        print(f"ADVERTENCIA: no existe {p005_root}")
    else:
        for image_path in sorted(p005_root.rglob("*.png")):
            quality = image_path.parent.name.lower()
            rows.append(
                {
                    "patient": "P005",
                    "view": "longitudinal",
                    "quality": quality,
                    "filename": image_path.name,
                    "image_path": str(image_path),
                    "has_manual_annotation": 0,
                    "dice_roi": "",
                    "dice_higado": "",
                    "dice_la": "",
                    "iou_roi": "",
                    "iou_higado": "",
                    "iou_la": "",
                    "notes": "reservado_para_validacion_externa;sin_anotacion_manual_actual",
                }
            )

    output_csv = REPORTS_ROOT / "external_validation_p005.csv"
    pd.DataFrame(rows).to_csv(output_csv, index=False, encoding="utf-8-sig")

    print(f"Imagenes longitudinales P005 encontradas: {len(rows)}")
    print(f"Reporte generado: {output_csv}")
    print("P005 no se usa para ajustar umbrales ni entrenar modelos.")


if __name__ == "__main__":
    main()
