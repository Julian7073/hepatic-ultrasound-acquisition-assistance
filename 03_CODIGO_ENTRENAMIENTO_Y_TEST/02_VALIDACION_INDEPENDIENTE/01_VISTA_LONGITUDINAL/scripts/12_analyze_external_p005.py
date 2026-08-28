"""Genera interpretación y candidatos de revisión manual para P005."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config_experimental import REPORTS_ROOT
from src.reports import table_markdown


def spaced(frame: pd.DataFrame, count: int) -> pd.DataFrame:
    if frame.empty:
        return frame
    indexes = np.linspace(
        0,
        len(frame) - 1,
        num=min(count, len(frame)),
        dtype=int,
    )
    return frame.iloc[indexes].copy()


def main() -> None:
    results_path = REPORTS_ROOT / "10_external_validation_p005.csv"
    if not results_path.exists():
        raise FileNotFoundError(
            "Ejecute scripts/10_evaluate_external_p005.py primero."
        )
    results = pd.read_csv(results_path)
    candidates = []

    groups = [
        (
            "clear_without_predicted_la",
            results[(results["quality"] == "clear") & (results["has_la"] == 0)],
            5,
        ),
        (
            "medium_with_predicted_la",
            results[(results["quality"] == "medium") & (results["has_la"] == 1)],
            5,
        ),
        (
            "medium_without_predicted_la",
            results[(results["quality"] == "medium") & (results["has_la"] == 0)],
            5,
        ),
        (
            "blurry_without_predicted_la",
            results[(results["quality"] == "blurry") & (results["has_la"] == 0)],
            5,
        ),
    ]
    for reason, frame, count in groups:
        selected = spaced(frame.sort_values("filename"), count)
        selected["manual_review_reason"] = reason
        candidates.append(selected)

    review = pd.concat(candidates, ignore_index=True)
    columns = [
        "patient",
        "quality",
        "filename",
        "image_path",
        "decision",
        "decision_reason",
        "has_la",
        "raw_area_la_px",
        "area_la_px",
        "la_std_intensity",
        "glcm_entropy",
        "manual_review_reason",
    ]
    review = review[columns]
    review_path = REPORTS_ROOT / "10_p005_manual_review_candidates.csv"
    review.to_csv(review_path, index=False, encoding="utf-8-sig")

    summary = pd.read_csv(
        REPORTS_ROOT / "10_external_validation_p005_summary.csv"
    )
    overall = summary[summary["quality"] == "overall"].iloc[0]
    report = [
        "# Interpretacion de la evaluacion externa P005", "",
        "## Hallazgos", "",
        f"- Se procesaron {int(overall['total_images'])} imagenes externas.",
        f"- ROI detectada: {overall['has_roi_rate']:.2%}.",
        f"- Higado detectado: {overall['has_higado_rate']:.2%}.",
        f"- LA detectado: {int(overall['has_la_count'])} imagenes "
        f"({overall['has_la_rate']:.2%}).",
        f"- Captura sugerida: {int(overall['capture_count'])} imagenes "
        f"({overall['capture_rate']:.2%}).",
        "",
        "Las siete detecciones LA aparecieron en la categoria medium. Ninguna supero "
        "simultaneamente area y criterios de textura, por lo que no se sugirio captura.", "",
        "## Interpretacion metodologica", "",
        "Sin mascaras P005 no es posible decidir si la baja deteccion LA representa "
        "falsos negativos del modelo, ausencia de lumen claramente visible en la adquisicion "
        "o ambos factores. La revision de overlays clear muestra estructuras oscuras plausibles "
        "que requieren confirmacion manual; por ello estos casos se registran para revision.", "",
        "El resultado demuestra que las metricas internas de P001-P003 no deben extrapolarse "
        "automaticamente a un paciente nuevo. ROI e Higado fueron estables, mientras LA sigue "
        "siendo el componente limitante.", "",
        "## Requisito para validacion cuantitativa", "",
        "Anotar manualmente al menos 30 frames P005 estratificados por calidad, incluyendo "
        "casos con y sin referencia anatomica visible. Solo entonces se podran calcular Dice, "
        "IoU, recall y falsos positivos externos.", "",
        "## Casos propuestos para revision", "",
        table_markdown(review), "",
    ]
    report_path = REPORTS_ROOT / "10_external_validation_p005_interpretation.md"
    report_path.write_text("\n".join(report), encoding="utf-8")
    print("\n".join(report[:24]))
    print(f"Candidatos: {review_path}")
    print(f"Interpretacion: {report_path}")


if __name__ == "__main__":
    main()
