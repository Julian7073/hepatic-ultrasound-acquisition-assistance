"""Genera reporte tecnico Markdown de la fase de segmentacion."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from config_segmentation import REPORTS_ROOT, ensure_output_dirs


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    """Convierte DataFrame a Markdown sin depender de tabulate."""
    if df.empty:
        return "Tabla vacia."
    headers = [str(column) for column in df.columns]
    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for _, row in df.iterrows():
        values = [str(row[column]) if pd.notna(row[column]) else "" for column in df.columns]
        values = [value.replace("|", "/") for value in values]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def table_or_note(path: Path, columns: list[str] | None = None) -> str:
    """Convierte CSV a tabla Markdown o deja nota si falta."""
    if not path.exists():
        return f"No disponible todavia: `{path.name}`"
    df = pd.read_csv(path)
    if columns:
        columns = [column for column in columns if column in df.columns]
        df = df[columns]
    return dataframe_to_markdown(df)


def main() -> None:
    """Escribe reporte Markdown."""
    ensure_output_dirs()
    audit_path = REPORTS_ROOT / "coco_separated_audit.csv"
    ranking_path = REPORTS_ROOT / "architecture_ranking_by_class.csv"
    best_path = REPORTS_ROOT / "best_models_by_class.csv"

    report = f"""# Reporte tecnico de entrenamiento local de segmentacion

## Objetivo

Comparar U-Net, DeepLabV3+ y SegFormer para segmentar por separado ROI, Higado y LA en vista longitudinal.

## Justificacion metodologica

Los datasets separados se usan porque cada estructura tiene una funcion distinta dentro del sistema:

- ROI delimita el campo ecografico util.
- Higado verifica que la estructura hepatica este visible.
- LA aporta la referencia anatomica interna usada en la regla longitudinal.

Roboflow se usa como herramienta de anotacion y exportacion COCO. El resize y las augmentations se aplican localmente para mantener una comparacion justa entre arquitecturas.

## Auditoria de datasets

{table_or_note(audit_path, ["dataset_class_expected", "split", "image_count_coco", "annotation_count_target", "images_with_target_annotation", "status", "notes"])}

## Arquitecturas

- **U-Net:** baseline eficiente para segmentacion biomedica.
- **DeepLabV3+:** arquitectura encoder-decoder con contexto multiescala.
- **SegFormer:** arquitectura basada en transformers para segmentacion semantica.

## Metricas

Se reportan Dice, IoU, precision, recall, F1, loss de entrenamiento/validacion, tiempo promedio de inferencia por frame y numero aproximado de parametros.

## Ranking por clase

{table_or_note(ranking_path)}

## Mejores modelos seleccionados

{table_or_note(best_path)}

## Limitaciones

- El entrenamiento depende de la disponibilidad de GPU para tiempos razonables.
- SegFormer entrenado desde cero puede requerir mas datos o fine-tuning con pesos preentrenados.
- LA aparece en menos imagenes que ROI e Higado, por lo que la evaluacion visual sigue siendo importante.

## Proximo paso

Despues de seleccionar mejores modelos, conectar ROI, Higado y LA al pipeline de inferencia frame por frame para generar mensajes de guia al usuario.
"""
    output_path = REPORTS_ROOT / "segmentation_training_report.md"
    output_path.write_text(report, encoding="utf-8")
    print(f"Reporte generado: {output_path}")


if __name__ == "__main__":
    main()
