"""Genera un resumen tecnico autocontenido de la fase DINOv2."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config_dino import EMBEDDINGS_ROOT, FIGURES_ROOT, REPORTS_ROOT, VIEWS


def table_markdown(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_Sin datos._"
    display = frame.copy().fillna("")
    lines = [
        "| " + " | ".join(display.columns) + " |",
        "| " + " | ".join(["---"] * len(display.columns)) + " |",
    ]
    for row in display.astype(str).itertuples(index=False, name=None):
        lines.append("| " + " | ".join(value.replace("|", "/") for value in row) + " |")
    return "\n".join(lines)


def main() -> None:
    audit = pd.read_csv(REPORTS_ROOT / "00_dino_dataset_audit.csv")
    videos = pd.read_csv(REPORTS_ROOT / "00_dino_video_groups.csv")
    selection = pd.read_csv(REPORTS_ROOT / "04_dino_classifier_selection.csv")
    winners = pd.read_csv(REPORTS_ROOT / "04_dino_best_classifier_by_view.csv")
    external = pd.read_csv(REPORTS_ROOT / "05_dino_p005_metrics.csv")
    external_videos = pd.read_csv(REPORTS_ROOT / "05_dino_p005_video_predictions.csv")
    embedding_info = json.loads(
        (EMBEDDINGS_ROOT / "dinov2_small_stride5_info.json").read_text(encoding="utf-8")
    )

    winner_columns = [
        "view", "classifier", "mean_video_f1_macro", "std_video_f1_macro",
        "mean_video_accuracy", "mean_frame_f1_macro", "mean_frame_accuracy",
        "p005_used_for_selection",
    ]
    external_columns = [
        "view", "classifier", "video_accuracy", "video_f1_macro",
        "frame_accuracy", "frame_precision_macro", "frame_recall_macro",
        "frame_f1_macro", "test_videos", "test_images",
    ]
    video_columns = [
        "view", "true_quality", "predicted_quality", "frame_count", "correct",
    ]
    report = [
        "# Resumen tecnico de la fase DINOv2", "",
        "## Objetivo", "",
        "Evaluar si representaciones auto-supervisadas DINOv2 permiten discriminar la "
        "calidad nominal clear, medium y blurry en las vistas transversal, oblicua y "
        "hepatorrenal, sin utilizar P005 durante la seleccion.", "",
        "## Dataset", "",
        f"- Frames auditados: {int(audit['image_count'].sum())}",
        f"- Videos fuente: {len(videos)}",
        f"- Videos de desarrollo P001-P003: {int((videos['role'] == 'development').sum())}",
        f"- Videos externos P005: {int((videos['role'] == 'external_test').sum())}",
        "- Duplicados exactos: 0",
        "- Imagenes ilegibles: 0",
        "- Resolucion original: 1024x768", "",
        "Los grupos con 202 frames contienen dos videos diferentes de 101 frames. "
        "Cada video se conserva como unidad indivisible.", "",
        "## Extraccion de caracteristicas", "",
        f"- Modelo: {embedding_info['model_id']}",
        f"- Dimension: {embedding_info['embedding_dim']}",
        f"- Stride temporal: {embedding_info['stride']}",
        f"- Imagenes con embedding: {embedding_info['images']}",
        f"- Desarrollo: {embedding_info['development_images']}",
        f"- P005: {embedding_info['external_test_images']}",
        f"- Device: {embedding_info['device']}", "",
        "DINOv2 permanece congelado. Se usa el token CLS de 384 dimensiones; no se "
        "realiza fine-tuning con el dataset ecografico.", "",
        "## Protocolo de seleccion", "",
        "Se compararon Logistic Regression, SVM RBF, Random Forest y k-NN mediante "
        "leave-one-patient-out sobre P001-P003. La metrica principal fue F1 macro por "
        "video. Las metricas por frame se conservaron como complemento.", "",
        "## Comparacion interna completa", "", table_markdown(selection), "",
        "## Modelos seleccionados", "", table_markdown(winners[winner_columns]), "",
        "## Evaluacion externa P005", "", table_markdown(external[external_columns]), "",
        "## Prediccion de los nueve videos P005", "", table_markdown(
            external_videos[video_columns].sort_values(["view", "true_quality"])
        ), "",
        "## Interpretacion", "",
        "Transversal presenta la mejor consistencia interna, pero su F1 macro por video "
        "sigue siendo moderado. Oblicua y hepatorrenal muestran alta variabilidad entre "
        "pacientes, indicando que los embeddings congelados no separan de forma robusta "
        "las tres calidades con el dataset actual.", "",
        "En P005 cada vista clasifica correctamente dos de tres videos. Hepatorrenal y "
        "transversal confunden blurry con medium; oblicua confunde medium con clear.", "",
        "La evaluacion externa contiene solo tres videos por vista, uno por clase. Un "
        "acierto o error cambia la exactitud en 33.3 puntos porcentuales; por ello no se "
        "debe presentar 66.7% como evidencia concluyente.", "",
        "## Limitaciones", "",
        "- Solo existen cuatro pacientes validos y tres pacientes de desarrollo.",
        "- Los frames de video siguen correlacionados incluso con stride 5.",
        "- Clear, medium y blurry son etiquetas nominales de adquisicion, no una "
        "evaluacion clinica independiente de informativeness.",
        "- No se realizo busqueda de hiperparametros para evitar sobreajuste.",
        "- P005 no participo en la seleccion, pero su tamaño externo es muy reducido.", "",
        "## Conclusion", "",
        "DINOv2 es un baseline reproducible y rapido para las vistas no longitudinales, "
        "pero el clasificador de tres calidades no es suficientemente estable para actuar "
        "como criterio clinico autonomo. Puede integrarse como señal auxiliar en la GUI y "
        "debe acompañarse de mensajes conservadores y validacion con mas pacientes.", "",
        "## Figuras", "",
        f"- Comparacion interna: {FIGURES_ROOT / '04_dino_classifier_comparison.png'}",
        f"- Matrices de confusion: {FIGURES_ROOT / '05_dino_selected_confusion_matrices.png'}",
        f"- PCA: {FIGURES_ROOT / '05_dino_pca_by_view_quality.png'}", "",
        "Referencia base: Oquab et al. (2023), DINOv2: Learning Robust Visual Features "
        "without Supervision.",
    ]
    output_path = REPORTS_ROOT / "06_dino_technical_summary.md"
    output_path.write_text("\n".join(report), encoding="utf-8")
    print(f"Reporte tecnico: {output_path}")


if __name__ == "__main__":
    main()
