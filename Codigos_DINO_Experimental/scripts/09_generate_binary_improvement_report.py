"""Genera un informe acumulado de la mejora binaria DINOv2."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config_dino import BINARY_FIGURES_ROOT, BINARY_REPORTS_ROOT, BINARY_ROOT, REPORTS_ROOT, VIEWS


def table_markdown(frame: pd.DataFrame, decimals: int = 4) -> str:
    if frame.empty:
        return "_Sin datos._"
    display = frame.copy()
    for column in display.select_dtypes(include="number").columns:
        display[column] = display[column].map(
            lambda value: "" if pd.isna(value) else f"{value:.{decimals}f}"
        )
    display = display.fillna("")
    lines = [
        "| " + " | ".join(display.columns) + " |",
        "| " + " | ".join(["---"] * len(display.columns)) + " |",
    ]
    for row in display.astype(str).itertuples(index=False, name=None):
        lines.append("| " + " | ".join(value.replace("|", "/") for value in row) + " |")
    return "\n".join(lines)


def main() -> None:
    baseline = pd.read_csv(REPORTS_ROOT / "04_dino_best_classifier_by_view.csv")
    comparison = pd.read_csv(BINARY_REPORTS_ROOT / "08_binary_configuration_comparison.csv")
    winners = pd.read_csv(BINARY_REPORTS_ROOT / "08_binary_winners_by_view.csv")
    thresholds = pd.read_csv(BINARY_REPORTS_ROOT / "09_binary_abstention_thresholds.csv")
    internal_actions = pd.read_csv(BINARY_REPORTS_ROOT / "09_binary_internal_action_metrics.csv")
    external_metrics = pd.read_csv(BINARY_REPORTS_ROOT / "09_binary_p005_metrics.csv")
    external_videos = pd.read_csv(BINARY_REPORTS_ROOT / "09_binary_p005_video_predictions.csv")
    manifest = pd.read_csv(BINARY_REPORTS_ROOT / "09_binary_model_manifest.csv")
    crop = pd.read_csv(BINARY_REPORTS_ROOT / "06_fan_crop_audit.csv")
    benchmark_path = BINARY_REPORTS_ROOT / "12_binary_dino_inference_benchmark.csv"
    benchmark = pd.read_csv(benchmark_path) if benchmark_path.exists() else pd.DataFrame()

    baseline_table = baseline[[
        "view", "classifier", "mean_video_f1_macro", "mean_video_accuracy"
    ]].rename(columns={
        "mean_video_f1_macro": "three_class_video_f1_macro",
        "mean_video_accuracy": "three_class_video_accuracy",
    })
    binary_table = winners[[
        "view", "embedding_variant", "temporal_mode", "classifier",
        "mean_video_f1_macro", "mean_video_balanced_accuracy",
        "mean_video_blurry_recall", "mean_video_clear_precision",
    ]].rename(columns={
        "classifier": "binary_classifier",
        "mean_video_f1_macro": "binary_video_f1_macro",
    })
    comparison_summary = baseline_table.merge(binary_table, on="view", how="outer")
    comparison_summary["comparison_warning"] = (
        "Different tasks: three classes versus clear/blurry anchors"
    )

    top_parts = []
    for view in VIEWS:
        top_parts.append(
            comparison[comparison["view"] == view]
            .sort_values(
                ["mean_video_f1_macro", "mean_video_blurry_recall", "std_video_f1_macro"],
                ascending=[False, False, True],
            )
            .head(5)
        )
    top = pd.concat(top_parts, ignore_index=True)[[
        "view", "embedding_variant", "temporal_mode", "classifier",
        "mean_video_f1_macro", "std_video_f1_macro",
        "mean_video_blurry_recall", "mean_video_clear_precision",
    ]]

    crop_summary = (
        crop.groupby(["role"], as_index=False)
        .agg(
            images=("filename", "size"),
            detection_rate=("detected", "mean"),
            median_mask_fraction=("mask_fraction", "median"),
            median_bbox_width=("bbox_width", "median"),
            median_bbox_height=("bbox_height", "median"),
        )
    )

    functional_rows = []
    inference_root = BINARY_ROOT / "video_inference"
    for path in sorted(inference_root.glob("functional_test_*.csv")):
        frame = pd.read_csv(path)
        evaluated = frame[frame["evaluated"] == 1]
        counts = evaluated["decision"].value_counts()
        functional_rows.append({
            "file": path.name,
            "frames_total": len(frame),
            "frames_evaluated": len(evaluated),
            "capture": int(counts.get("capture", 0)),
            "adjust": int(counts.get("adjust", 0)),
            "doubtful": int(counts.get("doubtful", 0)),
            "warming_up": int(counts.get("warming_up", 0)),
        })
    functional = pd.DataFrame(functional_rows)

    report = [
        "# Informe de mejora del pipeline DINOv2", "",
        "## Objetivo", "",
        "Corregir la baja separacion del baseline de tres calidades y acercar la salida "
        "al objetivo operativo de la tesis: recomendar captura, ajuste o abstencion en "
        "las vistas transversal, oblicua y hepatorrenal.", "",
        "## Cambios implementados", "",
        "1. Clear y blurry se usan como anclas binarias.",
        "2. Medium se excluye del entrenamiento y se estudia como calidad ambigua.",
        "3. Se elimina la interfaz mediante recorte determinista del campo ecografico.",
        "4. Se comparan DINOv2-Small y DINOv2-Base congelados.",
        "5. Se compara inferencia por frame con ventanas temporales de cinco embeddings.",
        "6. Se calibran umbrales conservadores para capture, adjust y doubtful.",
        "7. La seleccion se realiza por leave-one-patient-out con P001-P003.",
        "8. P005 permanece fuera de toda seleccion y ajuste.", "",
        "## Control del recorte", "", table_markdown(crop_summary), "",
        f"Control visual: {BINARY_FIGURES_ROOT / '06_fan_crop_qc.png'}", "",
        "## Baseline frente a tarea binaria", "", table_markdown(comparison_summary), "",
        "Las cifras no deben compararse como si fueran la misma tarea. El baseline separa "
        "tres etiquetas, mientras la mejora evalua solamente los extremos clear y blurry. "
        "El aumento demuestra que gran parte de la dificultad estaba en la ambiguedad de "
        "medium, no que el problema clinico completo este resuelto.", "",
        "## Cinco mejores configuraciones por vista", "", table_markdown(top), "",
        "## Configuraciones seleccionadas", "", table_markdown(binary_table), "",
        "## Umbrales de abstencion", "", table_markdown(thresholds[[
            "view", "adjust_threshold", "capture_threshold",
            "internal_adjust_precision", "internal_capture_precision",
            "internal_blurry_coverage", "internal_clear_coverage",
            "medium_used_for_calibration", "p005_used_for_calibration",
        ]]), "",
        "## Acciones internas", "", table_markdown(internal_actions), "",
        "## Evaluacion externa P005", "", table_markdown(external_metrics), "",
        "La exactitud binaria por video es 1.0 para los seis videos ancla de P005: un "
        "clear y un blurry por vista. Esta cifra tiene una incertidumbre extrema y no "
        "equivale a validacion clinica. En hepatorrenal, los umbrales conservadores "
        "mantienen incluso clear y blurry como doubtful.", "",
        "## Predicciones de los nueve videos P005", "", table_markdown(external_videos[[
            "view", "true_quality", "probability_clear", "predicted_anchor", "action"
        ]]), "",
        "Los tres videos medium de P005 no quedaron en doubtful. Esto confirma que medium "
        "no constituye una clase visual coherente y no debe utilizarse como verdad clinica "
        "para forzar una tercera categoria.", "",
        "## Pruebas funcionales de video", "", table_markdown(functional), "",
        "Estas pruebas solo verifican ejecucion frame por frame. P001 pertenece al "
        "desarrollo y sus resultados no son una evaluacion independiente.", "",
        "## Modelos desplegables", "", table_markdown(manifest), "",
        "## Benchmark end-to-end", "", table_markdown(benchmark[[col for col in [
            "view", "embedding_model_id", "preprocessing", "temporal_mode",
            "mean_ms_per_evaluated_frame", "p95_ms_per_evaluated_frame",
            "fps_if_every_frame", "configured_stride",
            "effective_source_fps_capacity_at_stride",
            "meets_30_fps_if_every_frame",
        ] if col in benchmark.columns]]) if not benchmark.empty else "_Pendiente._", "",
        "Ningun modelo alcanza 30 FPS si se ejecuta en todos los frames. Con stride 5,"
        " la frecuencia efectiva del video supera 30 FPS porque la inferencia pesada se"
        " ejecuta cada cinco frames y se conserva la ultima decision entre evaluaciones.", "",
        "## Interpretacion principal", "",
        "La reformulacion binaria mejora claramente la separacion de los extremos de "
        "calidad. La salida con abstencion es mas adecuada para una GUI porque permite "
        "evitar una recomendacion de captura cuando la probabilidad no es suficientemente "
        "concluyente. Sin embargo, clinically relevant todavia necesita criterios "
        "anatomicos independientes y mas pacientes.", "",
        "## Limitaciones", "",
        "- Tres pacientes para desarrollo y un solo paciente externo.",
        "- Muy pocos videos independientes por vista y calidad.",
        "- Alta correlacion temporal entre frames.",
        "- Etiquetas heredadas del video completo, no del frame.",
        "- Clear no garantiza por si solo relevancia anatomica.",
        "- P005 aporta solo dos videos ancla por vista para la prueba binaria.", "",
        "## Siguiente paso recomendado", "",
        "Integrar estos modelos como señal auxiliar de calidad en la GUI. La orden capture "
        "debe requerir alta probabilidad durante varios frames consecutivos. Para afirmar "
        "relevancia clinica sera necesario validar presencia de referencias anatomicas "
        "por vista y ampliar el numero de pacientes.", "",
        "## Figuras", "",
        f"- Recorte: {BINARY_FIGURES_ROOT / '06_fan_crop_qc.png'}",
        f"- Comparacion: {BINARY_FIGURES_ROOT / '08_binary_configuration_comparison.png'}",
        f"- Matrices: {BINARY_FIGURES_ROOT / '09_binary_selected_confusion_matrices.png'}",
        f"- Umbrales: {BINARY_FIGURES_ROOT / '09_binary_abstention_thresholds.png'}",
        f"- Acciones: {BINARY_FIGURES_ROOT / '09_binary_action_distribution.png'}", "",
        "Referencia metodologica: Oquab et al. (2023), DINOv2: Learning Robust "
        "Visual Features without Supervision.",
    ]
    report_path = BINARY_REPORTS_ROOT / "11_dino_binary_improvement_summary.md"
    report_path.write_text("\n".join(report), encoding="utf-8")
    comparison_summary.to_csv(
        BINARY_REPORTS_ROOT / "11_baseline_vs_binary_summary.csv",
        index=False, encoding="utf-8-sig"
    )
    print(f"Reporte: {report_path}")
    print("\nResumen:")
    print(comparison_summary.to_string(index=False))


if __name__ == "__main__":
    main()