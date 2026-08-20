"""Reportes por experimento y tablas acumuladas para tesis."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from config_experimental import EXPERIMENTS_ROOT, REPORTS_ROOT


SMOKE_WARNING = (
    "Este experimento es una prueba funcional del pipeline. Las metricas no se interpretan "
    "como desempeno final porque 2 epocas no permiten evaluar convergencia ni generalizacion."
)


def table_markdown(frame: pd.DataFrame) -> str:
    """Convierte una tabla pequena a Markdown."""
    if frame.empty:
        return "_Sin datos disponibles._"
    columns = [str(column) for column in frame.columns]
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in frame.fillna("").astype(str).itertuples(index=False, name=None):
        values = [value.replace("|", "/").replace("\n", " ") for value in row]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_experiment_summary(experiment_dir: Path, config: dict, validation: dict, test: dict | None, benchmark: dict | None) -> None:
    """Genera un resumen autocontenido del experimento."""
    lines = [
        f"# Experimento: {experiment_dir.name}",
        "",
        f"- Clase: {config['class_name']}",
        f"- Arquitectura: {config['architecture']}",
        f"- Encoder: {config['model_metadata']['encoder']}",
        f"- Pretrained: {config['pretrained']}",
        f"- Epocas: {config['epochs']}",
        f"- Resolucion: {config['image_size']}x{config['image_size']}",
        f"- Resize: {config['resize_mode']}",
        f"- Augmentation: {config['augmentation']}",
        f"- Sampling: {config.get('sampling_strategy', 'natural')}",
        f"- Split: {config['split_strategy']}",
        f"- Seed: {config['seed']}",
        f"- Early stopping patience: {config.get('early_stopping_patience', 'no aplicado')}",
        f"- Min delta checkpoint: {config.get('checkpoint_min_delta', 'no aplicado')}",
        "",
    ]
    if config["epochs"] <= 2 or experiment_dir.name.startswith("smoke"):
        lines.extend([f"> **Prueba funcional, no concluyente.** {SMOKE_WARNING}", ""])
    lines.extend(["## Validacion", "", table_markdown(pd.DataFrame([validation])), ""])
    if test:
        lines.extend(["## Test", "", table_markdown(pd.DataFrame([test])), ""])
    if benchmark:
        lines.extend(["## Benchmark", "", table_markdown(pd.DataFrame([benchmark])), ""])
    history_path = experiment_dir / "train_log.csv"
    if history_path.exists():
        history = pd.read_csv(history_path)
        min_loss_row = history.loc[history["valid_loss"].idxmin()]
        last_row = history.iloc[-1]
        lines.extend([
            "## Diagnostico de convergencia", "",
            f"- Menor loss de validacion: {min_loss_row['valid_loss']:.6f} en epoca {int(min_loss_row['epoch'])}.",
            f"- Loss de validacion final: {last_row['valid_loss']:.6f}.",
            f"- Dice de validacion final: {last_row['valid_dice']:.6f}.", "",
        ])
        if last_row["valid_loss"] > min_loss_row["valid_loss"] * 1.5:
            lines.extend([
                "**Advertencia:** la loss de validacion aumento mientras la loss de entrenamiento disminuyo. "
                "Existe evidencia de sobreajuste y debe priorizarse el checkpoint guardado, no la ultima epoca.", "",
            ])

    lines.extend(["## Advertencia metodologica", ""])
    if config["split_strategy"] == "coco":
        lines.append(
            "El split COCO comparte pacientes y videos entre train, valid y test; sus metricas pueden estar infladas."
        )
    elif config["split_strategy"] == "group_video":
        lines.append(
            "El split group_video evita compartir videos, pero mantiene pacientes entre splits. "
            "Con seed 42, test contiene un solo video de P001; es validacion interna y no sustituye P005."
        )
    else:
        lines.append(
            "El split group_patient separa pacientes, pero con tres pacientes cada split contiene solo uno y presenta alta varianza."
        )
    lines.extend(["Consultar 00_split_leakage_audit.md y 00_grouped_split_preview.md.", ""])
    (experiment_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def collect_experiments() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Consolida configuraciones, metricas y benchmarks."""
    index_rows, metric_rows, benchmark_rows = [], [], []
    if not EXPERIMENTS_ROOT.exists():
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    for directory in sorted(path for path in EXPERIMENTS_ROOT.iterdir() if path.is_dir()):
        config_path = directory / "config.json"
        if not config_path.exists():
            continue
        config = json.loads(config_path.read_text(encoding="utf-8"))
        index_row = {
            "experiment_name": directory.name,
            "class_name": config.get("class_name"),
            "architecture": config.get("architecture"),
            "epochs": config.get("epochs"),
            "image_size": config.get("image_size"),
            "resize_mode": config.get("resize_mode"),
            "augmentation": config.get("augmentation"),
            "sampling_strategy": config.get("sampling_strategy", "natural"),
            "pretrained": config.get("pretrained"),
            "split_strategy": config.get("split_strategy"),
            "seed": config.get("seed"),
            "early_stopping_patience": config.get("early_stopping_patience", 0),
            "checkpoint_min_delta": config.get("checkpoint_min_delta", 0.0),
            "timestamp": config.get("environment", {}).get("timestamp"),
            "experiment_path": str(directory),
            "result_status": (
                "preliminary_smoke" if int(config.get("epochs", 0)) <= 2
                else "controlled_experimental" if int(config.get("early_stopping_patience", 0)) > 0
                else "legacy_without_early_stopping"
            ),
        }
        index_rows.append(index_row)
        for split, filename in (("validation", "validation_metrics.csv"), ("test", "test_metrics.csv")):
            path = directory / filename
            if path.exists():
                row = pd.read_csv(path).iloc[0].to_dict()
                metric_rows.append({"experiment_name": directory.name, "evaluation_split": split, **index_row, **row})
        benchmark_path = directory / "benchmark_single_model.csv"
        if benchmark_path.exists():
            row = pd.read_csv(benchmark_path).iloc[0].to_dict()
            benchmark_rows.append({"experiment_name": directory.name, **row})
    return pd.DataFrame(index_rows), pd.DataFrame(metric_rows), pd.DataFrame(benchmark_rows)


def write_methodology_tables(index_df: pd.DataFrame, metrics_df: pd.DataFrame, benchmarks_df: pd.DataFrame) -> None:
    """Crea una base redactable para la seccion 3.3.5."""
    architectures = pd.DataFrame([
        {"Arquitectura": "U-Net", "Encoder": "ResNet-34", "Referencia": "Ronneberger et al. (2015)", "Uso": "Baseline biomedico"},
        {"Arquitectura": "DeepLabV3+", "Encoder": "ResNet-34", "Referencia": "Chen et al. (2018)", "Uso": "Contexto multiescala"},
        {"Arquitectura": "SegFormer", "Encoder": "MiT-B0", "Referencia": "Xie et al. (2021)", "Uso": "Transformer eficiente"},
    ])
    metric_table = pd.DataFrame([
        {"Metrica": "Dice/F1", "Formula": "2TP / (2TP + FP + FN)", "Uso": "Solapamiento"},
        {"Metrica": "IoU", "Formula": "TP / (TP + FP + FN)", "Uso": "Interseccion sobre union"},
        {"Metrica": "Precision", "Formula": "TP / (TP + FP)", "Uso": "Control de falsos positivos"},
        {"Metrica": "Recall", "Formula": "TP / (TP + FN)", "Uso": "Deteccion de positivos"},
        {"Metrica": "combined_la_score", "Formula": "Dice positivo - tasa FP en vacios", "Uso": "Seleccion de LA"},
    ])
    lines = [
        "# Tablas para la metodologia 3.3.5", "",
        "La segmentacion longitudinal se formula como tres problemas binarios independientes: ROI, Higado y lumen anecoico (LA).", "",
        "## Arquitecturas y encoders", "", table_markdown(architectures), "",
        "U-Net se usa como baseline por su diseno encoder-decoder y conexiones de salto en segmentacion biomedica.", "",
        "DeepLabV3+ captura contexto multiescala con convoluciones atrous y recupera bordes mediante un decoder.", "",
        "SegFormer representa la alternativa transformer eficiente; la implementacion usa exactamente MiT-B0.", "",
        "## Metricas y formulas", "", table_markdown(metric_table), "",
        "LA requiere Dice positivo y penalizacion de falsos positivos porque el Dice global puede favorecer mascaras vacias. El experimento balanced_la usa muestreo ponderado 50/50 durante train, sin modificar valid ni test.", "",
        "## Experimentos registrados", "", table_markdown(index_df), "",
        "## Resultados disponibles", "", table_markdown(metrics_df), "",
        "## Benchmarks disponibles", "", table_markdown(benchmarks_df), "",
        "## Justificaciones experimentales", "",
        "- Augmentation: solo train y conservadora para no crear anatomia irreal.",
        "- Resize: 128x128 puede acelerar inferencia, pero perder detalle de LA.",
        "- Transferencia: se compara ImageNet o MiT-B0 preentrenado con inicializacion aleatoria.",
        "- FPS: 30 fps equivale a 33.3 ms por frame y es la referencia de asistencia en tiempo real.", "",
        "> Los experimentos de 2 epocas son pruebas funcionales, no concluyentes.",
    ]
    (REPORTS_ROOT / "09_thesis_methodology_tables.md").write_text("\n".join(lines), encoding="utf-8")


def update_global_reports() -> None:
    """Actualiza indices acumulados sin inventar resultados faltantes."""
    REPORTS_ROOT.mkdir(parents=True, exist_ok=True)
    index_df, metrics_df, benchmarks_df = collect_experiments()
    index_df.to_csv(REPORTS_ROOT / "all_experiments_index.csv", index=False, encoding="utf-8-sig")
    metrics_df.to_csv(REPORTS_ROOT / "all_metrics.csv", index=False, encoding="utf-8-sig")
    benchmarks_df.to_csv(REPORTS_ROOT / "all_benchmarks.csv", index=False, encoding="utf-8-sig")

    smoke = index_df[index_df["epochs"] <= 2] if not index_df.empty else index_df
    smoke_lines = ["# Reporte de smoke tests", "", f"> **Prueba funcional, no concluyente.** {SMOKE_WARNING}", "", table_markdown(smoke)]
    (REPORTS_ROOT / "01_smoke_test_report.md").write_text("\n".join(smoke_lines), encoding="utf-8")

    test_metrics = metrics_df[metrics_df["evaluation_split"] == "test"].copy() if not metrics_df.empty else metrics_df
    validation_metrics = metrics_df[metrics_df["evaluation_split"] == "validation"].copy() if not metrics_df.empty else metrics_df
    best_rows = []
    if not test_metrics.empty:
        validation_columns = [
            "experiment_name", "checkpoint_score", "valid_loss", "valid_dice",
            "valid_positive_dice", "valid_empty_gt_false_positive_rate", "best_epoch",
        ]
        available_columns = [column for column in validation_columns if column in validation_metrics.columns]
        validation_selection = validation_metrics[available_columns].copy()
        validation_selection = validation_selection.rename(
            columns={column: f"selection_{column}" for column in available_columns if column != "experiment_name"}
        )
        candidates = test_metrics.merge(
            validation_selection,
            on="experiment_name",
            how="left",
        )
        if (candidates["epochs"] > 2).any():
            candidates = candidates[candidates["epochs"] > 2]
        controlled = pd.to_numeric(candidates["early_stopping_patience"], errors="coerce").fillna(0) > 0
        if controlled.any():
            candidates = candidates[controlled]
        for class_name, subset in candidates.groupby("class_name"):
            if class_name == "LA":
                subset = subset.sort_values(
                    ["selection_checkpoint_score", "selection_valid_positive_dice", "selection_valid_empty_gt_false_positive_rate"],
                    ascending=[False, False, True],
                )
            else:
                subset = subset.sort_values(
                    ["selection_checkpoint_score", "selection_valid_loss"],
                    ascending=[False, True],
                )
            selected = subset.iloc[0].to_dict()
            selected["selection_basis"] = (
                "combined_la_score de validacion; desempate con Dice positivo y FP vacios" if class_name == "LA"
                else "checkpoint_score de validacion"
            )
            best_rows.append(selected)
    best_frame = pd.DataFrame(best_rows)
    if not best_frame.empty:
        best_frame["selection_is_final"] = False
        best_frame["selection_note"] = "Seleccion provisional basada en validacion; test se conserva solo para evaluacion. Requiere P005."
    best_frame.to_csv(REPORTS_ROOT / "best_models_by_class.csv", index=False, encoding="utf-8-sig")

    report_specs = [
        ("02_unet_50ep_report.md", "U-Net 50 epocas", metrics_df[(metrics_df["architecture"] == "unet") & (metrics_df["epochs"] >= 50)] if not metrics_df.empty else metrics_df),
        ("03_architecture_comparison_report.md", "Comparacion de arquitecturas", metrics_df if not metrics_df.empty and metrics_df["architecture"].nunique() >= 2 else pd.DataFrame()),
        ("04_augmentation_comparison.md", "Comparacion de augmentation", metrics_df if not metrics_df.empty and metrics_df["augmentation"].nunique() >= 2 else pd.DataFrame()),
        ("05_resize_resolution_comparison.md", "Comparacion de resize y resolucion", metrics_df if not metrics_df.empty and metrics_df["resize_mode"].nunique() >= 2 else pd.DataFrame()),
        ("06_transfer_learning_comparison.md", "Comparacion de transferencia", metrics_df if not metrics_df.empty and metrics_df["pretrained"].nunique() >= 2 else pd.DataFrame()),
    ]
    for filename, title, subset in report_specs:
        lines = [f"# {title}", ""]
        if subset.empty:
            lines.append("Pendiente: no existen todavia experimentos comparables suficientes. No se generan conclusiones.")
        else:
            lines.append(table_markdown(subset))
        (REPORTS_ROOT / filename).write_text("\n".join(lines), encoding="utf-8")

    la_metrics = metrics_df[metrics_df["class_name"] == "LA"].copy() if not metrics_df.empty else pd.DataFrame()
    if not la_metrics.empty:
        la_valid = la_metrics[la_metrics["evaluation_split"] == "validation"].copy()
        la_test = la_metrics[la_metrics["evaluation_split"] == "test"].copy()
        comparison_columns_valid = [
            "experiment_name", "sampling_strategy", "selection_checkpoint_score",
            "valid_positive_dice", "valid_positive_iou", "valid_positive_recall",
            "valid_empty_gt_false_positive_rate", "valid_combined_la_score",
        ]
        comparison_columns_test = [
            "experiment_name", "test_positive_dice", "test_positive_iou",
            "test_positive_precision", "test_positive_recall",
            "test_empty_gt_false_positive_rate", "test_positive_mean_gt_area_px",
            "test_positive_mean_pred_area_px", "test_combined_la_score",
        ]
        valid_available = [column for column in comparison_columns_valid if column in la_valid.columns]
        test_available = [column for column in comparison_columns_test if column in la_test.columns]
        sampling_comparison = la_valid[valid_available].merge(
            la_test[test_available], on="experiment_name", how="inner"
        )
        sampling_comparison.to_csv(
            REPORTS_ROOT / "02_la_sampling_comparison.csv", index=False, encoding="utf-8-sig"
        )
        sampling_lines = [
            "# Comparacion de muestreo para LA", "",
            "La seleccion se realiza con combined_la_score de validacion. Test se reporta sin usarse para elegir el modelo.", "",
            table_markdown(sampling_comparison), "",
            "El muestreo balanceado busca aumentar deteccion positiva; debe revisarse junto con la tasa de falsos positivos en imagenes vacias.",
        ]
        (REPORTS_ROOT / "02_la_sampling_comparison.md").write_text(
            "\n".join(sampling_lines), encoding="utf-8"
        )

    pipeline_benchmark_path = REPORTS_ROOT / "07_inference_benchmark.csv"
    pipeline_benchmark = (
        pd.read_csv(pipeline_benchmark_path)
        if pipeline_benchmark_path.exists()
        else pd.DataFrame()
    )
    external_summary_path = REPORTS_ROOT / "10_external_validation_p005_summary.csv"
    external_summary = (
        pd.read_csv(external_summary_path)
        if external_summary_path.exists()
        else pd.DataFrame()
    )
    final_lines = [
        "# Resumen experimental acumulado", "",
        f"Experimentos registrados: {len(index_df)}.", "",
        "Las corridas controladas de hasta 50 epocas ya fueron ejecutadas. La seleccion actual es provisional "
        "porque se basa en validacion interna group_video.", "",
        "## Modelos seleccionados por validacion", "", table_markdown(best_frame), "",
        "## Evaluacion externa P005", "", table_markdown(external_summary), "",
        "P005 no dispone de mascaras ground truth. Sus resultados son predictivos y cualitativos; "
        "no constituyen Dice/IoU externos. Consultar 10_external_validation_p005_interpretation.md.", "",
        "## Indice", "", table_markdown(index_df), "",
        "## Metricas", "", table_markdown(metrics_df), "",
        "## Benchmarks individuales", "", table_markdown(benchmarks_df), "",
        "## Benchmark del pipeline de tres modelos", "", table_markdown(pipeline_benchmark),
    ]
    (REPORTS_ROOT / "08_final_experimental_summary.md").write_text("\n".join(final_lines), encoding="utf-8")
    write_methodology_tables(index_df, metrics_df, benchmarks_df)

    from src.architecture_comparison import generate_architecture_comparison
    generate_architecture_comparison(index_df, metrics_df, benchmarks_df)

    from src.augmentation_comparison import generate_augmentation_comparison
    generate_augmentation_comparison()

    from src.resize_comparison import generate_resize_comparison
    generate_resize_comparison()

    from src.transfer_comparison import generate_transfer_comparison
    generate_transfer_comparison()
