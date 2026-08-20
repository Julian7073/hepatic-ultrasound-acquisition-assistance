"""Comparacion reproducible de U-Net, DeepLabV3+ y SegFormer."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from config_experimental import FIGURES_ROOT, REPORTS_ROOT
from src.reports import table_markdown


ARCHITECTURE_LABELS = {
    "unet": "U-Net",
    "deeplabv3": "DeepLabV3+",
    "segformer": "SegFormer",
}


def _numeric(frame: pd.DataFrame, columns: list[str]) -> None:
    """Convierte columnas metricas sin fallar por celdas vacias."""
    for column in columns:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")


def _save_metric_figure(comparison: pd.DataFrame, metric: str, output_name: str, title: str) -> None:
    """Genera barras por clase y arquitectura."""
    pivot = comparison.pivot(index="class_name", columns="architecture_label", values=metric)
    figure, axis = plt.subplots(figsize=(8, 4.5), dpi=160)
    pivot.plot(kind="bar", ax=axis)
    axis.set_title(title)
    axis.set_xlabel("Clase")
    axis.set_ylabel(metric)
    axis.set_ylim(0, max(1.0, float(pivot.max().max()) * 1.08))
    axis.grid(True, axis="y", alpha=0.3)
    axis.legend(title="Arquitectura")
    figure.tight_layout()
    figure.savefig(FIGURES_ROOT / output_name)
    plt.close(figure)


def generate_architecture_comparison(
    index_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    benchmarks_df: pd.DataFrame,
) -> pd.DataFrame:
    """Consolida las nueve corridas base comparables y genera informe."""
    REPORTS_ROOT.mkdir(parents=True, exist_ok=True)
    FIGURES_ROOT.mkdir(parents=True, exist_ok=True)
    output_csv = REPORTS_ROOT / "03_architecture_comparison.csv"
    output_md = REPORTS_ROOT / "03_architecture_comparison_report.md"

    if index_df.empty or metrics_df.empty:
        output_md.write_text(
            "# Comparacion de arquitecturas\n\nPendiente: no existen metricas suficientes.",
            encoding="utf-8",
        )
        return pd.DataFrame()

    validation = metrics_df[metrics_df["evaluation_split"] == "validation"].copy()
    test = metrics_df[metrics_df["evaluation_split"] == "test"].copy()
    validation_columns = [
        "experiment_name", "class_name", "architecture", "epochs", "image_size",
        "resize_mode", "augmentation", "sampling_strategy", "pretrained",
        "split_strategy", "early_stopping_patience",
        "best_epoch", "checkpoint_score", "valid_loss", "valid_dice", "valid_iou",
        "valid_precision", "valid_recall", "valid_positive_dice",
        "valid_positive_iou", "valid_positive_recall",
        "valid_empty_gt_false_positive_rate", "valid_combined_la_score",
    ]
    test_columns = [
        "experiment_name", "parameter_count", "test_loss", "test_dice", "test_iou",
        "test_precision", "test_recall", "test_positive_dice",
        "test_positive_iou", "test_positive_precision", "test_positive_recall",
        "test_empty_gt_false_positive_rate", "test_positive_mean_gt_area_px",
        "test_positive_mean_pred_area_px", "test_combined_la_score",
    ]
    valid_available = [column for column in validation_columns if column in validation.columns]
    test_available = [column for column in test_columns if column in test.columns]
    comparison = validation[valid_available].merge(
        test[test_available], on="experiment_name", how="inner"
    )
    benchmark_columns = [
        "experiment_name", "mean_ms_per_frame", "median_ms_per_frame",
        "p95_ms_per_frame", "fps",
    ]
    benchmark_available = [column for column in benchmark_columns if column in benchmarks_df.columns]
    if benchmark_available:
        comparison = comparison.merge(
            benchmarks_df[benchmark_available], on="experiment_name", how="left"
        )

    _numeric(comparison, [
        "epochs", "image_size", "early_stopping_patience", "parameter_count",
        "best_epoch", "checkpoint_score", "valid_loss", "valid_dice", "valid_iou",
        "valid_positive_dice", "valid_positive_iou",
        "valid_empty_gt_false_positive_rate", "test_dice", "test_iou",
        "test_positive_dice", "test_positive_iou", "test_positive_recall",
        "test_empty_gt_false_positive_rate", "mean_ms_per_frame", "fps",
    ])

    base_filter = (
        (comparison["epochs"] >= 50)
        & (comparison["image_size"] == 512)
        & (comparison["resize_mode"] == "full_resize")
        & (comparison["augmentation"] == "none")
        & (comparison["pretrained"].astype(str).str.lower().isin(["false", "0"]))
        & (comparison["split_strategy"] == "group_video")
        & (comparison["early_stopping_patience"] > 0)
        & (comparison["architecture"].isin(ARCHITECTURE_LABELS))
    )
    comparison = comparison[base_filter].copy()
    sampling_ok = (
        ((comparison["class_name"] == "LA") & (comparison["sampling_strategy"] == "balanced_la"))
        | ((comparison["class_name"] != "LA") & (comparison["sampling_strategy"] == "natural"))
    )
    comparison = comparison[sampling_ok].copy()
    comparison = comparison.sort_values(
        ["class_name", "architecture", "checkpoint_score", "valid_loss"],
        ascending=[True, True, False, True],
    ).drop_duplicates(["class_name", "architecture"], keep="first")
    comparison["architecture_label"] = comparison["architecture"].map(ARCHITECTURE_LABELS)
    comparison["primary_validation_metric"] = comparison.apply(
        lambda row: row["valid_positive_dice"] if row["class_name"] == "LA" else row["valid_dice"],
        axis=1,
    )
    comparison["selection_metric"] = comparison.apply(
        lambda row: "combined_la_score" if row["class_name"] == "LA" else "Dice",
        axis=1,
    )
    comparison = comparison.sort_values(["class_name", "checkpoint_score"], ascending=[True, False])
    comparison.to_csv(output_csv, index=False, encoding="utf-8-sig")

    if not comparison.empty:
        _save_metric_figure(
            comparison, "valid_dice", "03_valid_dice_by_architecture_class.png",
            "Dice de validacion por arquitectura y clase",
        )
        _save_metric_figure(
            comparison, "primary_validation_metric",
            "03_primary_validation_metric_by_architecture_class.png",
            "Metrica primaria de validacion: Dice global o Dice positivo LA",
        )
        _save_metric_figure(
            comparison, "test_dice", "03_test_dice_by_architecture_class.png",
            "Dice de test por arquitectura y clase",
        )
        _save_metric_figure(
            comparison, "fps", "03_fps_by_architecture_class.png",
            "Velocidad de inferencia por arquitectura y clase",
        )

        parameters = comparison.groupby(
            ["architecture_label"], as_index=False
        )["parameter_count"].max()
        figure, axis = plt.subplots(figsize=(7, 4), dpi=160)
        parameters.plot(
            x="architecture_label", y="parameter_count", kind="bar",
            legend=False, ax=axis, color="#577590",
        )
        axis.set_title("Parametros entrenables por arquitectura")
        axis.set_xlabel("")
        axis.set_ylabel("Parametros")
        axis.grid(True, axis="y", alpha=0.3)
        figure.tight_layout()
        figure.savefig(FIGURES_ROOT / "03_parameter_count_by_architecture.png")
        plt.close(figure)

        la = comparison[comparison["class_name"] == "LA"].copy()
        if not la.empty:
            figure, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=160)
            la.plot(
                x="architecture_label", y="test_positive_dice", kind="bar",
                legend=False, ax=axes[0], color="#277DA1",
            )
            axes[0].set_title("Dice positivo LA en test")
            axes[0].set_xlabel("")
            axes[0].set_ylim(0, 1)
            la.plot(
                x="architecture_label", y="test_empty_gt_false_positive_rate",
                kind="bar", legend=False, ax=axes[1], color="#F94144",
            )
            axes[1].set_title("Falsos positivos en imagenes vacias")
            axes[1].set_xlabel("")
            axes[1].set_ylim(0, 1)
            for axis in axes:
                axis.grid(True, axis="y", alpha=0.3)
            figure.tight_layout()
            figure.savefig(FIGURES_ROOT / "03_la_positive_dice_and_fp_rate.png")
            plt.close(figure)

    winners = []
    for class_name, subset in comparison.groupby("class_name"):
        best = subset.sort_values(
            ["checkpoint_score", "valid_loss"], ascending=[False, True]
        ).iloc[0]
        winners.append({
            "class_name": class_name,
            "selected_architecture": best["architecture_label"],
            "experiment_name": best["experiment_name"],
            "selection_metric": best["selection_metric"],
            "selection_score": best["checkpoint_score"],
            "test_dice": best["test_dice"],
            "test_positive_dice": best.get("test_positive_dice"),
            "fps": best.get("fps"),
        })
    winners_df = pd.DataFrame(winners)

    lines = [
        "# Comparacion base de arquitecturas", "",
        "Se comparan corridas con 512x512, sin augmentation, sin pretraining, split por video y early stopping. "
        "ROI e Higado usan muestreo natural; LA usa balanced_la.", "",
        "La seleccion usa exclusivamente validacion: Dice para ROI/Higado y combined_la_score para LA. "
        "Las metricas de test no se usan para elegir modelos.", "",
        f"Combinaciones comparables encontradas: {len(comparison)} de 9.", "",
        "## Resultados completos", "", table_markdown(comparison), "",
        "## Ganadores provisionales", "", table_markdown(winners_df), "",
        "## Interpretacion", "",
        "- ROI: se prioriza Dice de validacion, seguido de IoU y velocidad.",
        "- Higado: se revisan Dice, IoU, precision/recall y overlays.",
        "- LA: se prioriza combined_la_score, Dice positivo y baja tasa de falsos positivos en vacias. El Dice global de LA no se usa para seleccionar porque esta inflado por mascaras vacias.",
        "- SegFormer se entreno desde cero como MiT-B0; su menor numero de parametros no garantiza mayor velocidad.",
        "",
        "## Limitaciones", "",
        "El split group_video evita compartir videos, pero conserva pacientes entre splits. "
        "El test interno contiene un solo video de P001. Estos resultados son experimentales y requieren validacion externa con P005.",
    ]
    output_md.write_text("\n".join(lines), encoding="utf-8")
    return comparison
