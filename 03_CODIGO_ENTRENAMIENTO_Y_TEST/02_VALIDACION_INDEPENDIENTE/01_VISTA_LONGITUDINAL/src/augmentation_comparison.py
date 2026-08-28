"""Comparacion de estrategias de augmentation para U-Net/LA."""

from __future__ import annotations

import json

import matplotlib.pyplot as plt
import pandas as pd

from config_experimental import EXPERIMENTS_ROOT, FIGURES_ROOT, REPORTS_ROOT
from src.reports import table_markdown


def generate_augmentation_comparison() -> pd.DataFrame:
    """Compara configuraciones LA relevantes sin ocultar cambios de sampling."""
    rows = []
    for experiment_dir in sorted(
        path for path in EXPERIMENTS_ROOT.iterdir() if path.is_dir()
    ):
        paths = {
            "config": experiment_dir / "config.json",
            "validation": experiment_dir / "validation_metrics.csv",
            "test": experiment_dir / "test_metrics.csv",
            "log": experiment_dir / "train_log.csv",
        }
        if not all(path.exists() for path in paths.values()):
            continue
        config = json.loads(paths["config"].read_text(encoding="utf-8"))
        augmentation = config.get("augmentation")
        sampling = config.get("sampling_strategy", "natural")
        if not (
            config.get("class_name") == "LA"
            and config.get("architecture") == "unet"
            and config.get("resize_mode") == "full_resize"
            and config.get("image_size") == 512
            and config.get("split_strategy") == "group_video"
            and str(config.get("pretrained")).lower() in {"false", "0"}
            and augmentation in {"none", "x4", "positive_x4"}
            and sampling in {"natural", "balanced_la"}
            and int(config.get("epochs", 0)) > 2
        ):
            continue

        validation = pd.read_csv(paths["validation"]).iloc[0]
        test = pd.read_csv(paths["test"]).iloc[0]
        executed_epochs = len(pd.read_csv(paths["log"]))
        effective_samples = int(config.get("effective_train_samples", 0))
        augmentation_metadata = config.get("augmentation_metadata", {})
        rows.append({
            "experiment_name": experiment_dir.name,
            "strategy": f"{augmentation}+{sampling}",
            "augmentation": augmentation,
            "sampling_strategy": sampling,
            "configured_epochs": config["epochs"],
            "executed_epochs": executed_epochs,
            "base_positive_samples": augmentation_metadata.get("base_positive_samples"),
            "base_empty_samples": augmentation_metadata.get("base_empty_samples"),
            "effective_positive_samples": augmentation_metadata.get("effective_positive_samples"),
            "effective_empty_samples": augmentation_metadata.get("effective_empty_samples"),
            "effective_samples_per_epoch": effective_samples,
            "total_sample_draws": executed_epochs * effective_samples,
            "best_epoch": validation["best_epoch"],
            "valid_combined_la_score": validation["valid_combined_la_score"],
            "valid_positive_dice": validation["valid_positive_dice"],
            "valid_positive_recall": validation["valid_positive_recall"],
            "valid_empty_gt_false_positive_rate": validation[
                "valid_empty_gt_false_positive_rate"
            ],
            "test_positive_dice": test["test_positive_dice"],
            "test_positive_iou": test["test_positive_iou"],
            "test_positive_recall": test["test_positive_recall"],
            "test_empty_gt_false_positive_rate": test[
                "test_empty_gt_false_positive_rate"
            ],
            "test_combined_la_score": test["test_combined_la_score"],
        })

    comparison = pd.DataFrame(rows)
    output_csv = REPORTS_ROOT / "04_augmentation_comparison.csv"
    output_md = REPORTS_ROOT / "04_augmentation_comparison.md"
    comparison.to_csv(output_csv, index=False, encoding="utf-8-sig")

    lines = [
        "# Comparacion de augmentation para LA", "",
        "positive_x4 aumenta solo imagenes positivas y conserva una copia de cada imagen vacia. "
        "Se usa con muestreo natural para evitar doble rebalanceo. x4 aumenta todas las imagenes; "
        "los experimentos existentes pueden combinarlo con balanced_la.", "",
        "Validacion y test nunca reciben augmentation ni balanceo.", "",
    ]
    if comparison.empty:
        lines.append("Pendiente: no existen configuraciones comparables.")
    else:
        lines.extend([
            table_markdown(comparison), "",
            "## Interpretacion", "",
            "La seleccion se basa en combined_la_score de validacion, Dice positivo y tasa de falsos "
            "positivos en imagenes vacias.", "",
            "Esta tabla compara estrategias completas. Cuando sampling_strategy o numero de epocas "
            "difieren, el contraste no debe describirse como un experimento de una sola variable.", "",
        ])
        positive_rows = comparison[comparison["augmentation"] == "positive_x4"]
        if not positive_rows.empty:
            positive_row = positive_rows.sort_values(
                "valid_combined_la_score", ascending=False
            ).iloc[0]
            alternatives = comparison[comparison["augmentation"] != "positive_x4"]
            best_alternative = alternatives.sort_values(
                "valid_combined_la_score", ascending=False
            ).iloc[0]
            lines.extend([
                "## Resultado de positive_x4", "",
                (
                    f"positive_x4 obtuvo combined_la_score={positive_row['valid_combined_la_score']:.4f} "
                    f"en validacion, Dice positivo={positive_row['valid_positive_dice']:.4f} y "
                    f"tasa de falsos positivos en imagenes vacias="
                    f"{positive_row['valid_empty_gt_false_positive_rate']:.2%}."
                ), "",
                (
                    f"En test, la tasa de falsos positivos vacios fue "
                    f"{positive_row['test_empty_gt_false_positive_rate']:.2%} y el combined_la_score "
                    f"fue {positive_row['test_combined_la_score']:.4f}."
                ), "",
                (
                    f"No se selecciona esta estrategia. La mejor alternativa comparable por validacion "
                    f"es {best_alternative['strategy']} con combined_la_score="
                    f"{best_alternative['valid_combined_la_score']:.4f}."
                ), "",
                "El aumento exclusivo de positivos desplazo el modelo hacia predicciones de LA en imagenes "
                "sin anotacion. Este resultado negativo se conserva como evidencia experimental.", "",
            ])

        metrics = [
            "valid_combined_la_score",
            "test_positive_dice",
            "test_empty_gt_false_positive_rate",
        ]
        plot_data = comparison.sort_values("strategy")
        figure, axes = plt.subplots(1, 3, figsize=(15, 4.5), dpi=160)
        for axis, metric in zip(axes, metrics):
            plot_data.plot(
                x="strategy",
                y=metric,
                kind="bar",
                legend=False,
                ax=axis,
            )
            axis.set_title(metric)
            axis.set_xlabel("")
            axis.tick_params(axis="x", rotation=25)
            axis.grid(True, axis="y", alpha=0.3)
        axes[1].set_ylim(0, 1)
        axes[2].set_ylim(0, 1)
        figure.tight_layout()
        figure.savefig(FIGURES_ROOT / "04_la_augmentation_comparison.png")
        plt.close(figure)

    output_md.write_text("\n".join(lines), encoding="utf-8")
    return comparison
