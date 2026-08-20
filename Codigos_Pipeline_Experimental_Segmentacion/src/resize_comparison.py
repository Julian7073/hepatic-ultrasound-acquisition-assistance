"""Comparacion de estrategias de resize para U-Net/LA."""

from __future__ import annotations

import json

import matplotlib.pyplot as plt
import pandas as pd

from config_experimental import EXPERIMENTS_ROOT, FIGURES_ROOT, REPORTS_ROOT
from src.reports import table_markdown


def generate_resize_comparison() -> pd.DataFrame:
    """Compara full resize, crop ROI 128 y letterbox/padding."""
    rows = []
    for experiment_dir in sorted(path for path in EXPERIMENTS_ROOT.iterdir() if path.is_dir()):
        paths = {
            "config": experiment_dir / "config.json",
            "validation": experiment_dir / "validation_metrics.csv",
            "test": experiment_dir / "test_metrics.csv",
            "benchmark": experiment_dir / "benchmark_single_model.csv",
        }
        if not all(path.exists() for path in paths.values()):
            continue
        config = json.loads(paths["config"].read_text(encoding="utf-8"))
        if not (
            config.get("class_name") == "LA"
            and config.get("architecture") == "unet"
            and config.get("sampling_strategy") == "balanced_la"
            and config.get("augmentation") == "none"
            and config.get("split_strategy") == "group_video"
            and str(config.get("pretrained")).lower() in {"false", "0"}
            and config.get("resize_mode") in {
                "full_resize", "roi_crop_resize", "original_or_padding"
            }
        ):
            continue
        validation = pd.read_csv(paths["validation"]).iloc[0]
        test = pd.read_csv(paths["test"]).iloc[0]
        benchmark = pd.read_csv(paths["benchmark"]).iloc[0]
        rows.append({
            "experiment_name": experiment_dir.name,
            "resize_mode": config["resize_mode"],
            "image_size": config["image_size"],
            "best_epoch": validation["best_epoch"],
            "valid_combined_la_score": validation["valid_combined_la_score"],
            "valid_positive_dice": validation["valid_positive_dice"],
            "valid_positive_recall": validation["valid_positive_recall"],
            "valid_empty_gt_false_positive_rate": validation["valid_empty_gt_false_positive_rate"],
            "test_positive_dice": test["test_positive_dice"],
            "test_positive_iou": test["test_positive_iou"],
            "test_positive_recall": test["test_positive_recall"],
            "test_empty_gt_false_positive_rate": test["test_empty_gt_false_positive_rate"],
            "test_combined_la_score": test["test_combined_la_score"],
            "mean_ms_per_frame": benchmark["mean_ms_per_frame"],
            "fps": benchmark["fps"],
        })

    comparison = pd.DataFrame(rows)
    if not comparison.empty:
        comparison = comparison.sort_values(
            ["valid_combined_la_score"], ascending=False
        ).drop_duplicates(["resize_mode", "image_size"], keep="first")
    comparison.to_csv(
        REPORTS_ROOT / "05_resize_resolution_comparison.csv",
        index=False,
        encoding="utf-8-sig",
    )

    lines = [
        "# Comparacion de resize y resolucion para LA", "",
        "Se usa U-Net, balanced_la, sin augmentation, sin pretraining y split group_video.", "",
        "Configuraciones:", "",
        "- full_resize 512: toda la imagen se transforma a 512x512, con deformacion de proporcion.",
        "- roi_crop_resize 128: usa la caja de la ROI anotada y la redimensiona a 128x128; es un experimento oracle.",
        "- original_or_padding 512: ajusta el lado mayor a 512 y completa con padding negro hasta 512x512.", "",
    ]
    if comparison.empty or comparison["resize_mode"].nunique() < 3:
        lines.append("Pendiente: faltan configuraciones comparables.")
    else:
        lines.extend([
            table_markdown(comparison), "",
            "## Resultado", "",
            "La seleccion se basa en combined_la_score de validacion. "
            "El Dice global no se usa porque una prediccion completamente vacia puede obtener un valor alto cuando predominan mascaras GT vacias.", "",
            "full_resize 512 se conserva provisionalmente. roi_crop_resize 128 incrementa velocidad, pero produce demasiados falsos positivos. "
            "original_or_padding colapsa a mascaras vacias y no detecta LA.", "",
            "El recorte ROI utiliza anotacion GT; en una aplicacion real tendria que usar la ROI predicha.",
        ])
        metrics = [
            "valid_combined_la_score",
            "test_positive_dice",
            "test_empty_gt_false_positive_rate",
            "fps",
        ]
        figure, axes = plt.subplots(2, 2, figsize=(11, 8), dpi=160)
        for axis, metric in zip(axes.ravel(), metrics):
            comparison.plot(
                x="resize_mode", y=metric, kind="bar",
                legend=False, ax=axis,
            )
            axis.set_title(metric)
            axis.set_xlabel("")
            axis.grid(True, axis="y", alpha=0.3)
        axes[0, 1].set_ylim(0, 1)
        axes[1, 0].set_ylim(0, 1)
        figure.tight_layout()
        figure.savefig(FIGURES_ROOT / "05_la_resize_resolution_comparison.png")
        plt.close(figure)

    (REPORTS_ROOT / "05_resize_resolution_comparison.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    return comparison
