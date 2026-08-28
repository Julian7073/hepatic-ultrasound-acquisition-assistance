"""Comparacion de entrenamiento desde cero y transferencia."""

from __future__ import annotations

import json

import matplotlib.pyplot as plt
import pandas as pd

from config_experimental import EXPERIMENTS_ROOT, FIGURES_ROOT, REPORTS_ROOT
from src.reports import table_markdown


def generate_transfer_comparison() -> pd.DataFrame:
    """Consolida pares comparables con y sin pesos preentrenados."""
    rows = []
    for experiment_dir in sorted(path for path in EXPERIMENTS_ROOT.iterdir() if path.is_dir()):
        paths = {
            "config": experiment_dir / "config.json",
            "validation": experiment_dir / "validation_metrics.csv",
            "test": experiment_dir / "test_metrics.csv",
        }
        if not all(path.exists() for path in paths.values()):
            continue
        config = json.loads(paths["config"].read_text(encoding="utf-8"))
        expected_sampling = "balanced_la" if config.get("class_name") == "LA" else "natural"
        if not (
            config.get("architecture") in {"unet", "deeplabv3", "segformer"}
            and config.get("resize_mode") == "full_resize"
            and config.get("image_size") == 512
            and config.get("augmentation") == "none"
            and config.get("sampling_strategy", "natural") == expected_sampling
            and config.get("split_strategy") == "group_video"
            and int(config.get("early_stopping_patience", 0)) > 0
        ):
            continue
        validation = pd.read_csv(paths["validation"]).iloc[0]
        test = pd.read_csv(paths["test"]).iloc[0]
        rows.append({
            "experiment_name": experiment_dir.name,
            "class_name": config["class_name"],
            "architecture": config["architecture"],
            "pretrained": bool(config.get("pretrained")),
            "encoder": config.get("model_metadata", {}).get("encoder"),
            "encoder_weights": config.get("model_metadata", {}).get("encoder_weights"),
            "fine_tuning": config.get("model_metadata", {}).get("fine_tuning"),
            "best_epoch": validation["best_epoch"],
            "selection_score": validation["checkpoint_score"],
            "valid_dice": validation["valid_dice"],
            "valid_positive_dice": validation["valid_positive_dice"],
            "valid_empty_gt_false_positive_rate": validation["valid_empty_gt_false_positive_rate"],
            "test_dice": test["test_dice"],
            "test_positive_dice": test["test_positive_dice"],
            "test_positive_recall": test["test_positive_recall"],
            "test_empty_gt_false_positive_rate": test["test_empty_gt_false_positive_rate"],
            "test_combined_la_score": test["test_combined_la_score"],
        })

    all_rows = pd.DataFrame(rows)
    comparable = pd.DataFrame()
    if not all_rows.empty:
        matched_groups = []
        for _, subset in all_rows.groupby(["class_name", "architecture"]):
            if subset["pretrained"].nunique() == 2:
                matched_groups.append(subset)
        if matched_groups:
            comparable = pd.concat(matched_groups, ignore_index=True)
            comparable = comparable.sort_values(
                ["class_name", "architecture", "pretrained"]
            )

    comparable.to_csv(
        REPORTS_ROOT / "06_transfer_learning_comparison.csv",
        index=False,
        encoding="utf-8-sig",
    )
    lines = [
        "# Comparacion de transferencia de aprendizaje", "",
        "Los pares mantienen clase, arquitectura, split, resize, sampling y augmentation. "
        "Solo cambia la inicializacion del encoder.", "",
        "U-Net y DeepLabV3+ usan ResNet-34 con pesos ImageNet cuando pretrained=true. "
        "SegFormer usa MiT-B0 y, cuando corresponde, pesos nvidia/mit-b0. "
        "En todos los casos se realiza fine-tuning completo sin congelar el encoder.", "",
        "En SegFormer preentrenado se reutiliza el encoder MiT-B0. El decoder de segmentacion y el cabezal "
        "binario se inicializan nuevamente porque el checkpoint nvidia/mit-b0 no contiene esos pesos para "
        "esta tarea. Los mensajes MISSING/UNEXPECTED de Transformers durante esa carga son esperados.", "",
    ]
    if comparable.empty:
        lines.append("Pendiente: no existen pares completos con/sin pretraining.")
    else:
        lines.extend([
            table_markdown(comparable), "",
            "## Criterio", "",
            "ROI e Higado se comparan mediante Dice de validacion. LA se compara mediante combined_la_score de validacion, "
            "considerando por separado Dice positivo y falsos positivos en imagenes vacias.", "",
            "Las diferencias pequenas observadas con una sola semilla se consideran indicativas, no concluyentes.",
        ])
        labels = comparable.apply(
            lambda row: f"{row['architecture']}-{row['class_name']}", axis=1
        )
        plot_data = comparable.assign(pair=labels)
        pivot = plot_data.pivot(
            index="pair", columns="pretrained", values="selection_score"
        )
        pivot.columns = ["sin_pretraining" if not value else "con_pretraining" for value in pivot.columns]
        figure, axis = plt.subplots(figsize=(8, 4.5), dpi=160)
        pivot.plot(kind="bar", ax=axis)
        axis.set_title("Metrica de seleccion en validacion")
        axis.set_xlabel("Arquitectura y clase")
        axis.set_ylabel("Dice o combined_la_score")
        axis.grid(True, axis="y", alpha=0.3)
        figure.tight_layout()
        figure.savefig(FIGURES_ROOT / "06_transfer_learning_comparison.png")
        plt.close(figure)

    (REPORTS_ROOT / "06_transfer_learning_comparison.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    return comparable
