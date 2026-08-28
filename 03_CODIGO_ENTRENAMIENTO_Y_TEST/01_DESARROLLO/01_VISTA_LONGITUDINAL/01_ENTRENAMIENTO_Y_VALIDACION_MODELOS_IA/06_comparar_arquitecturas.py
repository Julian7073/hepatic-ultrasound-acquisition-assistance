"""Compara arquitecturas y selecciona mejores modelos por clase."""

from __future__ import annotations

import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from config_segmentation import FINAL_MODELS_ROOT, FIGURES_ROOT, METRICS_ROOT, REPORTS_ROOT, TARGET_CLASSES, ensure_output_dirs


def numeric(series: pd.Series) -> pd.Series:
    """Convierte una columna a numerica de forma robusta."""
    return pd.to_numeric(series, errors="coerce").fillna(0.0)


def is_canonical_checkpoint(row: pd.Series) -> bool:
    """Identifica el checkpoint activo, no una copia de respaldo."""
    expected = f"{row['architecture']}_{str(row['class_name']).lower()}_best.pth"
    return Path(str(row["checkpoint_path"])).name == expected


def prepare_ranking(df: pd.DataFrame) -> pd.DataFrame:
    """Consolida duplicados y calcula el orden metodologico por clase."""
    required_numeric = [
        "test_dice",
        "test_iou",
        "test_positive_dice",
        "test_positive_iou",
        "empty_gt_false_positive_rate",
        "inference_time_s_per_frame",
    ]
    for column in required_numeric:
        if column not in df.columns:
            df[column] = 0.0
        df[column] = numeric(df[column])

    df["_canonical_checkpoint"] = df.apply(is_canonical_checkpoint, axis=1)
    df["_primary_score"] = df.apply(
        lambda row: row["test_positive_dice"] if row["class_name"] == "LA" else row["test_dice"],
        axis=1,
    )
    df["_secondary_score"] = df.apply(
        lambda row: row["test_positive_iou"] if row["class_name"] == "LA" else row["test_iou"],
        axis=1,
    )
    df["_fp_penalty"] = df.apply(
        lambda row: row["empty_gt_false_positive_rate"] if row["class_name"] == "LA" else 0.0,
        axis=1,
    )

    # Primero se consolida arquitectura/clase para evitar que copias de respaldo aparezcan duplicadas.
    deduped = df.sort_values(
        [
            "class_name",
            "architecture",
            "_canonical_checkpoint",
            "_primary_score",
            "_secondary_score",
            "_fp_penalty",
            "inference_time_s_per_frame",
        ],
        ascending=[True, True, False, False, False, True, True],
    ).drop_duplicates(subset=["architecture", "class_name"], keep="first")

    ranked = deduped.sort_values(
        ["class_name", "_primary_score", "_secondary_score", "_fp_penalty", "inference_time_s_per_frame"],
        ascending=[True, False, False, True, True],
    ).reset_index(drop=True)
    return ranked


def public_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Elimina columnas internas antes de guardar reportes."""
    return df.drop(columns=[column for column in df.columns if column.startswith("_")], errors="ignore")


def main() -> None:
    """Genera ranking y copia mejores modelos por clase."""
    ensure_output_dirs()
    metrics_path = METRICS_ROOT / "test_metrics_all_available.csv"
    if not metrics_path.exists():
        raise FileNotFoundError(f"No existe {metrics_path}. Ejecuta 05_evaluar_modelos.py primero.")

    df = pd.read_csv(metrics_path)
    if df.empty:
        raise ValueError("La tabla de metricas test esta vacia.")

    ranked = prepare_ranking(df)
    ranking_path = REPORTS_ROOT / "architecture_ranking_by_class.csv"
    public_columns(ranked).to_csv(ranking_path, index=False, encoding="utf-8-sig")

    best_rows = []
    for class_name in TARGET_CLASSES:
        subset = ranked[ranked["class_name"] == class_name]
        if subset.empty:
            continue
        best = subset.iloc[0].to_dict()
        best_rows.append(best)
        source = Path(str(best["checkpoint_path"]))
        destination = FINAL_MODELS_ROOT / f"best_{class_name.lower()}_model.pth"
        if source.exists():
            shutil.copy2(source, destination)

    best_df = public_columns(pd.DataFrame(best_rows))
    best_path = REPORTS_ROOT / "best_models_by_class.csv"
    best_df.to_csv(best_path, index=False, encoding="utf-8-sig")

    plot_df = public_columns(ranked)
    for metric in ["test_dice", "test_iou", "test_positive_dice", "inference_time_s_per_frame"]:
        if metric not in plot_df.columns:
            continue
        fig, ax = plt.subplots(figsize=(8, 4), dpi=150)
        pivot = plot_df.pivot(index="class_name", columns="architecture", values=metric)
        pivot.plot(kind="bar", ax=ax)
        ax.set_title(metric)
        ax.set_xlabel("Clase")
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(FIGURES_ROOT / f"comparison_{metric}.png")
        plt.close(fig)

    print(f"Ranking guardado: {ranking_path}")
    print(f"Mejores modelos: {best_path}")
    print(f"Modelos finales copiados en: {FINAL_MODELS_ROOT}")


if __name__ == "__main__":
    main()