"""Define umbrales iniciales de aceptabilidad para lumen longitudinal."""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from config import FIGURES_ROOT, METRICS_ROOT, REPORTS_ROOT, ensure_output_dirs


METRIC_COLUMNS = ["la_std_intensity", "glcm_entropy", "glcm_contrast", "glcm_homogeneity", "glcm_energy"]


def boxplot_by_quality(df: pd.DataFrame, column: str, output_name: str) -> None:
    """Genera boxplot por calidad."""
    data = [df.loc[df["quality"] == quality, column].dropna().values for quality in ["clear", "medium", "blurry"]]
    plt.figure(figsize=(8, 5))
    plt.boxplot(data, tick_labels=["clear", "medium", "blurry"], showmeans=True)
    plt.title(f"{column} por calidad")
    plt.xlabel("Calidad")
    plt.ylabel(column)
    plt.tight_layout()
    plt.savefig(FIGURES_ROOT / output_name, dpi=150)
    plt.close()


def main() -> None:
    """Calcula umbrales usando clear de train/valid como referencia."""
    ensure_output_dirs()
    input_csv = METRICS_ROOT / "glcm_longitudinal_metrics.csv"
    if not input_csv.exists():
        raise FileNotFoundError(f"No existe {input_csv}. Ejecuta primero 04_pipeline_glcm_longitudinal.py")

    df = pd.read_csv(input_csv)
    df_dev = df[df["split"].isin(["train", "valid"])].copy()
    reference = df_dev[(df_dev["quality"] == "clear") & (df_dev["has_la"] == 1) & (df_dev["and_area_px"] > 0)].copy()

    if reference.empty:
        raise ValueError("No hay imagenes clear con LA para calcular umbrales.")

    threshold_std = float(np.nanpercentile(reference["la_std_intensity"], 90))
    threshold_entropy = float(np.nanpercentile(reference["glcm_entropy"], 90))
    min_area = int(max(20, np.nanpercentile(reference["la_area_px"], 5)))

    threshold_rows = [
        {"parameter": "min_la_area_px", "value": min_area, "method": "max(20, percentil_5_clear_train_valid)"},
        {"parameter": "max_la_std_intensity", "value": threshold_std, "method": "percentil_90_clear_train_valid"},
        {"parameter": "max_glcm_entropy", "value": threshold_entropy, "method": "percentil_90_clear_train_valid"},
    ]

    df["acceptable_lumen_threshold_rule"] = (
        (df["has_la"] == 1)
        & (df["la_area_px"] >= min_area)
        & (df["la_std_intensity"] <= threshold_std)
        & (df["glcm_entropy"] <= threshold_entropy)
    ).astype(int)

    group_summary = (
        df.groupby(["split", "quality"], dropna=False)
        .agg(
            n=("filename", "count"),
            has_la_rate=("has_la", "mean"),
            acceptable_rate=("acceptable_lumen_threshold_rule", "mean"),
            la_area_mean=("la_area_px", "mean"),
            la_std_mean=("la_std_intensity", "mean"),
            entropy_mean=("glcm_entropy", "mean"),
            contrast_mean=("glcm_contrast", "mean"),
        )
        .reset_index()
    )

    pd.DataFrame(threshold_rows).to_csv(REPORTS_ROOT / "threshold_summary.csv", index=False, encoding="utf-8-sig")
    group_summary.to_csv(REPORTS_ROOT / "quality_group_summary.csv", index=False, encoding="utf-8-sig")
    df.to_csv(METRICS_ROOT / "glcm_longitudinal_metrics_with_thresholds.csv", index=False, encoding="utf-8-sig")

    boxplot_by_quality(df_dev, "la_std_intensity", "boxplot_std_by_quality.png")
    boxplot_by_quality(df_dev, "glcm_entropy", "boxplot_entropy_by_quality.png")
    boxplot_by_quality(df_dev, "glcm_contrast", "boxplot_contrast_by_quality.png")

    print("Umbrales iniciales generados:")
    for row in threshold_rows:
        print(f"  {row['parameter']}: {row['value']}")
    print(f"CSV: {REPORTS_ROOT / 'threshold_summary.csv'}")
    print(f"CSV: {REPORTS_ROOT / 'quality_group_summary.csv'}")


if __name__ == "__main__":
    main()

