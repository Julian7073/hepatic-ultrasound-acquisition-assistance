"""Analiza resultados de aceptabilidad longitudinal y genera reporte base.

Entradas:
    outputs/metrics/glcm_longitudinal_metrics_with_thresholds.csv
    outputs/reports/quality_group_summary.csv

Salidas:
    outputs/reports/longitudinal_acceptability_analysis.csv
    outputs/reports/longitudinal_acceptability_summary.md
    outputs/figures/acceptance_rate_by_quality.png
    outputs/figures/acceptance_rate_by_patient.png
    outputs/figures/acceptance_rate_by_split.png
    outputs/reports/longitudinal_acceptability_examples.csv
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


TESIS_ROOT = Path(os.environ.get("THESIS_PROJECT_ROOT", Path(__file__).resolve().parents[1]))
METRICS_CSV = TESIS_ROOT / "outputs" / "metrics" / "glcm_longitudinal_metrics_with_thresholds.csv"
QUALITY_SUMMARY_CSV = TESIS_ROOT / "outputs" / "reports" / "quality_group_summary.csv"
REPORTS_DIR = TESIS_ROOT / "outputs" / "reports"
FIGURES_DIR = TESIS_ROOT / "outputs" / "figures"

ACCEPTED_COL = "acceptable_lumen_threshold_rule"


def load_metrics() -> pd.DataFrame:
    """Carga metricas y normaliza columnas necesarias."""
    if not METRICS_CSV.exists():
        raise FileNotFoundError(f"No existe {METRICS_CSV}")
    if not QUALITY_SUMMARY_CSV.exists():
        raise FileNotFoundError(f"No existe {QUALITY_SUMMARY_CSV}")

    df = pd.read_csv(METRICS_CSV)
    df[ACCEPTED_COL] = pd.to_numeric(df[ACCEPTED_COL], errors="coerce").fillna(0).astype(int)
    df["has_la"] = pd.to_numeric(df["has_la"], errors="coerce").fillna(0).astype(int)
    df["patient"] = df["patient"].fillna("unknown")
    df["quality"] = df["quality"].fillna("unknown")
    df["split"] = df["split"].fillna("unknown")
    return df


def summarize_by(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """Crea tabla de aceptabilidad por una variable."""
    summary = (
        df.groupby(group_col, dropna=False)
        .agg(
            total_images=("filename", "count"),
            images_with_LA=("has_la", "sum"),
            accepted=(ACCEPTED_COL, "sum"),
        )
        .reset_index()
        .rename(columns={group_col: "group"})
    )
    summary["rejected"] = summary["total_images"] - summary["accepted"]
    summary["acceptance_rate"] = summary["accepted"] / summary["total_images"]
    summary["acceptance_rate_with_LA"] = np.where(
        summary["images_with_LA"] > 0,
        summary["accepted"] / summary["images_with_LA"],
        np.nan,
    )
    summary.insert(0, "section", f"by_{group_col}")
    return summary


def plot_acceptance(summary: pd.DataFrame, title: str, output_path: Path) -> None:
    """Genera grafica de barras de tasa de aceptacion."""
    data = summary.sort_values("group")
    plt.figure(figsize=(8, 4.8))
    bars = plt.bar(data["group"].astype(str), data["acceptance_rate"] * 100, color="#2E74B5")
    plt.title(title)
    plt.ylabel("Tasa de aceptacion (%)")
    plt.xlabel("")
    plt.ylim(0, max(5, float((data["acceptance_rate"] * 100).max()) + 5))
    plt.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, data["acceptance_rate"] * 100):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.6, f"{value:.1f}%", ha="center", fontsize=9)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=160)
    plt.close()


def choose_examples(df: pd.DataFrame) -> pd.DataFrame:
    """Selecciona ejemplos representativos para revision cualitativa."""
    accepted = df[df[ACCEPTED_COL] == 1].copy()
    rejected = df[df[ACCEPTED_COL] == 0].copy()

    clear_accepted = accepted[accepted["quality"] == "clear"].sort_values(
        ["la_std_intensity", "glcm_entropy"], ascending=[True, True]
    ).head(5)
    clear_accepted = clear_accepted.assign(example_group="clear_accepted", example_reason="clear aceptado por regla")

    medium_pool = df[df["quality"] == "medium"].copy()
    medium_accepted = medium_pool[medium_pool[ACCEPTED_COL] == 1].sort_values("glcm_entropy").head(3)
    medium_rejected = medium_pool[medium_pool[ACCEPTED_COL] == 0].sort_values("glcm_entropy", ascending=False).head(2)
    medium_examples = pd.concat([medium_accepted, medium_rejected], ignore_index=True)
    medium_examples = medium_examples.assign(
        example_group="medium_mixed",
        example_reason="medium aceptado/rechazado para comparar comportamiento de la regla",
    )

    blurry_rejected = rejected[rejected["quality"] == "blurry"].sort_values(
        ["has_la", "glcm_entropy"], ascending=[True, False]
    ).head(5)
    blurry_rejected = blurry_rejected.assign(example_group="blurry_rejected", example_reason="blurry rechazado")

    thresholds = {
        "std": df["la_std_intensity"].dropna().median(),
        "entropy": df["glcm_entropy"].dropna().median(),
    }
    doubtful_pool = df[df["has_la"] == 1].copy()
    doubtful_pool["borderline_score"] = (
        (doubtful_pool["la_std_intensity"] - thresholds["std"]).abs().rank(pct=True)
        + (doubtful_pool["glcm_entropy"] - thresholds["entropy"]).abs().rank(pct=True)
    )
    doubtful = doubtful_pool.sort_values("borderline_score").head(5)
    doubtful = doubtful.assign(
        example_group="doubtful_cases",
        example_reason="caso dudoso: LA existe y las metricas quedan cerca de valores centrales",
    )

    examples = pd.concat([clear_accepted, medium_examples, blurry_rejected, doubtful], ignore_index=True)
    columns = [
        "example_group",
        "example_reason",
        "split",
        "patient",
        "quality",
        "filename",
        "has_la",
        ACCEPTED_COL,
        "la_area_px",
        "la_std_intensity",
        "glcm_entropy",
        "glcm_contrast",
        "glcm_homogeneity",
        "glcm_energy",
        "notes",
    ]
    return examples[columns]


def to_markdown_table(df: pd.DataFrame, value_cols: list[str]) -> str:
    """Convierte un DataFrame pequeno a tabla Markdown."""
    out = df.copy()
    for col in ["acceptance_rate", "acceptance_rate_with_LA"]:
        if col in out.columns:
            out[col] = (out[col] * 100).round(2).astype(str) + "%"
    out = out[value_cols].fillna("")
    header = "| " + " | ".join(value_cols) + " |"
    separator = "| " + " | ".join(["---"] * len(value_cols)) + " |"
    rows = []
    for _, row in out.iterrows():
        rows.append("| " + " | ".join(str(row[col]) for col in value_cols) + " |")
    return "\n".join([header, separator, *rows])


def write_summary_md(
    by_quality: pd.DataFrame,
    by_split: pd.DataFrame,
    by_patient: pd.DataFrame,
    examples: pd.DataFrame,
    total_images: int,
    with_la: int,
    accepted: int,
) -> None:
    """Escribe reporte Markdown tecnico y legible."""
    rejected = total_images - accepted
    content = []
    content.append("# Analisis de aceptabilidad longitudinal\n")
    content.append("Este reporte resume los resultados del pipeline longitudinal basado en mascaras COCO, dilatacion de LA, operador AND con ROI y metricas GLCM.\n")
    content.append("## Resumen general\n")
    content.append(f"- Imagenes analizadas: **{total_images}**")
    content.append(f"- Imagenes con LA anotado: **{with_la}**")
    content.append(f"- Imagenes aceptadas por la regla inicial: **{accepted}**")
    content.append(f"- Imagenes rechazadas por la regla inicial: **{rejected}**")
    content.append(f"- Tasa global de aceptacion: **{accepted / total_images * 100:.2f}%**\n")

    cols = ["group", "total_images", "images_with_LA", "accepted", "rejected", "acceptance_rate", "acceptance_rate_with_LA"]
    content.append("## Tabla por calidad\n")
    content.append(to_markdown_table(by_quality, cols))
    content.append("\n## Tabla por split\n")
    content.append(to_markdown_table(by_split, cols))
    content.append("\n## Tabla por paciente\n")
    content.append(to_markdown_table(by_patient, cols))

    content.append("\n## Ejemplos seleccionados\n")
    example_cols = [
        "example_group",
        "split",
        "patient",
        "quality",
        "has_la",
        ACCEPTED_COL,
        "la_area_px",
        "la_std_intensity",
        "glcm_entropy",
        "filename",
    ]
    content.append(to_markdown_table(examples, example_cols))

    content.append("\n## Interpretacion inicial\n")
    content.append("- La aceptacion depende de que exista LA anotado y de que cumpla los umbrales de area, desviacion estandar y entropia.")
    content.append("- Las imagenes sin LA anotado se consideran no aceptables para esta regla, porque no hay lumen evaluable.")
    content.append("- Esta regla es inicial y debe revisarse visualmente contra overlays y casos dudosos antes de presentarla como criterio final.")

    output_md = REPORTS_DIR / "longitudinal_acceptability_summary.md"
    output_md.write_text("\n".join(content), encoding="utf-8")


def main() -> None:
    """Ejecuta todo el analisis de aceptabilidad."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    df = load_metrics()
    by_quality = summarize_by(df, "quality")
    by_split = summarize_by(df, "split")
    by_patient = summarize_by(df, "patient")
    examples = choose_examples(df)

    plot_acceptance(by_quality, "Tasa de aceptacion por calidad", FIGURES_DIR / "acceptance_rate_by_quality.png")
    plot_acceptance(by_patient, "Tasa de aceptacion por paciente", FIGURES_DIR / "acceptance_rate_by_patient.png")
    plot_acceptance(by_split, "Tasa de aceptacion por split", FIGURES_DIR / "acceptance_rate_by_split.png")

    all_sections = pd.concat([by_quality, by_split, by_patient], ignore_index=True)
    examples_csv = examples.copy()
    examples_csv.insert(0, "section", "examples")
    analysis_csv = pd.concat([all_sections, examples_csv], ignore_index=True, sort=False)

    analysis_csv.to_csv(REPORTS_DIR / "longitudinal_acceptability_analysis.csv", index=False, encoding="utf-8-sig")
    examples.to_csv(REPORTS_DIR / "longitudinal_acceptability_examples.csv", index=False, encoding="utf-8-sig")
    write_summary_md(
        by_quality=by_quality,
        by_split=by_split,
        by_patient=by_patient,
        examples=examples,
        total_images=len(df),
        with_la=int(df["has_la"].sum()),
        accepted=int(df[ACCEPTED_COL].sum()),
    )

    print("Analisis longitudinal generado.")
    print(f"CSV: {REPORTS_DIR / 'longitudinal_acceptability_analysis.csv'}")
    print(f"Markdown: {REPORTS_DIR / 'longitudinal_acceptability_summary.md'}")


if __name__ == "__main__":
    main()

