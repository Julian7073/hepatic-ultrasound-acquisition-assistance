"""Ajusta la regla de aceptabilidad usando evidencia de borde hiperecogenico.

La regla original acepta LA si cumple criterios de area, desviacion estandar
y entropia. Este script agrega una validacion visual-computable:
el lumen anecoico debe estar rodeado por un borde/pared con intensidad,
gradiente suficiente o contraste local claro entre lumen oscuro y entorno.

No modifica resultados previos. Genera nuevos CSV y un resumen Markdown.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from config import MASKS_ROOT, METRICS_ROOT, OUTPUTS_ROOT, REPORTS_ROOT, ROBOFLOW_COCO_ROOT, ensure_output_dirs
from utils.mask_utils import dilate_mask, load_binary_mask
from utils.metadata_utils import mask_filename_for_image


INPUT_METRICS = METRICS_ROOT / "glcm_longitudinal_metrics_with_thresholds.csv"
OUTPUT_METRICS = METRICS_ROOT / "glcm_longitudinal_metrics_with_border_rule.csv"
OUTPUT_SUMMARY = REPORTS_ROOT / "longitudinal_acceptability_border_rule_summary.csv"
OUTPUT_THRESHOLDS = REPORTS_ROOT / "border_rule_threshold_summary.csv"
OUTPUT_DOUBTFUL = REPORTS_ROOT / "longitudinal_doubtful_cases.csv"
OUTPUT_DOUBTFUL_REVIEW = REPORTS_ROOT / "longitudinal_doubtful_cases_review_list.csv"
OUTPUT_MARKDOWN = REPORTS_ROOT / "longitudinal_acceptability_rule_adjustment_summary.md"
OUTPUT_FIGURE = OUTPUTS_ROOT / "figures" / "doubtful_cases_top20.png"

REFERENCE_PATIENT = "P001"
REFERENCE_QUALITY = "clear"
REFERENCE_SPLITS = ["train", "valid"]
REFERENCE_QUANTILE = 0.15

# Anillo externo estrecho alrededor de LA. Debe ser impar por Pillow MaxFilter.
BORDER_RING_KERNEL_SIZE = 9


def load_gray_image(path: Path) -> np.ndarray:
    """Carga imagen en escala de grises sin cambiar contraste ni tamano."""
    with Image.open(path) as image:
        return np.array(image.convert("L"), dtype=np.float32)


def gradient_magnitude(gray: np.ndarray) -> np.ndarray:
    """Calcula una magnitud de gradiente simple y reproducible con NumPy."""
    gx = np.zeros_like(gray, dtype=np.float32)
    gy = np.zeros_like(gray, dtype=np.float32)
    gx[:, 1:-1] = (gray[:, 2:] - gray[:, :-2]) / 2.0
    gy[1:-1, :] = (gray[2:, :] - gray[:-2, :]) / 2.0
    return np.sqrt(gx * gx + gy * gy)


def compute_border_metrics(row: pd.Series) -> dict:
    """Calcula metricas del borde alrededor de LA para una imagen."""
    split = str(row["split"])
    filename = str(row["filename"])
    image_path = ROBOFLOW_COCO_ROOT / split / filename
    mask_name = mask_filename_for_image(filename)
    la_path = MASKS_ROOT / split / "LA" / mask_name
    roi_path = MASKS_ROOT / split / "ROI" / mask_name

    if int(row.get("has_la", 0)) != 1:
        return empty_border_metrics("sin_LA")
    if not image_path.exists():
        return empty_border_metrics("falta_imagen")
    if not la_path.exists() or not roi_path.exists():
        return empty_border_metrics("falta_mascara")

    gray = load_gray_image(image_path)
    expected_size = (gray.shape[1], gray.shape[0])
    la_mask = load_binary_mask(la_path, expected_size=expected_size)
    roi_mask = load_binary_mask(roi_path, expected_size=expected_size)

    if int(np.sum(la_mask > 0)) == 0:
        return empty_border_metrics("mascara_LA_vacia")

    la_dilated_small = dilate_mask(la_mask, kernel_size=BORDER_RING_KERNEL_SIZE)
    border_ring = ((la_dilated_small > 0) & (la_mask == 0) & (roi_mask > 0))

    if int(np.sum(border_ring)) == 0:
        return empty_border_metrics("anillo_borde_vacio")

    la_values = gray[la_mask > 0]
    border_values = gray[border_ring]
    border_gradient = gradient_magnitude(gray)[border_ring]
    la_median = float(np.median(la_values))
    la_p10 = float(np.percentile(la_values, 10))
    border_p90 = float(np.percentile(border_values, 90))

    return {
        "border_status": "ok",
        "border_area_px": int(np.sum(border_ring)),
        "la_median_intensity": la_median,
        "la_p10_intensity": la_p10,
        "border_mean_intensity": float(np.mean(border_values)),
        "border_p75_intensity": float(np.percentile(border_values, 75)),
        "border_p90_intensity": border_p90,
        "border_p90_minus_la_median": float(border_p90 - la_median),
        "border_p90_minus_la_p10": float(border_p90 - la_p10),
        "border_high_ratio_10": float(np.mean(border_values >= la_median + 10.0)),
        "border_high_ratio_20": float(np.mean(border_values >= la_median + 20.0)),
        "border_high_ratio_la_p10_20": float(np.mean(border_values >= la_p10 + 20.0)),
        "border_gradient_mean": float(np.mean(border_gradient)),
        "border_gradient_p75": float(np.percentile(border_gradient, 75)),
        "border_gradient_p90": float(np.percentile(border_gradient, 90)),
        "roboflow_image_path": str(image_path),
        "la_mask_path": str(la_path),
        "roi_mask_path": str(roi_path),
    }


def empty_border_metrics(status: str) -> dict:
    """Devuelve metricas vacias cuando no es posible analizar el borde."""
    return {
        "border_status": status,
        "border_area_px": 0,
        "la_median_intensity": np.nan,
        "la_p10_intensity": np.nan,
        "border_mean_intensity": np.nan,
        "border_p75_intensity": np.nan,
        "border_p90_intensity": np.nan,
        "border_p90_minus_la_median": np.nan,
        "border_p90_minus_la_p10": np.nan,
        "border_high_ratio_10": np.nan,
        "border_high_ratio_20": np.nan,
        "border_high_ratio_la_p10_20": np.nan,
        "border_gradient_mean": np.nan,
        "border_gradient_p75": np.nan,
        "border_gradient_p90": np.nan,
        "roboflow_image_path": "",
        "la_mask_path": "",
        "roi_mask_path": "",
    }


def derive_thresholds(df: pd.DataFrame) -> dict:
    """Deriva umbrales desde P001 clear de train/valid aceptados previamente."""
    reference = df[
        (df["patient"] == REFERENCE_PATIENT)
        & (df["quality"] == REFERENCE_QUALITY)
        & (df["split"].isin(REFERENCE_SPLITS))
        & (df["acceptable_lumen_threshold_rule"] == 1)
        & (df["border_status"] == "ok")
    ].copy()

    if reference.empty:
        raise ValueError("No hay casos de referencia P001 clear aceptados para calcular umbrales de borde.")

    thresholds = {
        "min_border_p90_minus_la_median": float(reference["border_p90_minus_la_median"].quantile(REFERENCE_QUANTILE)),
        "min_border_gradient_p75": float(reference["border_gradient_p75"].quantile(REFERENCE_QUANTILE)),
        "min_border_high_ratio_10": float(reference["border_high_ratio_10"].quantile(REFERENCE_QUANTILE)),
        "min_border_p90_minus_la_p10": float(reference["border_p90_minus_la_p10"].quantile(REFERENCE_QUANTILE)),
        "reference_count": int(len(reference)),
        "reference_patient": REFERENCE_PATIENT,
        "reference_quality": REFERENCE_QUALITY,
        "reference_splits": ",".join(REFERENCE_SPLITS),
        "reference_quantile": REFERENCE_QUANTILE,
    }
    return thresholds


def apply_adjusted_rules(df: pd.DataFrame, thresholds: dict) -> pd.DataFrame:
    """Aplica la regla ajustada de borde.

    Revision metodologica:
    - La evidencia de pared puede aparecer como borde brillante o como gradiente alto.
    - No se exige que ambas condiciones ocurran al mismo tiempo.
    - La etiqueta manual blurry no rechaza automaticamente si LA es objetivo y claro.
    """
    output = df.copy()

    bright_border_ok = (
        (output["border_status"] == "ok")
        & (output["border_p90_minus_la_median"] >= thresholds["min_border_p90_minus_la_median"])
        & (output["border_high_ratio_10"] >= thresholds["min_border_high_ratio_10"])
    )
    gradient_border_ok = (
        (output["border_status"] == "ok")
        & (output["border_gradient_p75"] >= thresholds["min_border_gradient_p75"])
    )
    anechoic_local_contrast_ok = (
        (output["border_status"] == "ok")
        & (output["border_p90_minus_la_p10"] >= thresholds["min_border_p90_minus_la_p10"])
    )
    border_ok = bright_border_ok | gradient_border_ok | anechoic_local_contrast_ok

    output["bright_border_rule"] = bright_border_ok.astype(int)
    output["gradient_border_rule"] = gradient_border_ok.astype(int)
    output["anechoic_local_contrast_rule"] = anechoic_local_contrast_ok.astype(int)
    output["has_hyperechoic_border_rule"] = border_ok.astype(int)
    output["acceptable_lumen_border_rule"] = (
        (output["acceptable_lumen_threshold_rule"] == 1) & (output["has_hyperechoic_border_rule"] == 1)
    ).astype(int)

    # Regla final V2: la aceptabilidad depende de LA y del borde. La calidad manual
    # se conserva para detectar desacuerdos, pero no rechaza automaticamente.
    output["acceptable_lumen_final_rule"] = output["acceptable_lumen_border_rule"].astype(int)
    output["manual_quality_disagreement"] = (
        (output["quality"] == "blurry") & (output["acceptable_lumen_final_rule"] == 1)
    ).astype(int)
    output["quality_warning"] = np.where(
        output["manual_quality_disagreement"] == 1,
        "etiqueta_blurry_pero_LA_objetivamente_claro",
        "",
    )

    output["doubtful_case_rule"] = (
        (output["acceptable_lumen_threshold_rule"] == 1) & (output["acceptable_lumen_final_rule"] == 0)
    ).astype(int)
    output["doubt_reason"] = output.apply(lambda row: build_doubt_reason(row, thresholds), axis=1)
    return output


def build_doubt_reason(row: pd.Series, thresholds: dict) -> str:
    """Construye una explicacion legible para casos dudosos."""
    if int(row.get("doubtful_case_rule", 0)) != 1:
        return ""

    reasons = []
    if int(row.get("has_hyperechoic_border_rule", 0)) != 1:
        reasons.append("sin_evidencia_por_borde_ni_contraste_local")
    if pd.notna(row.get("border_p90_minus_la_median")) and row["border_p90_minus_la_median"] < thresholds["min_border_p90_minus_la_median"]:
        reasons.append("p90_borde_bajo")
    if pd.notna(row.get("border_high_ratio_10")) and row["border_high_ratio_10"] < thresholds["min_border_high_ratio_10"]:
        reasons.append("pocos_pixeles_borde_brillante")
    if pd.notna(row.get("border_gradient_p75")) and row["border_gradient_p75"] < thresholds["min_border_gradient_p75"]:
        reasons.append("gradiente_borde_bajo")
    if pd.notna(row.get("border_p90_minus_la_p10")) and row["border_p90_minus_la_p10"] < thresholds["min_border_p90_minus_la_p10"]:
        reasons.append("contraste_local_lumen_entorno_bajo")
    if not reasons:
        reasons.append("requiere_revision_manual")
    return ";".join(reasons)


def summarize_group(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """Resume resultados por calidad, paciente o split."""
    grouped = (
        df.groupby(group_col)
        .agg(
            total_images=("filename", "count"),
            images_with_LA=("has_la", "sum"),
            accepted_texture_rule=("acceptable_lumen_threshold_rule", "sum"),
            accepted_border_rule=("acceptable_lumen_border_rule", "sum"),
            accepted_final_rule=("acceptable_lumen_final_rule", "sum"),
            manual_quality_disagreements=("manual_quality_disagreement", "sum"),
            doubtful_cases=("doubtful_case_rule", "sum"),
        )
        .reset_index()
        .rename(columns={group_col: "group"})
    )
    grouped["section"] = f"by_{group_col}"
    grouped["final_acceptance_rate"] = grouped["accepted_final_rule"] / grouped["total_images"]
    grouped["border_acceptance_rate_with_LA"] = grouped["accepted_border_rule"] / grouped["images_with_LA"].replace(0, np.nan)
    return grouped


def save_summaries(df: pd.DataFrame, thresholds: dict) -> None:
    """Guarda resumenes CSV y Markdown."""
    summary = pd.concat(
        [
            summarize_group(df, "quality"),
            summarize_group(df, "patient"),
            summarize_group(df, "split"),
        ],
        ignore_index=True,
    )
    summary.to_csv(OUTPUT_SUMMARY, index=False, encoding="utf-8-sig")

    threshold_rows = [
        {
            "parameter": key,
            "value": value,
            "method": f"percentil {REFERENCE_QUANTILE:.2f} de P001 clear aceptados en train/valid",
        }
        for key, value in thresholds.items()
    ]
    pd.DataFrame(threshold_rows).to_csv(OUTPUT_THRESHOLDS, index=False, encoding="utf-8-sig")

    doubtful = df[df["doubtful_case_rule"] == 1].copy()
    doubtful = doubtful.sort_values(
        by=["quality", "patient", "border_gradient_p75", "border_p90_minus_la_median"],
        ascending=[True, True, True, True],
    )
    doubtful.to_csv(OUTPUT_DOUBTFUL, index=False, encoding="utf-8-sig")
    review_columns = [
        "split",
        "patient",
        "quality",
        "doubt_reason",
        "la_area_px",
        "la_std_intensity",
        "glcm_entropy",
        "border_p90_minus_la_median",
        "border_gradient_p75",
        "border_high_ratio_10",
        "border_p90_minus_la_p10",
        "anechoic_local_contrast_rule",
        "filename",
        "roboflow_image_path",
        "la_mask_path",
    ]
    available_review_columns = [column for column in review_columns if column in doubtful.columns]
    doubtful[available_review_columns].to_csv(OUTPUT_DOUBTFUL_REVIEW, index=False, encoding="utf-8-sig")

    old_accepted = int(df["acceptable_lumen_threshold_rule"].sum())
    border_accepted = int(df["acceptable_lumen_border_rule"].sum())
    final_accepted = int(df["acceptable_lumen_final_rule"].sum())
    total = int(len(df))
    doubtful_count = int(df["doubtful_case_rule"].sum())

    markdown = f"""# Ajuste de regla de aceptabilidad longitudinal

Este reporte agrega una validacion de borde hiperecogenico alrededor de LA.

Nota de revision: tras revisar las figuras de casos dudosos, se corrigio la regla para no rechazar casos con lumen claramente visible cuando el higado de fondo es relativamente claro y el borde blanco no destaca por gradiente.

## Criterio agregado

La regla anterior se conserva como filtro inicial:

`has_la == 1 AND area/std/entropia dentro de umbrales`

La regla ajustada agrega:

`evidencia de LA valido por brillo del borde OR gradiente del borde OR contraste local entre lumen oscuro y entorno`

Para aproximar ese criterio se calcula un anillo externo estrecho alrededor de LA y se miden:

- diferencia entre percentil 90 del borde y mediana del lumen;
- gradiente local del borde;
- proporcion de pixeles del borde al menos 10 niveles mas brillantes que la mediana del lumen;
- diferencia entre el percentil 90 del entorno inmediato y el percentil 10 del LA, para capturar el centro anecoico aun si la mascara incluye parte de la pared.

## Umbrales usados

Los umbrales se derivaron de P001 clear en train/valid, porque fue la adquisicion revisada como longitudinal bien adquirida y con LA visible.

| Parametro | Valor |
| --- | ---: |
| min_border_p90_minus_la_median | {thresholds['min_border_p90_minus_la_median']:.4f} |
| min_border_gradient_p75 | {thresholds['min_border_gradient_p75']:.4f} |
| min_border_high_ratio_10 | {thresholds['min_border_high_ratio_10']:.4f} |
| min_border_p90_minus_la_p10 | {thresholds['min_border_p90_minus_la_p10']:.4f} |
| referencia | {thresholds['reference_count']} imagenes P001 clear train/valid |

## Resultado global

| Indicador | Valor |
| --- | ---: |
| Imagenes totales | {total} |
| Aceptadas por regla anterior | {old_accepted} |
| Aceptadas tras criterio de borde | {border_accepted} |
| Aceptadas por regla final corregida | {final_accepted} |
| Casos dudosos/reclasificados | {doubtful_count} |
| Tasa final de aceptacion | {final_accepted / total:.2%} |

## Interpretacion

Los casos dudosos corresponden a imagenes que antes pasaban los umbrales de textura, pero no muestran evidencia suficiente por brillo del borde, gradiente del borde ni contraste local lumen-entorno. Las imagenes blurry con LA claramente visible ya no se rechazan automaticamente; se marcan como desacuerdo entre etiqueta manual y criterio objetivo.

Archivos generados:

- `outputs/metrics/glcm_longitudinal_metrics_with_border_rule.csv`
- `outputs/reports/longitudinal_acceptability_border_rule_summary.csv`
- `outputs/reports/border_rule_threshold_summary.csv`
- `outputs/reports/longitudinal_doubtful_cases.csv`
- `outputs/reports/longitudinal_doubtful_cases_review_list.csv`
- `outputs/figures/doubtful_cases_top20.png`
"""
    OUTPUT_MARKDOWN.write_text(markdown, encoding="utf-8")


def make_doubtful_figure(df: pd.DataFrame, max_cases: int = 20) -> None:
    """Genera una lamina con los primeros casos dudosos para revision rapida."""
    doubtful = df[df["doubtful_case_rule"] == 1].head(max_cases).copy()
    if doubtful.empty:
        return

    OUTPUT_FIGURE.parent.mkdir(parents=True, exist_ok=True)
    columns = 4
    thumb_w, thumb_h = 260, 195
    label_h = 76
    margin = 18
    title_h = 52
    rows = int(np.ceil(len(doubtful) / columns))
    canvas_w = margin * (columns + 1) + thumb_w * columns
    canvas_h = title_h + margin * (rows + 1) + (thumb_h + label_h) * rows
    canvas = Image.new("RGB", (canvas_w, canvas_h), "white")
    draw = ImageDraw.Draw(canvas)
    title_font = load_font(18)
    label_font = load_font(11)
    draw.text((margin, 16), "Primeros casos dudosos tras ajuste de regla", fill=(20, 20, 20), font=title_font)

    for index, (_, row) in enumerate(doubtful.iterrows()):
        col = index % columns
        line = index // columns
        x = margin + col * (thumb_w + margin)
        y = title_h + margin + line * (thumb_h + label_h + margin)
        image_path = Path(str(row["roboflow_image_path"]))

        if image_path.exists():
            image = Image.open(image_path).convert("RGB")
            image.thumbnail((thumb_w, thumb_h), Image.Resampling.LANCZOS)
            canvas.paste(image, (x + (thumb_w - image.width) // 2, y + (thumb_h - image.height) // 2))
        draw.rectangle((x, y, x + thumb_w, y + thumb_h), outline=(80, 80, 80), width=1)

        reason = str(row["doubt_reason"])[:55]
        draw.text((x, y + thumb_h + 6), f"{row['patient']} | {row['quality']} | {row['split']}", fill=(20, 20, 20), font=label_font)
        draw.text((x, y + thumb_h + 24), f"grad={row['border_gradient_p75']:.2f} p90-med={row['border_p90_minus_la_median']:.2f}", fill=(70, 70, 70), font=label_font)
        draw.text((x, y + thumb_h + 40), f"p90-p10LA={row['border_p90_minus_la_p10']:.2f}", fill=(70, 70, 70), font=label_font)
        draw.text((x, y + thumb_h + 58), reason, fill=(130, 70, 0), font=label_font)

    canvas.save(OUTPUT_FIGURE)


def load_font(size: int) -> ImageFont.ImageFont:
    """Carga fuente disponible en Windows o usa fallback."""
    for path in [Path(r"C:\Windows\Fonts\arial.ttf"), Path(r"C:\Windows\Fonts\calibri.ttf")]:
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def main() -> None:
    """Ejecuta el ajuste de regla y guarda salidas."""
    ensure_output_dirs()
    if not INPUT_METRICS.exists():
        raise FileNotFoundError(f"No existe el CSV de metricas: {INPUT_METRICS}")

    metrics = pd.read_csv(INPUT_METRICS)
    border_rows = []
    for _, row in metrics.iterrows():
        border_rows.append(compute_border_metrics(row))
    border_df = pd.DataFrame(border_rows)
    combined = pd.concat([metrics.reset_index(drop=True), border_df.reset_index(drop=True)], axis=1)

    thresholds = derive_thresholds(combined)
    adjusted = apply_adjusted_rules(combined, thresholds)
    adjusted.to_csv(OUTPUT_METRICS, index=False, encoding="utf-8-sig")

    save_summaries(adjusted, thresholds)
    make_doubtful_figure(adjusted)

    print(f"Metricas ajustadas: {OUTPUT_METRICS}")
    print(f"Resumen: {OUTPUT_SUMMARY}")
    print(f"Umbrales de borde: {OUTPUT_THRESHOLDS}")
    print(f"Casos dudosos: {OUTPUT_DOUBTFUL}")
    print(f"Lista reducida de revision: {OUTPUT_DOUBTFUL_REVIEW}")
    print(f"Reporte Markdown: {OUTPUT_MARKDOWN}")
    print(f"Figura de casos dudosos: {OUTPUT_FIGURE}")
    print(f"Aceptadas regla anterior: {int(adjusted['acceptable_lumen_threshold_rule'].sum())}")
    print(f"Aceptadas con borde: {int(adjusted['acceptable_lumen_border_rule'].sum())}")
    print(f"Aceptadas regla final: {int(adjusted['acceptable_lumen_final_rule'].sum())}")
    print(f"Desacuerdos etiqueta blurry vs LA claro: {int(adjusted['manual_quality_disagreement'].sum())}")
    print(f"Casos dudosos/reclasificados: {int(adjusted['doubtful_case_rule'].sum())}")


if __name__ == "__main__":
    main()
