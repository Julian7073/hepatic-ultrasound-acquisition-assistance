"""Evalua externamente P005 sin usarlo para entrenamiento ni ajuste."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config_experimental import (
    EXTERNAL_P005_ROOT,
    FIGURES_ROOT,
    FINAL_MODELS_ROOT,
    P005_FRAMES_ROOT,
    REPORTS_ROOT,
    ensure_directories,
)
from src.longitudinal_inference import (
    create_overlay,
    infer_frame,
    load_selected_models,
)
from src.reports import table_markdown


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def selected_paths() -> dict[str, Path]:
    return {
        "ROI": FINAL_MODELS_ROOT / "best_roi_model.pth",
        "Higado": FINAL_MODELS_ROOT / "best_higado_model.pth",
        "LA": FINAL_MODELS_ROOT / "best_la_model.pth",
    }


def choose_overlay_paths(paths: list[Path], per_quality: int) -> set[Path]:
    selected = set()
    qualities = sorted({path.parent.name.lower() for path in paths})
    for quality in qualities:
        group = [path for path in paths if path.parent.name.lower() == quality]
        if not group:
            continue
        indexes = np.linspace(
            0,
            len(group) - 1,
            num=min(per_quality, len(group)),
            dtype=int,
        )
        selected.update(group[int(index)] for index in indexes)
    return selected


def summarize_group(frame: pd.DataFrame, label: str) -> dict:
    total = len(frame)
    return {
        "quality": label,
        "total_images": total,
        "has_roi_count": int(frame["has_roi"].sum()),
        "has_roi_rate": float(frame["has_roi"].mean()),
        "has_higado_count": int(frame["has_higado"].sum()),
        "has_higado_rate": float(frame["has_higado"].mean()),
        "has_la_count": int(frame["has_la"].sum()),
        "has_la_rate": float(frame["has_la"].mean()),
        "capture_count": int((frame["decision"] == "capture").sum()),
        "capture_rate": float((frame["decision"] == "capture").mean()),
        "median_roi_area_px": float(frame["area_roi_px"].median()),
        "median_higado_area_px": float(frame["area_higado_px"].median()),
        "median_la_area_px": float(frame["area_la_px"].median()),
        "mean_pipeline_ms": float(frame["pipeline_ms_per_image"].mean()),
        "estimated_pipeline_fps": float(
            1000.0 / max(frame["pipeline_ms_per_image"].mean(), 1e-9)
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluacion externa no anotada de P005."
    )
    parser.add_argument("--image_root", type=Path, default=P005_FRAMES_ROOT)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--max_images_per_quality", type=int, default=0)
    parser.add_argument("--overlays_per_quality", type=int, default=5)
    parser.add_argument("--save_overlays", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    ensure_directories()
    if not args.image_root.exists():
        raise FileNotFoundError(f"No existe P005: {args.image_root}")
    paths = sorted(
        path
        for path in args.image_root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if args.max_images_per_quality > 0:
        limited = []
        for quality in sorted({path.parent.name.lower() for path in paths}):
            limited.extend(
                [path for path in paths if path.parent.name.lower() == quality][
                    : args.max_images_per_quality
                ]
            )
        paths = sorted(limited)
    if not paths:
        raise RuntimeError("No se encontraron imagenes P005.")

    device = torch.device(
        "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    )
    print(f"Device: {device}")
    checkpoints = selected_paths()
    for name, path in checkpoints.items():
        print(f"Modelo {name}: {path}")
    models = load_selected_models(checkpoints, device)

    sample = cv2.imread(str(paths[0]), cv2.IMREAD_COLOR)
    if sample is None:
        raise FileNotFoundError(f"No se pudo leer: {paths[0]}")
    print("Warmup...")
    for _ in range(3):
        infer_frame(sample, models, device, args.threshold)

    overlay_paths = (
        choose_overlay_paths(paths, args.overlays_per_quality)
        if args.save_overlays
        else set()
    )
    overlay_root = EXTERNAL_P005_ROOT / "overlays"
    if overlay_paths:
        overlay_root.mkdir(parents=True, exist_ok=True)

    rows = []
    start_all = time.perf_counter()
    for index, path in enumerate(paths, start=1):
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            print(f"ADVERTENCIA: imagen ilegible: {path}")
            continue
        start = time.perf_counter()
        result = infer_frame(image, models, device, args.threshold)
        pipeline_ms = (time.perf_counter() - start) * 1000.0
        row = {
            "patient": "P005",
            "view": "longitudinal",
            "quality": path.parent.name.lower(),
            "filename": path.name,
            "image_path": str(path),
            "width": image.shape[1],
            "height": image.shape[0],
            "ground_truth_available": 0,
            "external_validation_type": "qualitative_unannotated",
            "threshold": args.threshold,
            "pipeline_ms_per_image": pipeline_ms,
            **result.to_row(),
        }
        rows.append(row)
        if path in overlay_paths:
            quality_dir = overlay_root / path.parent.name.lower()
            quality_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(
                str(quality_dir / f"{path.stem}__{result.decision}.png"),
                create_overlay(image, result),
            )
        if index % 25 == 0 or index == len(paths):
            print(f"P005: {index}/{len(paths)}")

    elapsed_s = time.perf_counter() - start_all
    results = pd.DataFrame(rows)
    if results.empty:
        raise RuntimeError("No fue posible procesar P005.")

    per_image_path = REPORTS_ROOT / "10_external_validation_p005.csv"
    results.to_csv(per_image_path, index=False, encoding="utf-8-sig")

    summary_rows = [summarize_group(results, "overall")]
    summary_rows.extend(
        summarize_group(group, quality)
        for quality, group in results.groupby("quality", sort=True)
    )
    summary = pd.DataFrame(summary_rows)
    summary_path = REPORTS_ROOT / "10_external_validation_p005_summary.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

    decisions = pd.crosstab(
        results["quality"],
        results["decision"],
        margins=True,
        margins_name="total",
    ).reset_index()
    decisions_path = REPORTS_ROOT / "10_external_validation_p005_decisions.csv"
    decisions.to_csv(decisions_path, index=False, encoding="utf-8-sig")

    quality_plot = summary[summary["quality"] != "overall"].copy()
    figure, axis = plt.subplots(figsize=(8, 4.8), dpi=160)
    x = np.arange(len(quality_plot))
    width = 0.35
    axis.bar(
        x - width / 2,
        quality_plot["has_la_rate"],
        width,
        label="LA detectado",
    )
    axis.bar(
        x + width / 2,
        quality_plot["capture_rate"],
        width,
        label="Captura sugerida",
    )
    axis.set_xticks(x, quality_plot["quality"])
    axis.set_ylim(0, 1)
    axis.set_ylabel("Proporcion de imagenes")
    axis.set_title("P005: resultados predictivos por calidad nominal")
    axis.grid(True, axis="y", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure_path = FIGURES_ROOT / "10_p005_predictions_by_quality.png"
    figure.savefig(figure_path)
    plt.close(figure)

    report_lines = [
        "# Evaluacion externa longitudinal P005", "",
        "> P005 no posee mascaras ground truth. Este analisis es predictivo, cualitativo "
        "y operativo; no permite calcular Dice, IoU ni sensibilidad clinica externa.", "",
        f"- Imagenes procesadas: {len(results)}",
        f"- Tiempo total: {elapsed_s:.2f} s",
        f"- Device: {device}",
        f"- Umbral binario: {args.threshold}",
        "- P005 no fue utilizado para entrenamiento, seleccion de checkpoints ni ajuste de umbrales.",
        "",
        "## Modelos utilizados", "",
        table_markdown(pd.DataFrame([
            {
                "class_name": name,
                "checkpoint": str(path),
                "architecture": models[name].architecture,
                "image_size": models[name].image_size,
                "resize_mode": models[name].resize_mode,
            }
            for name, path in checkpoints.items()
        ])), "",
        "## Resumen por calidad nominal", "",
        table_markdown(summary), "",
        "## Distribucion de decisiones", "",
        table_markdown(decisions), "",
        "Las carpetas clear, medium y blurry son etiquetas nominales de adquisicion. "
        "No sustituyen una anotacion externa de ROI, Higado o LA.", "",
        "La decision capture conserva la regla preliminar de area LA, desviacion "
        "estandar y entropia. Debe revisarse visualmente antes de considerarla regla clinica.", "",
        f"Figura: {figure_path}", "",
    ]
    report_path = REPORTS_ROOT / "10_external_validation_p005.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    print("\nEvaluacion P005 finalizada.")
    print(summary.to_string(index=False))
    print(f"CSV por imagen: {per_image_path}")
    print(f"Resumen: {summary_path}")
    print(f"Reporte: {report_path}")
    if overlay_paths:
        print(f"Overlays: {overlay_root}")


if __name__ == "__main__":
    main()
