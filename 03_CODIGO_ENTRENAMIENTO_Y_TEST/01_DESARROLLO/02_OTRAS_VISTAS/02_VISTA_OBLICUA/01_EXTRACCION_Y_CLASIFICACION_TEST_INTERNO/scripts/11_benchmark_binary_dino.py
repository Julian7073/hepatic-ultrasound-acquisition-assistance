"""Benchmark end-to-end de los modelos DINOv2 binarios seleccionados."""

from __future__ import annotations

import platform
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

from config_dino import BINARY_FIGURES_ROOT, BINARY_REPORTS_ROOT, REPORTS_ROOT, VIEWS
from src.binary_inference import BinaryDinoVideoPredictor


def synchronize() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def main() -> None:
    index = pd.read_csv(REPORTS_ROOT / "02_dino_embedding_index_stride5.csv")
    development = index[index["role"] == "development"]
    rows = []
    warmup = 5
    iterations = 30
    for view in VIEWS:
        sample = development[development["view"] == view].iloc[0]
        bgr = cv2.imread(sample["image_path"], cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError(f"No se pudo leer {sample['image_path']}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        predictor = BinaryDinoVideoPredictor(view=view)
        predictor.reset()
        for _ in range(max(warmup, predictor.window_size)):
            predictor.predict_rgb(rgb)
        timings = []
        for _ in range(iterations):
            synchronize()
            started = time.perf_counter()
            predictor.predict_rgb(rgb)
            synchronize()
            timings.append((time.perf_counter() - started) * 1000.0)
        values = np.asarray(timings)
        rows.append({
            "view": view,
            "embedding_model_id": predictor.bundle["embedding_model_id"],
            "preprocessing": predictor.bundle["preprocessing"],
            "temporal_mode": predictor.bundle["temporal_mode"],
            "classifier": predictor.bundle["classifier"],
            "iterations": iterations,
            "mean_ms_per_evaluated_frame": float(values.mean()),
            "median_ms_per_evaluated_frame": float(np.median(values)),
            "p95_ms_per_evaluated_frame": float(np.percentile(values, 95)),
            "fps_if_every_frame": float(1000.0 / values.mean()),
            "configured_stride": int(predictor.bundle["stride"]),
            "effective_source_fps_capacity_at_stride": float(
                predictor.bundle["stride"] * 1000.0 / values.mean()
            ),
            "meets_30_fps_if_every_frame": int(values.mean() <= 33.333),
            "device": str(predictor.device),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "",
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
        })
        del predictor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    benchmark = pd.DataFrame(rows)
    csv_path = BINARY_REPORTS_ROOT / "12_binary_dino_inference_benchmark.csv"
    benchmark.to_csv(csv_path, index=False, encoding="utf-8-sig")

    figure, axis = plt.subplots(figsize=(9, 5), dpi=160)
    colors = [
        "#2A9D8F" if value else "#C44536"
        for value in benchmark["meets_30_fps_if_every_frame"]
    ]
    axis.bar(benchmark["view"], benchmark["mean_ms_per_evaluated_frame"], color=colors)
    axis.axhline(33.333, color="black", linestyle="--", label="Limite 30 FPS")
    axis.set_ylabel("ms por frame evaluado")
    axis.set_title("Benchmark end-to-end DINOv2 binario")
    axis.grid(True, axis="y", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure_path = BINARY_FIGURES_ROOT / "12_binary_dino_inference_benchmark.png"
    figure.savefig(figure_path)
    plt.close(figure)

    columns = [
        "view", "embedding_model_id", "preprocessing", "temporal_mode",
        "mean_ms_per_evaluated_frame", "median_ms_per_evaluated_frame",
        "p95_ms_per_evaluated_frame", "fps_if_every_frame", "configured_stride",
        "effective_source_fps_capacity_at_stride", "meets_30_fps_if_every_frame",
    ]
    display = benchmark[columns].copy()
    report_lines = [
        "# Benchmark del pipeline DINOv2 binario", "",
        "Se mide con batch size 1, modelo ya cargado, cinco iteraciones de calentamiento "
        "y sincronizacion CUDA. El tiempo incluye recorte cuando corresponde, processor "
        "DINOv2, backbone y clasificador.", "",
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in display.itertuples(index=False, name=None):
        report_lines.append("| " + " | ".join(str(value) for value in row) + " |")
    report_lines.extend([
        "",
        "El stride configurado reduce la frecuencia de inferencia pesada. Los frames "
        "intermedios pueden conservar la ultima decision, pero capture debe requerir "
        "consenso temporal y no una unica prediccion.", "",
        f"- CSV: {csv_path}",
        f"- Figura: {figure_path}",
    ])
    report_path = BINARY_REPORTS_ROOT / "12_binary_dino_inference_benchmark.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    print(benchmark[columns].to_string(index=False))
    print(f"Reporte: {report_path}")


if __name__ == "__main__":
    main()