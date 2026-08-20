"""Mide ROI, Higado, LA y las tres redes en serie sobre frames reales."""

from __future__ import annotations

import argparse
import platform
import statistics
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

from config_experimental import DATASET_ROOTS, FIGURES_ROOT, REPORTS_ROOT
from src.dataset_coco import load_records
from src.models import count_parameters, create_model
from src.reports import table_markdown


def find_checkpoint(class_name: str) -> Path:
    """Lee el modelo seleccionado por validacion, sin consultar metricas de test."""
    selection_path = REPORTS_ROOT / "best_models_by_class.csv"
    if not selection_path.exists():
        raise FileNotFoundError(
            "Falta best_models_by_class.csv. Ejecute scripts/08_update_reports.py primero."
        )
    selection = pd.read_csv(selection_path)
    rows = selection[selection["class_name"] == class_name]
    if rows.empty:
        raise FileNotFoundError(f"No existe seleccion por validacion para {class_name}.")
    checkpoint_path = Path(str(rows.iloc[0]["checkpoint_path"]))
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint seleccionado no encontrado: {checkpoint_path}")
    return checkpoint_path


def load_model(path: Path, device):
    """Carga arquitectura y pesos sin descargar pretraining."""
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    config = checkpoint["config"]
    model, metadata = create_model(config["architecture"], pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()
    return model, config, metadata


def synchronize(device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def summarize(label: str, values: list[float], **extra) -> dict:
    mean = statistics.fmean(values)
    return {
        "component": label,
        "mean_ms_per_frame": mean,
        "median_ms_per_frame": statistics.median(values),
        "p95_ms_per_frame": float(np.percentile(values, 95)),
        "fps": 1000.0 / mean,
        "meets_30_fps": mean <= 33.333,
        **extra,
    }


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--roi_checkpoint", type=Path)
    parser.add_argument("--higado_checkpoint", type=Path)
    parser.add_argument("--la_checkpoint", type=Path)
    parser.add_argument("--max_frames", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    paths = {
        "ROI": args.roi_checkpoint or find_checkpoint("ROI"),
        "Higado": args.higado_checkpoint or find_checkpoint("Higado"),
        "LA": args.la_checkpoint or find_checkpoint("LA"),
    }
    print(f"Device: {device}", flush=True)
    loaded = {}
    for name, path in paths.items():
        print(f"Cargando {name}: {path}", flush=True)
        loaded[name] = load_model(path, device)
        print(f"{name} cargado.", flush=True)
    records = [record for record in load_records(DATASET_ROOTS["ROI"], "ROI") if record.original_split == "test"][: args.max_frames]
    if not records:
        raise RuntimeError("No se encontraron frames de test para el benchmark.")
    print(f"Frames para medir: {len(records)}", flush=True)

    sample = cv2.imread(str(records[0].image_path))
    for name, (model, config, _) in loaded.items():
        resized = cv2.resize(sample, (config["image_size"], config["image_size"]))
        tensor = torch.from_numpy(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)).float().unsqueeze(0).to(device) / 255.0
        for _ in range(args.warmup):
            _ = model(tensor)
    synchronize(device)
    print("Warmup terminado. Iniciando medicion...", flush=True)

    inference_times = {name: [] for name in loaded}
    read_times, preprocess_times, postprocess_times, total_times = [], [], [], []
    for record in records:
        total_start = time.perf_counter()
        start = time.perf_counter()
        image = cv2.imread(str(record.image_path))
        read_times.append((time.perf_counter() - start) * 1000.0)

        tensors = {}
        start = time.perf_counter()
        for name, (_, config, _) in loaded.items():
            resized = cv2.resize(image, (config["image_size"], config["image_size"]), interpolation=cv2.INTER_AREA)
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            tensors[name] = torch.from_numpy(rgb.transpose(2, 0, 1).copy()).float().unsqueeze(0).to(device) / 255.0
        synchronize(device)
        preprocess_times.append((time.perf_counter() - start) * 1000.0)

        logits = {}
        for name, (model, _, _) in loaded.items():
            synchronize(device)
            start = time.perf_counter()
            logits[name] = model(tensors[name])
            synchronize(device)
            inference_times[name].append((time.perf_counter() - start) * 1000.0)

        start = time.perf_counter()
        _ = {name: (torch.sigmoid(value) >= 0.5).to(torch.uint8).cpu().numpy() for name, value in logits.items()}
        synchronize(device)
        postprocess_times.append((time.perf_counter() - start) * 1000.0)
        total_times.append((time.perf_counter() - total_start) * 1000.0)

    rows = []
    for name, values in inference_times.items():
        model, config, metadata = loaded[name]
        rows.append(summarize(
            name,
            values,
            resolution=f"{config['image_size']}x{config['image_size']}",
            parameter_count=count_parameters(model),
            checkpoint_path=str(paths[name]),
            device=str(device),
        ))
    rows.extend([
        summarize("preprocessing", preprocess_times, resolution="mixed", parameter_count=0, checkpoint_path="", device=str(device)),
        summarize("postprocessing", postprocess_times, resolution="mixed", parameter_count=0, checkpoint_path="", device=str(device)),
        summarize("read_io", read_times, resolution="original", parameter_count=0, checkpoint_path="", device=str(device)),
        summarize("pipeline_3_models_total", total_times, resolution="mixed", parameter_count=sum(count_parameters(item[0]) for item in loaded.values()), checkpoint_path="", device=str(device)),
    ])
    frame = pd.DataFrame(rows)
    REPORTS_ROOT.mkdir(parents=True, exist_ok=True)
    csv_path = REPORTS_ROOT / "07_inference_benchmark.csv"
    md_path = REPORTS_ROOT / "07_inference_benchmark.md"
    frame.to_csv(csv_path, index=False, encoding="utf-8-sig")

    plot_components = ["ROI", "Higado", "LA", "pipeline_3_models_total"]
    plot_frame = frame[frame["component"].isin(plot_components)].copy()
    figure, axis = plt.subplots(figsize=(8, 4.8), dpi=160)
    colors = ["#2A6F97" if value >= 30.0 else "#C44536" for value in plot_frame["fps"]]
    bars = axis.bar(plot_frame["component"], plot_frame["fps"], color=colors)
    axis.axhline(30.0, color="#202020", linestyle="--", linewidth=1.2, label="Objetivo 30 fps")
    axis.set_ylabel("Frames por segundo (FPS)")
    axis.set_title("Velocidad de inferencia: modelos y pipeline completo")
    axis.tick_params(axis="x", rotation=12)
    axis.grid(True, axis="y", alpha=0.25)
    axis.legend()
    for bar, value in zip(bars, plot_frame["fps"]):
        axis.text(bar.get_x() + bar.get_width() / 2, value + 1.5, f"{value:.1f}", ha="center", fontsize=8)
    figure.tight_layout()
    FIGURES_ROOT.mkdir(parents=True, exist_ok=True)
    figure.savefig(FIGURES_ROOT / "07_inference_fps.png")
    plt.close(figure)

    total_row = frame[frame["component"] == "pipeline_3_models_total"].iloc[0]
    environment = [
        "# Benchmark de inferencia", "",
        f"- GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No disponible'}",
        f"- CPU: {platform.processor() or 'No reportado'}",
        f"- PyTorch: {torch.__version__}",
        f"- CUDA disponible: {torch.cuda.is_available()}",
        f"- Frames medidos: {len(records)}", "",
        table_markdown(frame), "",
        "El objetivo de 30 fps corresponde a un maximo aproximado de 33.3 ms por frame.", "",
        (
            f"El pipeline completo alcanza {total_row['fps']:.2f} fps y " 
            f"{total_row['mean_ms_per_frame']:.2f} ms/frame; por tanto, " 
            + ("cumple" if total_row["meets_30_fps"] else "no cumple")
            + " el objetivo de 30 fps."
        ),
    ]
    md_path.write_text("\n".join(environment), encoding="utf-8")
    print(frame.to_string(index=False))
    print(f"CSV: {csv_path}")
    print(f"Markdown: {md_path}")


if __name__ == "__main__":
    main()
