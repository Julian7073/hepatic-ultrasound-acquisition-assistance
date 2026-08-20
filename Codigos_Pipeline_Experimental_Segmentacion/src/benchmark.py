"""Benchmark reproducible de inferencia frame por frame."""

from __future__ import annotations

import platform
import statistics
import time

import pandas as pd
import torch

from src.models import count_parameters


@torch.no_grad()
def benchmark_model(model, loader, device, architecture: str, class_name: str, image_size: int, warmup: int = 10, max_frames: int = 100) -> dict:
    """Mide inferencia pura con batch 1 y sincronizacion CUDA."""
    model.eval()
    sample = next(iter(loader))["image"][:1].to(device)
    for _ in range(warmup):
        _ = model(sample)
    if device.type == "cuda":
        torch.cuda.synchronize()

    times_ms = []
    for batch in loader:
        image = batch["image"][:1].to(device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        _ = model(image)
        if device.type == "cuda":
            torch.cuda.synchronize()
        times_ms.append((time.perf_counter() - start) * 1000.0)
        if len(times_ms) >= max_frames:
            break

    mean_ms = statistics.fmean(times_ms)
    median_ms = statistics.median(times_ms)
    p95_ms = float(torch.tensor(times_ms).quantile(0.95).item())
    return {
        "architecture": architecture,
        "class_name": class_name,
        "resolution": f"{image_size}x{image_size}",
        "frames_measured": len(times_ms),
        "mean_ms_per_frame": mean_ms,
        "median_ms_per_frame": median_ms,
        "p95_ms_per_frame": p95_ms,
        "fps": 1000.0 / mean_ms,
        "parameter_count": count_parameters(model),
        "meets_30_fps": mean_ms <= 33.333,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No disponible",
        "cpu": platform.processor() or "No reportado",
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
