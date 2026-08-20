"""Prueba funcional de descarga y extraccion DINOv2."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config_dino import EMBEDDINGS_ROOT, REPORTS_ROOT, ensure_directories
from src.dino_embeddings import DinoV2Extractor, balanced_smoke_sample, save_embeddings


def markdown_table(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in frame.astype(str).itertuples(index=False, name=None):
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_images", type=int, default=18)
    parser.add_argument("--batch_size", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    ensure_directories()
    index_path = REPORTS_ROOT / "00_dino_frame_index.csv"
    if not index_path.exists():
        raise FileNotFoundError("Ejecute primero scripts/00_audit_dino_dataset.py.")
    index = pd.read_csv(index_path)
    development = index[
        (index["role"] == "development") & (index["readable"] == 1)
    ].copy()
    sample = balanced_smoke_sample(development, args.max_images, args.seed)
    device = torch.device(
        "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    )
    print(f"Device: {device}")
    print(f"Imagenes smoke: {len(sample)}")
    extractor = DinoV2Extractor(device=device)
    matrix, info = extractor.extract(
        [Path(path) for path in sample["image_path"]],
        batch_size=args.batch_size,
    )
    if matrix.shape != (len(sample), extractor.embedding_dim):
        raise RuntimeError(f"Shape inesperado: {matrix.shape}")
    if not torch.isfinite(torch.from_numpy(matrix)).all():
        raise RuntimeError("Embeddings con NaN o infinito.")

    output_prefix = EMBEDDINGS_ROOT / "smoke_dinov2_small"
    paths = save_embeddings(matrix, sample, output_prefix, info)
    distribution = (
        sample.groupby(["view", "quality"])
        .size().rename("images").reset_index()
    )
    report = [
        "# Smoke test DINOv2", "",
        "> Prueba funcional, no concluyente. No se entrenaron clasificadores ni se evaluo P005.", "",
        f"- Modelo: {info['model_id']}",
        f"- Device: {info['device']}",
        f"- Imagenes: {info['images']}",
        f"- Dimension del embedding: {info['embedding_dim']}",
        f"- Batch size: {info['batch_size']}",
        f"- Velocidad del modelo: {info['model_images_per_second']:.2f} imagenes/s", "",
        "## Distribucion de la muestra", "", markdown_table(distribution), "",
    ]
    report_path = REPORTS_ROOT / "01_dinov2_smoke_test.md"
    report_path.write_text("\n".join(report), encoding="utf-8")
    print(json.dumps(info, indent=2))
    print(f"Embeddings shape: {matrix.shape}")
    print(f"Archivos: {[str(path) for path in paths]}")
    print(f"Reporte: {report_path}")


if __name__ == "__main__":
    main()
