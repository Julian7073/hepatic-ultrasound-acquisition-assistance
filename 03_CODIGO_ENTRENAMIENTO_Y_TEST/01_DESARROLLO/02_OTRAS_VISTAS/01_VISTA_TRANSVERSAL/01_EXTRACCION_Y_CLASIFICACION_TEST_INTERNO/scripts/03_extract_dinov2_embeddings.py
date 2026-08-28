"""Extrae los embeddings definitivos del indice muestreado."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config_dino import EMBEDDINGS_ROOT, REPORTS_ROOT, ensure_directories
from src.dino_embeddings import DinoV2Extractor, save_embeddings


def table_markdown(frame: pd.DataFrame) -> str:
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
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    ensure_directories()

    index_path = REPORTS_ROOT / f"02_dino_embedding_index_stride{args.stride}.csv"
    if not index_path.exists():
        raise FileNotFoundError(
            f"Falta {index_path}. Ejecute scripts/02_prepare_embedding_index.py."
        )
    index = pd.read_csv(index_path)
    if set(index["role"].unique()) != {"development", "external_test"}:
        raise RuntimeError("El indice no conserva development y external_test separados.")

    device = torch.device(
        "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    )
    print(f"Device: {device}")
    print(f"Imagenes: {len(index)}")
    started = time.perf_counter()
    extractor = DinoV2Extractor(device=device)
    matrix, info = extractor.extract(
        [Path(path) for path in index["image_path"]],
        batch_size=args.batch_size,
    )
    elapsed_s = time.perf_counter() - started
    if matrix.shape != (len(index), extractor.embedding_dim):
        raise RuntimeError(f"Shape inesperado: {matrix.shape}")
    if not np.isfinite(matrix).all():
        raise RuntimeError("Los embeddings contienen NaN o infinito.")

    info.update({
        "stride": args.stride,
        "total_elapsed_seconds": elapsed_s,
        "total_images_per_second": len(index) / elapsed_s,
        "development_images": int((index["role"] == "development").sum()),
        "external_test_images": int((index["role"] == "external_test").sum()),
        "external_test_used_for_selection": False,
    })
    prefix = EMBEDDINGS_ROOT / f"dinov2_small_stride{args.stride}"
    paths = save_embeddings(matrix, index, prefix, info)

    distribution = (
        index.groupby(["role", "patient", "view", "quality"], as_index=False)
        .agg(images=("filename", "size"), videos=("video_id", "nunique"))
    )
    report = [
        "# Extraccion definitiva de embeddings DINOv2", "",
        f"- Modelo: {info['model_id']}",
        f"- Device: {info['device']}",
        f"- Stride: {args.stride}",
        f"- Imagenes: {info['images']}",
        f"- Desarrollo: {info['development_images']}",
        f"- P005 externo: {info['external_test_images']}",
        f"- Dimension: {info['embedding_dim']}",
        f"- Batch size: {info['batch_size']}",
        f"- Tiempo total: {elapsed_s:.2f} s",
        f"- Velocidad total: {info['total_images_per_second']:.2f} imagenes/s", "",
        "P005 se transformo con el mismo extractor congelado, pero sus etiquetas no se "
        "usaron para seleccionar clasificador ni hiperparametros.", "",
        "## Distribucion", "", table_markdown(distribution), "",
        f"Archivos: {[str(path) for path in paths]}",
    ]
    report_path = REPORTS_ROOT / "03_dinov2_embedding_extraction.md"
    report_path.write_text("\n".join(report), encoding="utf-8")
    print(json.dumps(info, indent=2))
    print(f"Embeddings: {paths[0]}")
    print(f"Reporte: {report_path}")


if __name__ == "__main__":
    main()
