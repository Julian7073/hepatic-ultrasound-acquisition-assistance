"""Extrae embeddings DINOv2 para variantes de entrada controladas."""

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

from config_dino import (
    BINARY_EMBEDDINGS_ROOT,
    BINARY_REPORTS_ROOT,
    DINOV2_MODEL_IDS,
    REPORTS_ROOT,
    ensure_directories,
)
from src.dino_embeddings import DinoV2Extractor, save_embeddings
from src.ultrasound_preprocessing import fan_crop_pil


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", choices=sorted(DINOV2_MODEL_IDS), default="small")
    parser.add_argument("--preprocessing", choices=("fan_crop", "full"), default="fan_crop")
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=12)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    ensure_directories()

    index_path = REPORTS_ROOT / f"02_dino_embedding_index_stride{args.stride}.csv"
    if not index_path.exists():
        raise FileNotFoundError(f"Falta {index_path}")
    index = pd.read_csv(index_path)
    if set(index["role"].unique()) != {"development", "external_test"}:
        raise RuntimeError("El indice no conserva P005 como external_test.")

    name = f"dinov2_{args.backbone}_{args.preprocessing}_stride{args.stride}"
    prefix = BINARY_EMBEDDINGS_ROOT / name
    if prefix.with_suffix(".npz").exists() and not args.force:
        print(f"Ya existe: {prefix.with_suffix('.npz')}")
        print("Use --force solo si desea regenerarlo.")
        return

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    model_id = DINOV2_MODEL_IDS[args.backbone]
    transform = fan_crop_pil if args.preprocessing == "fan_crop" else None
    print(f"Modelo: {model_id}")
    print(f"Preprocesamiento: {args.preprocessing}")
    print(f"Device: {device} | Imagenes: {len(index)}")
    started = time.perf_counter()
    extractor = DinoV2Extractor(device=device, model_id=model_id)
    matrix, info = extractor.extract(
        [Path(value) for value in index["image_path"]],
        batch_size=args.batch_size,
        image_transform=transform,
    )
    elapsed = time.perf_counter() - started
    if matrix.shape != (len(index), extractor.embedding_dim):
        raise RuntimeError(f"Shape inesperado: {matrix.shape}")
    if not np.isfinite(matrix).all():
        raise RuntimeError("Los embeddings contienen NaN o infinito.")

    info.update({
        "backbone": args.backbone,
        "preprocessing": args.preprocessing,
        "stride": args.stride,
        "total_elapsed_seconds": elapsed,
        "total_images_per_second": len(index) / elapsed,
        "development_images": int((index["role"] == "development").sum()),
        "external_test_images": int((index["role"] == "external_test").sum()),
        "external_test_used_for_selection": False,
    })
    paths = save_embeddings(matrix, index, prefix, info)
    report = [
        f"# Embeddings {name}", "",
        f"- Modelo: {model_id}",
        f"- Preprocesamiento: {args.preprocessing}",
        f"- Dimension: {extractor.embedding_dim}",
        f"- Imagenes: {len(index)}",
        f"- Device: {device}",
        f"- Tiempo total: {elapsed:.2f} s",
        f"- Velocidad total: {len(index) / elapsed:.2f} imagenes/s", "",
        "P005 se transforma con el mismo extractor, pero no participa en la seleccion "
        "de entrada, clasificador, modo temporal ni umbrales.", "",
        f"- Matriz: {paths[0]}",
        f"- Metadata: {paths[1]}",
        f"- Configuracion: {paths[2]}",
    ]
    report_path = BINARY_REPORTS_ROOT / f"07_{name}.md"
    report_path.write_text("\n".join(report), encoding="utf-8")
    print(json.dumps(info, indent=2))
    print(f"Reporte: {report_path}")


if __name__ == "__main__":
    main()