"""Prepara el indice muestreado que se usara para embeddings DINOv2."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config_dino import REPORTS_ROOT, ensure_directories
from src.dino_embeddings import select_by_video_stride


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
    parser.add_argument("--stride", type=int, default=5)
    args = parser.parse_args()
    ensure_directories()
    source_path = REPORTS_ROOT / "00_dino_frame_index.csv"
    if not source_path.exists():
        raise FileNotFoundError("Ejecute primero scripts/00_audit_dino_dataset.py.")
    index = pd.read_csv(source_path)
    index = index[index["readable"] == 1].copy()
    sampled = select_by_video_stride(index, args.stride)
    sampled["embedding_stride"] = args.stride

    output_path = REPORTS_ROOT / f"02_dino_embedding_index_stride{args.stride}.csv"
    sampled.to_csv(output_path, index=False, encoding="utf-8-sig")
    summary = (
        sampled.groupby(["role", "patient", "view", "quality"], as_index=False)
        .agg(images=("filename", "size"), videos=("video_id", "nunique"))
    )
    summary_path = REPORTS_ROOT / f"02_dino_embedding_index_stride{args.stride}_summary.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

    role_summary = (
        sampled.groupby("role", as_index=False)
        .agg(images=("filename", "size"), videos=("video_id", "nunique"), patients=("patient", "nunique"))
    )
    report = [
        "# Indice de embeddings DINOv2", "",
        f"- Stride por video: {args.stride}",
        f"- Frames originales: {len(index)}",
        f"- Frames seleccionados: {len(sampled)}",
        f"- Videos conservados: {sampled['video_id'].nunique()}", "",
        "## Resumen por rol", "", markdown_table(role_summary), "",
        "## Resumen detallado", "", markdown_table(summary), "",
        "El muestreo se reinicia dentro de cada video y conserva el frame inicial. "
        "P005 permanece etiquetado exclusivamente como external_test.", "",
        "Los clasificadores se compararan por vista. La seleccion interna usara "
        "leave-one-patient-out con P001-P003 y metricas macro para reducir el efecto "
        "del desbalance entre videos.",
    ]
    report_path = REPORTS_ROOT / f"02_dino_embedding_index_stride{args.stride}.md"
    report_path.write_text("\n".join(report), encoding="utf-8")
    print(role_summary.to_string(index=False))
    print(f"Indice: {output_path}")
    print(f"Resumen: {report_path}")


if __name__ == "__main__":
    main()
