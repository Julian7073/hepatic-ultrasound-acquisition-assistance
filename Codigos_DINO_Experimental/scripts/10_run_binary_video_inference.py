"""Ejecuta el modelo DINOv2 binario seleccionado sobre un video nuevo."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config_dino import BINARY_ROOT, VIEWS
from src.binary_inference import process_video


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--view", choices=VIEWS, required=True)
    parser.add_argument("--output_csv", type=Path)
    parser.add_argument("--inference_stride", type=int)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    if not args.video.exists():
        parser.error(
            f"No existe el archivo: {args.video}. "
            "C:/ruta/video.mp4 era solamente un ejemplo; use la ruta real."
        )
    if not args.video.is_file():
        parser.error(f"La ruta no corresponde a un archivo: {args.video}")

    output = args.output_csv or (
        BINARY_ROOT / "video_inference" / f"{args.video.stem}__{args.view}.csv"
    )
    result = process_video(
        video_path=args.video,
        view=args.view,
        output_csv=output,
        inference_stride=args.inference_stride,
        device="cpu" if args.cpu else None,
    )
    evaluated = result[result["evaluated"] == 1]
    print(f"Frames totales: {len(result)}")
    print(f"Frames evaluados: {len(evaluated)}")
    print("Decisiones evaluadas:")
    print(evaluated["decision"].value_counts(dropna=False).to_string())
    print(f"CSV: {output}")


if __name__ == "__main__":
    main()