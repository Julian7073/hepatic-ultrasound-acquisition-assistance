"""Procesa un video longitudinal con los tres modelos seleccionados."""

from __future__ import annotations

import argparse
import sys
import time
from collections import Counter
from pathlib import Path

import cv2
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from config_experimental import (
    FINAL_MODELS_ROOT,
    VIDEO_INFERENCE_ROOT,
    ensure_directories,
)
from src.longitudinal_inference import (
    create_overlay,
    infer_frame,
    load_selected_models,
)
from src.reports import table_markdown


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", type=Path, required=True)
    parser.add_argument("--frame_stride", type=int, default=1)
    parser.add_argument("--max_frames", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--save_overlay_every", type=int, default=30)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    if args.frame_stride < 1:
        raise ValueError("frame_stride debe ser >= 1.")
    if not args.video_path.exists():
        raise FileNotFoundError(args.video_path)

    ensure_directories()
    device = torch.device(
        "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    )
    checkpoints = {
        "ROI": FINAL_MODELS_ROOT / "best_roi_model.pth",
        "Higado": FINAL_MODELS_ROOT / "best_higado_model.pth",
        "LA": FINAL_MODELS_ROOT / "best_la_model.pth",
    }
    models = load_selected_models(checkpoints, device)
    print(f"Device: {device}")

    capture = cv2.VideoCapture(str(args.video_path))
    if not capture.isOpened():
        raise RuntimeError(f"No se pudo abrir video: {args.video_path}")
    fps_source = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    ok, warmup_frame = capture.read()
    if not ok:
        raise RuntimeError("No se pudo leer el primer frame del video.")
    print(f"Warmup: {args.warmup} iteraciones")
    for _ in range(max(args.warmup, 0)):
        infer_frame(warmup_frame, models, device, args.threshold)
    capture.set(cv2.CAP_PROP_POS_FRAMES, 0)

    output_root = VIDEO_INFERENCE_ROOT / args.video_path.stem
    overlay_root = output_root / "overlays"
    output_root.mkdir(parents=True, exist_ok=True)
    if args.save_overlay_every > 0:
        overlay_root.mkdir(parents=True, exist_ok=True)

    rows = []
    frame_id = 0
    processed = 0
    start_all = time.perf_counter()
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if frame_id % args.frame_stride != 0:
            frame_id += 1
            continue
        if args.max_frames > 0 and processed >= args.max_frames:
            break

        start = time.perf_counter()
        result = infer_frame(frame, models, device, args.threshold)
        pipeline_ms = (time.perf_counter() - start) * 1000.0
        rows.append({
            "frame_id": frame_id,
            "timestamp_s": frame_id / fps_source if fps_source > 0 else None,
            "view": "longitudinal",
            "source_video": str(args.video_path),
            "source_fps": fps_source,
            "threshold": args.threshold,
            "pipeline_ms_per_frame": pipeline_ms,
            **result.to_row(),
        })
        if (
            args.save_overlay_every > 0
            and processed % args.save_overlay_every == 0
        ):
            cv2.imwrite(
                str(
                    overlay_root
                    / f"frame_{frame_id:06d}__{result.decision}.png"
                ),
                create_overlay(frame, result),
            )
        processed += 1
        frame_id += 1
        if processed % 50 == 0:
            print(f"Frames procesados: {processed}")

    capture.release()
    elapsed_s = time.perf_counter() - start_all
    results = pd.DataFrame(rows)
    if results.empty:
        raise RuntimeError("No se procesaron frames.")

    csv_path = output_root / "frame_results.csv"
    results.to_csv(csv_path, index=False, encoding="utf-8-sig")
    counts = Counter(results["decision"])
    summary = pd.DataFrame([
        {
            "processed_frames": len(results),
            "source_fps": fps_source,
            "elapsed_s": elapsed_s,
            "mean_pipeline_ms": results["pipeline_ms_per_frame"].mean(),
            "median_pipeline_ms": results["pipeline_ms_per_frame"].median(),
            "p95_pipeline_ms": results["pipeline_ms_per_frame"].quantile(0.95),
            "estimated_processing_fps": (
                1000.0 / results["pipeline_ms_per_frame"].mean()
            ),
            "realtime_ratio_vs_source": (
                (1000.0 / results["pipeline_ms_per_frame"].mean()) / fps_source
                if fps_source > 0 else None
            ),
            **{f"decision_{key}": value for key, value in counts.items()},
        }
    ])
    summary_path = output_root / "summary.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    report = [
        "# Inferencia longitudinal de video", "",
        f"- Video: {args.video_path}",
        f"- Frames procesados: {len(results)}",
        f"- Frame stride: {args.frame_stride}",
        f"- Device: {device}", "",
        table_markdown(summary), "",
        "Los mensajes son asistencia experimental y no constituyen diagnostico medico.",
    ]
    report_path = output_root / "summary.md"
    report_path.write_text("\n".join(report), encoding="utf-8")
    print(summary.to_string(index=False))
    print(f"Resultados: {csv_path}")
    print(f"Reporte: {report_path}")


if __name__ == "__main__":
    main()
