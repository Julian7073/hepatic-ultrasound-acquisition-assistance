"""Inferencia longitudinal frame por frame para video ecografico.

Usa los modelos finales seleccionados en outputs/segmentation_training/final_models.
El objetivo es dejar lista la base que despues puede conectarse a una GUI simple.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import torch

from src.inference_utils import decide_longitudinal_frame, expected_final_model_paths, load_binary_model, predict_mask


def safe_float(value: float | None) -> str:
    """Convierte valores opcionales a texto seguro para CSV."""
    if value is None:
        return ""
    return f"{float(value):.6f}"


def create_overlay(frame_bgr: np.ndarray, roi_mask: np.ndarray, higado_mask: np.ndarray, la_mask: np.ndarray) -> np.ndarray:
    """Genera overlay simple de mascaras para revision visual.

    Colores BGR:
    - ROI: verde
    - Higado: azul
    - LA: rojo
    """
    overlay = frame_bgr.copy()
    color_layer = np.zeros_like(frame_bgr)
    color_layer[roi_mask > 0] = (0, 180, 0)
    color_layer[higado_mask > 0] = (255, 0, 0)
    color_layer[la_mask > 0] = (0, 0, 255)
    blended = cv2.addWeighted(overlay, 0.72, color_layer, 0.28, 0)

    for mask, color in [(roi_mask, (0, 255, 0)), (higado_mask, (255, 60, 60)), (la_mask, (0, 0, 255))]:
        contours, _ = cv2.findContours((mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(blended, contours, -1, color, 2)

    return blended


def main() -> None:
    """Procesa un video frame por frame usando modelos finales."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", required=True, help="Ruta del video ecografico a procesar.")
    parser.add_argument(
        "--output_csv",
        default="outputs/segmentation_training/reports/video_inference_results.csv",
        help="CSV de salida con metricas y decision por frame.",
    )
    parser.add_argument("--view", default="Longitudinal", choices=["Longitudinal"], help="Vista seleccionada por el usuario.")
    parser.add_argument("--frame_stride", type=int, default=1, help="Procesa 1 de cada N frames.")
    parser.add_argument("--max_frames", type=int, default=0, help="Limite opcional de frames procesados. 0 procesa todo.")
    parser.add_argument("--save_overlays", action="store_true", help="Guarda overlays PNG para revisar mascaras predichas.")
    parser.add_argument(
        "--overlay_dir",
        default="outputs/segmentation_training/overlays/video_inference",
        help="Carpeta donde se guardan overlays si --save_overlays esta activo.",
    )
    parser.add_argument("--cpu", action="store_true", help="Fuerza inferencia en CPU aunque exista GPU.")
    args = parser.parse_args()

    if args.frame_stride < 1:
        raise ValueError("--frame_stride debe ser mayor o igual a 1")

    model_paths = expected_final_model_paths()
    missing = [str(path) for path in model_paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Faltan modelos finales. Ejecuta primero la comparacion/seleccion de modelos:\n" + "\n".join(missing)
        )

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Device: {device}")

    models = {}
    image_sizes = {}
    for class_name, path in model_paths.items():
        model, image_size = load_binary_model(path, device)
        models[class_name] = model
        image_sizes[class_name] = image_size
        print(f"Modelo {class_name}: {path.name} | image_size={image_size}")

    cap = cv2.VideoCapture(str(args.video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"No se pudo abrir video: {args.video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    overlay_dir = Path(args.overlay_dir)
    if args.save_overlays:
        overlay_dir.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "frame_id",
        "timestamp_s",
        "vista",
        "has_roi",
        "has_higado",
        "has_la",
        "area_roi_px",
        "area_higado_px",
        "area_la_px",
        "proporcion_higado_roi",
        "glcm_entropy",
        "glcm_contrast",
        "glcm_homogeneity",
        "glcm_energy",
        "la_std_intensity",
        "decision",
        "message",
    ]

    processed = 0
    decision_counts: Counter[str] = Counter()
    with output_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        frame_id = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_id % args.frame_stride != 0:
                frame_id += 1
                continue
            if args.max_frames and processed >= args.max_frames:
                break

            roi_mask = predict_mask(models["ROI"], frame, image_sizes["ROI"], device)
            higado_mask = predict_mask(models["Higado"], frame, image_sizes["Higado"], device)
            la_mask = predict_mask(models["LA"], frame, image_sizes["LA"], device)
            decision = decide_longitudinal_frame(frame, roi_mask, higado_mask, la_mask)
            decision_counts[decision.decision] += 1

            writer.writerow(
                {
                    "frame_id": frame_id,
                    "timestamp_s": safe_float(frame_id / fps if fps > 0 else None),
                    "vista": args.view,
                    "has_roi": decision.has_roi,
                    "has_higado": decision.has_higado,
                    "has_la": decision.has_la,
                    "area_roi_px": decision.area_roi_px,
                    "area_higado_px": decision.area_higado_px,
                    "area_la_px": decision.area_la_px,
                    "proporcion_higado_roi": safe_float(decision.higado_roi_ratio),
                    "glcm_entropy": safe_float(decision.glcm_entropy),
                    "glcm_contrast": safe_float(decision.glcm_contrast),
                    "glcm_homogeneity": safe_float(decision.glcm_homogeneity),
                    "glcm_energy": safe_float(decision.glcm_energy),
                    "la_std_intensity": safe_float(decision.la_std_intensity),
                    "decision": decision.decision,
                    "message": decision.message,
                }
            )

            if args.save_overlays:
                overlay = create_overlay(frame, roi_mask, higado_mask, la_mask)
                cv2.putText(
                    overlay,
                    decision.decision,
                    (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                cv2.imwrite(str(overlay_dir / f"frame_{frame_id:06d}_{decision.decision}.png"), overlay)

            processed += 1
            frame_id += 1

    cap.release()
    print(f"Inferencia guardada: {output_path}")
    print(f"Frames procesados: {processed}")
    print("Resumen de decisiones:")
    for decision_name, count in decision_counts.most_common():
        print(f"  {decision_name}: {count}")
    if args.save_overlays:
        print(f"Overlays guardados: {overlay_dir}")


if __name__ == "__main__":
    main()
