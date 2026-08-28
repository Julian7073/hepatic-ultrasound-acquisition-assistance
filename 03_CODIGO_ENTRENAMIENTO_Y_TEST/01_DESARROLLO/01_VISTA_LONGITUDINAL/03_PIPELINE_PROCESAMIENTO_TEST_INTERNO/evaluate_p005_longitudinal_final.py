"""Evaluacion operacional final del pipeline longitudinal en P005.

P005 no tiene mascaras ground truth. Este script no calcula Dice ni IoU:
registra detecciones, decisiones, tiempos y evidencia cualitativa.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent
TESIS_ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))

from config_experimental import FINAL_MODELS_ROOT, FIGURES_ROOT, OUTPUT_ROOT
from src.longitudinal_inference import create_overlay, infer_frame, load_selected_models
from src.longitudinal_quality_rules import load_decision_config
from src.temporal_guidance import TemporalGuidance


VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".wmv"}
QUALITY_ORDER = {"clear": 0, "medium": 1, "blurry": 2, "unknown": 3}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evalua P005 longitudinal sin ground truth."
    )
    parser.add_argument("--p005_longitudinal_root", type=Path)
    parser.add_argument("--frame_stride", type=int, default=3)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--max_frames", type=int, default=0)
    parser.add_argument("--decision_config", type=Path)
    parser.add_argument("--save_overlays", action="store_true")
    parser.add_argument(
        "--save_csv", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--temporal_window", type=int, default=5)
    parser.add_argument("--capture_streak", type=int, default=3)
    parser.add_argument("--capture_cooldown", type=int, default=10)
    parser.add_argument("--output_dir", type=Path)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_selected_models() -> tuple[dict[str, Path], pd.DataFrame]:
    manifest_path = FINAL_MODELS_ROOT / "selected_models_manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"No existe el manifiesto: {manifest_path}")
    manifest = pd.read_csv(manifest_path)
    required = {"class_name", "alias_checkpoint", "sha256"}
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"Manifiesto incompleto; faltan: {sorted(missing)}")

    checkpoints = {}
    rows = []
    for class_name in ("ROI", "Higado", "LA"):
        selected = manifest[manifest["class_name"] == class_name]
        if len(selected) != 1:
            raise ValueError(
                f"Se esperaba una fila para {class_name}; encontradas {len(selected)}."
            )
        item = selected.iloc[0]
        checkpoint = Path(str(item["alias_checkpoint"]))
        if not checkpoint.exists():
            raise FileNotFoundError(f"Falta checkpoint {class_name}: {checkpoint}")
        actual_hash = sha256_file(checkpoint)
        expected_hash = str(item["sha256"]).strip().lower()
        if actual_hash.lower() != expected_hash:
            raise ValueError(
                f"Hash no coincide para {class_name}: "
                f"esperado={expected_hash}, actual={actual_hash}"
            )
        checkpoints[class_name] = checkpoint
        rows.append({
            "class_name": class_name,
            "architecture": item.get("architecture", ""),
            "experiment_name": item.get("experiment_name", ""),
            "checkpoint": str(checkpoint),
            "expected_sha256": expected_hash,
            "actual_sha256": actual_hash,
            "hash_verified": True,
        })
    return checkpoints, pd.DataFrame(rows)


def infer_quality(path: Path) -> str:
    text = " ".join(part.lower() for part in path.parts)
    for quality in ("clear", "medium", "blurry"):
        if quality in text:
            return quality
    return "unknown"


def find_p005_videos(explicit_root: Path | None) -> list[tuple[str, Path]]:
    search_root = explicit_root or (TESIS_ROOT / "Dataset")
    if not search_root.exists():
        raise FileNotFoundError(
            f"No existe la raiz P005: {search_root}. "
            "Use --p005_longitudinal_root."
        )
    videos = []
    for path in search_root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in VIDEO_SUFFIXES:
            continue
        lower_path = str(path).lower()
        if explicit_root is None:
            is_p005 = (
                "p005" in lower_path
                or "paciente 005" in lower_path
                or "paciente_005" in lower_path
            )
            if not is_p005 or "longitudinal" not in lower_path:
                continue
        videos.append((infer_quality(path), path.resolve()))
    videos.sort(key=lambda item: (QUALITY_ORDER[item[0]], str(item[1]).lower()))
    if not videos:
        raise FileNotFoundError(
            f"No se encontraron videos longitudinales P005 bajo {search_root}."
        )
    return videos


def markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_Sin registros._"
    display = frame.fillna("")
    headers = [str(column) for column in display.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in display.iterrows():
        values = [str(row[column]).replace("|", "\\|") for column in display.columns]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def prepare_output(args: argparse.Namespace) -> Path:
    base = args.output_dir or (OUTPUT_ROOT / "p005_longitudinal_final")
    if args.max_frames > 0 and args.output_dir is None:
        base = base / f"smoke_{args.max_frames}_frames"
    for folder in (
        base,
        base / "overlays",
        base / "captured_frames",
        base / "best_candidate_frames",
    ):
        folder.mkdir(parents=True, exist_ok=True)
    return base


def frame_quota(max_frames: int, video_index: int, video_count: int, used: int) -> int:
    if max_frames <= 0:
        return 0
    remaining = max_frames - used
    remaining_videos = video_count - video_index
    return max(0, math.ceil(remaining / max(remaining_videos, 1)))


def candidate_score(row: dict) -> tuple:
    return (
        int(row["capture_confirmed"]),
        int(row["raw_decision"] == "capture"),
        int(row["base_lumen_rule"]),
        int(row["la_present"]),
        int(row["la_area_px"]),
        -float(row["glcm_entropy"]) if pd.notna(row["glcm_entropy"]) else -1e9,
    )


def save_example_figure(results: pd.DataFrame, destination: Path) -> None:
    if results.empty or "overlay_path" not in results:
        return
    usable = results[
        results["overlay_path"].astype(str).map(lambda item: Path(item).exists())
    ]
    failures = usable[
        (usable["roi_present"] == 1)
        & (usable["liver_present"] == 1)
        & (usable["stable_decision"] != "capture")
    ].sort_values(
        ["la_present", "la_area_px", "glcm_entropy"],
        ascending=[False, False, True],
        na_position="last",
    ).head(3)
    captures = usable[
        (usable["capture_confirmed"] == 1)
        | (usable["stable_decision"] == "capture")
    ].head(3)

    items = [("Sin captura", row) for _, row in failures.iterrows()]
    items += [("Captura", row) for _, row in captures.iterrows()]
    if not items:
        return

    figure, axes = plt.subplots(2, 3, figsize=(15, 8.5))
    axes = np.asarray(axes).reshape(-1)
    for axis in axes:
        axis.axis("off")
    for axis, (label, row) in zip(axes, items):
        image = cv2.imread(str(row["overlay_path"]))
        if image is None:
            continue
        axis.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        axis.set_title(
            f"{label} | {row['quality']} | frame {int(row['frame_index'])}\n"
            f"LA={int(row['la_area_px'])} px | {row['raw_decision']}"
        )
        axis.axis("off")
    if captures.empty:
        axes[-1].text(
            0.5, 0.5, "sin capturas confirmadas",
            ha="center", va="center", fontsize=15,
        )
    figure.suptitle("P005 longitudinal: ejemplos operacionales", fontsize=16)
    figure.tight_layout()
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=180, bbox_inches="tight")
    plt.close(figure)


def summarize_by_quality(results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for quality in sorted(results["quality"].unique(), key=QUALITY_ORDER.get):
        group = results[results["quality"] == quality]
        mean_ms = float(group["ms_total"].mean())
        rows.append({
            "quality": quality,
            "frames_processed": len(group),
            "roi_present_count": int(group["roi_present"].sum()),
            "liver_present_count": int(group["liver_present"].sum()),
            "la_present_count": int(group["la_present"].sum()),
            "raw_capture_count": int((group["raw_decision"] == "capture").sum()),
            "stable_capture_count": int(
                (group["stable_decision"] == "capture").sum()
            ),
            "saved_captures": int(group["saved_capture"].sum()),
            "no_structure_count": int(
                (group["stable_decision"] == "no_structure").sum()
            ),
            "partial_liver_count": int(
                (group["stable_decision"] == "partial_liver").sum()
            ),
            "liver_without_reference_count": int(
                (group["stable_decision"] == "liver_without_reference").sum()
            ),
            "capture_count": int(group["capture_confirmed"].sum()),
            "mean_ms": mean_ms,
            "fps_effective": 1000.0 / mean_ms if mean_ms > 0 else 0.0,
        })
    return pd.DataFrame(rows)


def decision_distribution(results: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for level, column in (("raw", "raw_decision"), ("stable", "stable_decision")):
        counts = (
            results.groupby(["quality", column], dropna=False)
            .size()
            .reset_index(name="count")
            .rename(columns={column: "decision"})
        )
        counts.insert(1, "decision_level", level)
        frames.append(counts)
    return pd.concat(frames, ignore_index=True)


def write_report(
    output_dir: Path,
    results: pd.DataFrame,
    summary: pd.DataFrame,
    decisions: pd.DataFrame,
    config: dict,
    verification: pd.DataFrame,
    videos: list[tuple[str, Path]],
    args: argparse.Namespace,
    figure_path: Path,
) -> Path:
    mean_ms = float(results["ms_total"].mean())
    filename_warning = any("paciente 003" in path.name.lower() for _, path in videos)
    config_report = {key: value for key, value in config.items() if key != "config_path"}
    lines = [
        "# Evaluacion operacional final P005 longitudinal",
        "",
        "## Objetivo",
        "",
        "Aplicar los modelos longitudinales congelados y la regla final unificada "
        "sobre videos de P005. Esta prueba es funcional y operacional; no es una "
        "validacion clinica cuantitativa.",
        "",
        "## Modelos y hashes verificados",
        "",
        markdown_table(verification[[
            "class_name", "architecture", "experiment_name",
            "actual_sha256", "hash_verified",
        ]]),
        "",
        "## Configuracion de decision",
        "",
        "~~~json",
        json.dumps(config_report, indent=2, ensure_ascii=False),
        "~~~",
        "",
        f"- Ventana temporal: {args.temporal_window}",
        f"- Capturas crudas consecutivas requeridas: {args.capture_streak}",
        f"- Cooldown: {args.capture_cooldown} evaluaciones",
        f"- Stride: {args.frame_stride}",
        f"- Maximo evaluado: {args.max_frames or 'sin limite'}",
        "",
        "## Videos procesados",
        "",
        markdown_table(pd.DataFrame(videos, columns=["quality", "video_path"])),
        "",
        "## Resumen por calidad",
        "",
        markdown_table(summary.round(4)),
        "",
        "## Distribucion de decisiones",
        "",
        markdown_table(decisions),
        "",
        "## Capturas y rendimiento",
        "",
        f"- Capturas guardadas: {int(results['saved_capture'].sum())}",
        f"- Frames evaluados: {len(results)}",
        f"- Tiempo medio total: {mean_ms:.3f} ms/frame",
        f"- FPS efectivo medio: {1000.0 / mean_ms if mean_ms > 0 else 0.0:.2f}",
        f"- Figura cualitativa: {figure_path}",
        "",
        "## Interpretacion",
        "",
        "La salida indica cuantas veces los modelos y la politica congelada "
        "identificaron ROI, higado y LA suficiente. Cero capturas no demuestra por "
        "si solo un fallo: puede reflejar una politica conservadora, ausencia real "
        "de LA, falsos negativos o umbrales estrictos.",
        "",
        "## Limitaciones",
        "",
        "- P005 no tiene mascaras ground truth.",
        "- No se puede calcular Dice ni IoU externo.",
        "- Las etiquetas clear/medium/blurry son nominales por carpeta.",
        "- La prueba no establece validez clinica ni rendimiento diagnostico.",
        "- Para afirmar rendimiento cuantitativo externo se deben anotar P005 o "
        "nuevos pacientes.",
    ]
    if filename_warning:
        lines.extend([
            "",
            "## Advertencia de trazabilidad",
            "",
            "Los videos estan bajo la carpeta de P005, pero sus nombres internos "
            "contienen 'PACIENTE 003'. La asignacion operacional a P005 se basa en "
            "la ruta del dataset. Esta inconsistencia debe aclararse en la tesis.",
        ])
    report_path = output_dir / "p005_longitudinal_final_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def main() -> None:
    args = parse_args()
    if args.frame_stride < 1:
        raise ValueError("--frame_stride debe ser >=1.")
    if args.max_frames < 0:
        raise ValueError("--max_frames debe ser >=0.")

    config = load_decision_config(args.decision_config)
    videos = find_p005_videos(args.p005_longitudinal_root)
    output_dir = prepare_output(args)
    checkpoint_paths, verification = verify_selected_models()
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")

    print("Videos P005 longitudinales:")
    for quality, video in videos:
        print(f"- {quality}: {video}")
    print(f"Device: {device}")
    print(f"Regla: {config['final_rule_mode']} | GLCM: {config['glcm_region_mode']}")
    models = load_selected_models(checkpoint_paths, device)

    warmup_capture = cv2.VideoCapture(str(videos[0][1]))
    ok, warmup_frame = warmup_capture.read()
    warmup_capture.release()
    if not ok:
        raise RuntimeError(f"No se pudo leer el video de warmup: {videos[0][1]}")
    for _ in range(3):
        infer_frame(
            warmup_frame, models, device,
            threshold=args.threshold, decision_config=config,
        )

    rows = []
    best_by_quality = {}
    used_total = 0

    for video_index, (quality, video_path) in enumerate(videos):
        quota = frame_quota(args.max_frames, video_index, len(videos), used_total)
        if args.max_frames > 0 and quota <= 0:
            break
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            print(f"ADVERTENCIA: no se pudo abrir {video_path}; se omite.")
            continue
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        temporal = TemporalGuidance(
            window_size=args.temporal_window,
            required_capture_streak=args.capture_streak,
        )
        frame_index = 0
        evaluated_video = 0
        last_saved_eval = -args.capture_cooldown

        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame_index % args.frame_stride != 0:
                frame_index += 1
                continue
            if quota > 0 and evaluated_video >= quota:
                break

            if device.type == "cuda":
                torch.cuda.synchronize()
            started = time.perf_counter()
            inference = infer_frame(
                frame, models, device,
                threshold=args.threshold, decision_config=config,
            )
            temporal_result = temporal.update(inference.decision)
            if device.type == "cuda":
                torch.cuda.synchronize()
            ms_total = (time.perf_counter() - started) * 1000.0

            confirmed = int(temporal_result.capture_confirmed)
            saved_capture = 0
            capture_path = ""
            if (
                confirmed
                and evaluated_video - last_saved_eval >= args.capture_cooldown
            ):
                destination = (
                    output_dir / "captured_frames"
                    / f"{quality}_frame_{frame_index:06d}.png"
                )
                cv2.imwrite(str(destination), frame)
                capture_path = str(destination)
                saved_capture = 1
                last_saved_eval = evaluated_video

            overlay_path = ""
            overlay = create_overlay(frame, inference)
            if args.save_overlays:
                destination = (
                    output_dir / "overlays"
                    / f"{quality}_frame_{frame_index:06d}.png"
                )
                cv2.imwrite(str(destination), overlay)
                overlay_path = str(destination)

            row = {
                "quality": quality,
                "video_path": str(video_path),
                "frame_index": frame_index,
                "timestamp_s": frame_index / fps if fps > 0 else None,
                "raw_decision": inference.decision,
                "stable_decision": temporal_result.stable_decision,
                "capture_confirmed": confirmed,
                "saved_capture": saved_capture,
                "roi_present": inference.has_roi,
                "liver_present": inference.has_higado,
                "la_present": inference.has_la,
                "roi_area_px": inference.area_roi_px,
                "liver_area_px": inference.area_higado_px,
                "la_area_px": inference.area_la_px,
                "liver_roi_ratio": inference.higado_roi_ratio,
                "la_roi_ratio": inference.la_roi_ratio,
                "la_area_threshold_scaled": inference.min_la_area_px_scaled,
                "la_std": inference.la_std_intensity,
                "glcm_entropy": inference.glcm_entropy,
                "glcm_contrast": inference.glcm_contrast,
                "glcm_homogeneity": inference.glcm_homogeneity,
                "glcm_energy": inference.glcm_energy,
                "border_bright_delta": inference.border_bright_delta,
                "border_gradient_p75": inference.border_gradient_p75,
                "border_ring_bright_ratio": inference.border_ring_bright_ratio,
                "border_p90_minus_la_p10": inference.border_p90_minus_la_p10,
                "border_evidence": inference.border_evidence,
                "rule_mode": inference.rule_mode,
                "glcm_region_mode": inference.glcm_region_mode,
                "ms_total": ms_total,
                "fps_effective": 1000.0 / ms_total if ms_total > 0 else None,
                "message": temporal_result.stable_message,
                "decision_reason": inference.decision_reason,
                "base_lumen_rule": inference.base_lumen_rule,
                "final_lumen_rule": inference.final_lumen_rule,
                "la_area_ok": inference.la_area_ok,
                "la_std_ok": inference.la_std_ok,
                "la_entropy_ok": inference.la_entropy_ok,
                "border_status": inference.border_status,
                "border_all_evidence": inference.border_all_evidence,
                "capture_path": capture_path,
                "overlay_path": overlay_path,
            }
            rows.append(row)

            score = candidate_score(row)
            current = best_by_quality.get(quality)
            if current is None or score > current["score"]:
                best_by_quality[quality] = {
                    "score": score,
                    "frame": frame.copy(),
                    "overlay": overlay.copy(),
                    "frame_index": frame_index,
                }

            evaluated_video += 1
            used_total += 1
            frame_index += 1

        capture.release()
        print(f"{quality}: {evaluated_video} frames evaluados")

    results = pd.DataFrame(rows)
    if results.empty:
        raise RuntimeError("No se evaluaron frames P005.")

    for quality, candidate in best_by_quality.items():
        stem = f"{quality}_best_frame_{candidate['frame_index']:06d}"
        cv2.imwrite(
            str(output_dir / "best_candidate_frames" / f"{stem}.png"),
            candidate["frame"],
        )
        cv2.imwrite(
            str(output_dir / "best_candidate_frames" / f"{stem}_overlay.png"),
            candidate["overlay"],
        )

    summary = summarize_by_quality(results)
    decisions = decision_distribution(results)
    if args.save_csv:
        results.to_csv(
            output_dir / "frame_results.csv", index=False, encoding="utf-8-sig"
        )
        summary.to_csv(
            output_dir / "summary_by_quality.csv", index=False, encoding="utf-8-sig"
        )
        decisions.to_csv(
            output_dir / "decision_summary.csv", index=False, encoding="utf-8-sig"
        )
        verification.to_csv(
            output_dir / "model_hash_verification.csv",
            index=False, encoding="utf-8-sig",
        )

    figure_name = (
        f"p005_longitudinal_smoke_{args.max_frames}_examples.png"
        if args.max_frames > 0
        else "p005_longitudinal_final_examples.png"
    )
    figure_path = FIGURES_ROOT / figure_name
    save_example_figure(results, figure_path)

    report_path = write_report(
        output_dir, results, summary, decisions, config,
        verification, videos, args, figure_path,
    )
    print("\nResumen por calidad:")
    print(summary.to_string(index=False))
    print("\nDecisiones estables:")
    print(results["stable_decision"].value_counts(dropna=False).to_string())
    print(f"\nCSV: {output_dir / 'frame_results.csv'}")
    print(f"Reporte: {report_path}")
    print(f"Figura: {figure_path}")


if __name__ == "__main__":
    main()
