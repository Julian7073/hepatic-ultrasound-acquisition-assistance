"""Post-hoc sensitivity audit of the frozen longitudinal pipeline on P005.

This script does not select or replace a deployment threshold.  It reuses the
frozen checkpoints and the same 101 frames per quality sequence used in the
independent evaluation, caches each model probability map once, and reports
what changes when the common sigmoid threshold or one decision gate is varied.
The clear/blurry labels are used only to describe the observed trade-off.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
PIPELINE_ROOT = REPO_ROOT / "Codigos_Pipeline_Experimental_Segmentacion"
sys.path.insert(0, str(PIPELINE_ROOT))

from src.longitudinal_inference import (  # noqa: E402
    guidance_decision,
    load_selected_models,
    postprocess_masks,
    preprocess_frame,
)
from src.longitudinal_quality_rules import load_decision_config  # noqa: E402


QUALITY_ORDER = {"clear": 0, "medium": 1, "blurry": 2}
MODEL_FILES = {
    "ROI": "best_roi_model.pth",
    "Higado": "best_higado_model.pth",
    "LA": "best_la_model.pth",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--decision-config", type=Path)
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "results" / "tables")
    parser.add_argument("--frames-per-video", type=int, default=101)
    parser.add_argument(
        "--mask-thresholds",
        type=float,
        nargs="+",
        default=[0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90],
    )
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def infer_quality(path: Path) -> str | None:
    text = " ".join(part.lower() for part in path.parts)
    for quality in QUALITY_ORDER:
        if quality in text:
            return quality
    return None


def find_videos(root: Path) -> list[tuple[str, Path]]:
    candidates: list[tuple[str, Path]] = []
    for path in root.rglob("*"):
        if path.suffix.lower() not in {".mp4", ".avi", ".mov", ".mkv", ".wmv"}:
            continue
        quality = infer_quality(path)
        if quality is not None:
            candidates.append((quality, path.resolve()))
    candidates.sort(key=lambda item: (QUALITY_ORDER[item[0]], str(item[1]).lower()))
    if len(candidates) != 3:
        raise RuntimeError(f"Expected three longitudinal videos; found {len(candidates)} under {root}")
    return candidates


@torch.inference_mode()
def probability_map(model_info, frame_bgr: np.ndarray, device: torch.device) -> np.ndarray:
    tensor = preprocess_frame(frame_bgr, model_info).to(device)
    logits = model_info.model(tensor)
    probability = torch.sigmoid(logits)[0, 0].detach().cpu().numpy()
    height, width = frame_bgr.shape[:2]
    return cv2.resize(probability, (width, height), interpolation=cv2.INTER_LINEAR)


def outcome_counts(rows: pd.DataFrame) -> dict[str, float | int]:
    clear = rows[rows["quality"] == "clear"]
    blurry = rows[rows["quality"] == "blurry"]
    tp = int((clear["raw_capture"] == 1).sum())
    fn = int(len(clear) - tp)
    fp = int((blurry["raw_capture"] == 1).sum())
    tn = int(len(blurry) - fp)
    sensitivity = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    return {
        "tp": tp,
        "fn": fn,
        "fp": fp,
        "tn": tn,
        "sensitivity_pct": 100.0 * sensitivity,
        "specificity_pct": 100.0 * specificity,
        "balanced_accuracy_pct": 50.0 * (sensitivity + specificity),
    }


def gate_acceptance(row: pd.Series, omitted: str | None = None) -> int:
    gates = {
        "roi": bool(row["roi_present"]),
        "liver": bool(row["liver_present"]),
        "liver_ratio": bool(row["liver_ratio_ok"]),
        "la_presence": bool(row["la_present"]),
        "la_area": bool(row["la_area_ok"]),
        "la_std": bool(row["la_std_ok"]),
        "la_entropy": bool(row["la_entropy_ok"]),
        "border": bool(row["border_evidence"]),
    }
    if omitted in gates:
        gates[omitted] = True
    if omitted == "all_la_dependent":
        for key in ("la_presence", "la_area", "la_std", "la_entropy", "border"):
            gates[key] = True
    return int(all(gates.values()))


def summarize_thresholds(frame_rows: pd.DataFrame) -> pd.DataFrame:
    records = []
    for threshold, group in frame_rows.groupby("mask_threshold", sort=True):
        counts = outcome_counts(group)
        for quality, quality_rows in group.groupby("quality", sort=False):
            records.append({
                "mask_threshold": threshold,
                "quality": quality,
                "frames": len(quality_rows),
                "roi_present": int(quality_rows["roi_present"].sum()),
                "liver_present": int(quality_rows["liver_present"].sum()),
                "raw_la_nonempty": int((quality_rows["raw_la_area_px"] > 0).sum()),
                "raw_la_max_px": int(quality_rows["raw_la_area_px"].max()),
                "clean_la_nonempty": int((quality_rows["la_area_px"] > 0).sum()),
                "clean_la_max_px": int(quality_rows["la_area_px"].max()),
                "la_present": int(quality_rows["la_present"].sum()),
                "base_lumen_rule": int(quality_rows["base_lumen_rule"].sum()),
                "border_evidence": int(quality_rows["border_evidence"].sum()),
                "raw_capture": int(quality_rows["raw_capture"].sum()),
                **counts,
            })
    return pd.DataFrame(records)


def summarize_ablations(frame_rows: pd.DataFrame) -> pd.DataFrame:
    scenarios = [
        ("complete_rule", None),
        ("without_roi_presence", "roi"),
        ("without_liver_presence", "liver"),
        ("without_liver_ratio", "liver_ratio"),
        ("without_la_presence", "la_presence"),
        ("without_la_minimum_area", "la_area"),
        ("without_la_standard_deviation", "la_std"),
        ("without_glcm_entropy", "la_entropy"),
        ("without_border_evidence", "border"),
        ("without_all_la_dependent_conditions", "all_la_dependent"),
    ]
    records = []
    for threshold, group in frame_rows.groupby("mask_threshold", sort=True):
        for scenario, omitted in scenarios:
            evaluated = group.copy()
            evaluated["raw_capture"] = evaluated.apply(gate_acceptance, axis=1, omitted=omitted)
            counts = outcome_counts(evaluated)
            records.append({
                "mask_threshold": threshold,
                "scenario": scenario,
                "clear_accepted": int(evaluated.loc[evaluated.quality == "clear", "raw_capture"].sum()),
                "medium_accepted": int(evaluated.loc[evaluated.quality == "medium", "raw_capture"].sum()),
                "blurry_accepted": int(evaluated.loc[evaluated.quality == "blurry", "raw_capture"].sum()),
                **counts,
            })
    return pd.DataFrame(records)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    checkpoint_paths = {key: args.model_root / value for key, value in MODEL_FILES.items()}
    for path in checkpoint_paths.values():
        if not path.exists():
            raise FileNotFoundError(path)
    models = load_selected_models(checkpoint_paths, device)
    config = load_decision_config(args.decision_config)
    thresholds = sorted(set(float(value) for value in args.mask_thresholds))

    rows: list[dict] = []
    for quality, video_path in find_videos(args.video_root):
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise RuntimeError(f"Could not open {video_path}")
        frame_index = 0
        while frame_index < args.frames_per_video:
            ok, frame = capture.read()
            if not ok:
                break
            probabilities = {
                name: probability_map(model, frame, device)
                for name, model in models.items()
            }
            for threshold in thresholds:
                raw_masks = {
                    name: (probability >= threshold).astype(np.uint8)
                    for name, probability in probabilities.items()
                }
                masks = postprocess_masks(raw_masks, frame.shape, config)
                result = guidance_decision(
                    frame,
                    raw_masks,
                    masks,
                    {"ROI": 0.0, "Higado": 0.0, "LA": 0.0},
                    config,
                )
                rows.append({
                    "quality": quality,
                    "video": video_path.name,
                    "frame_index": frame_index,
                    "mask_threshold": threshold,
                    "roi_present": result.has_roi,
                    "liver_present": result.has_higado,
                    "liver_roi_ratio": result.higado_roi_ratio,
                    "liver_ratio_ok": int(result.higado_roi_ratio >= float(config["liver_min_roi_ratio"])),
                    "raw_la_area_px": result.raw_area_la_px,
                    "la_area_px": result.area_la_px,
                    "la_present": result.has_la,
                    "la_area_ok": result.la_area_ok,
                    "la_std": result.la_std_intensity,
                    "la_std_ok": result.la_std_ok,
                    "glcm_entropy": result.glcm_entropy,
                    "la_entropy_ok": result.la_entropy_ok,
                    "border_evidence": result.border_evidence,
                    "base_lumen_rule": result.base_lumen_rule,
                    "raw_capture": int(result.decision == "capture"),
                    "decision": result.decision,
                })
            frame_index += 1
        capture.release()
        if frame_index != args.frames_per_video:
            raise RuntimeError(f"Expected {args.frames_per_video} frames from {video_path}; read {frame_index}")

    frame_rows = pd.DataFrame(rows)
    frame_rows.to_csv(args.output_dir / "longitudinal_mask_threshold_sensitivity_frames.csv", index=False)
    summarize_thresholds(frame_rows).to_csv(
        args.output_dir / "longitudinal_mask_threshold_sensitivity_summary.csv", index=False
    )
    summarize_ablations(frame_rows).to_csv(
        args.output_dir / "longitudinal_gate_ablation_by_mask_threshold.csv", index=False
    )
    deployed = frame_rows[np.isclose(frame_rows["mask_threshold"], 0.50)].copy()
    pd.DataFrame([{"mask_threshold": 0.50, **outcome_counts(deployed)}]).to_csv(
        args.output_dir / "longitudinal_independent_confusion.csv", index=False
    )
    print(f"device={device}; rows={len(frame_rows)}; outputs={args.output_dir}")


if __name__ == "__main__":
    main()
