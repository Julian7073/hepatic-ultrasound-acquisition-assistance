"""Utilidades para calidad binaria, ventanas temporales y abstencion."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_recall_fscore_support,
)


ANCHOR_QUALITIES = ("blurry", "clear")
TEMPORAL_MODES = ("frame", "window5")


def make_temporal_samples(
    features: np.ndarray,
    metadata: pd.DataFrame,
    mode: str,
    window_size: int = 5,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Convierte embeddings de frame en frames o ventanas no solapadas."""
    if len(features) != len(metadata):
        raise ValueError("Features y metadata deben tener igual longitud.")
    if mode == "frame":
        sample_metadata = metadata.reset_index(drop=True).copy()
        sample_metadata["sample_id"] = (
            sample_metadata["video_id"].astype(str)
            + "__frame_"
            + sample_metadata["frame_number"].astype(str)
        )
        sample_metadata["sample_frame_count"] = 1
        return features.astype(np.float32, copy=False), sample_metadata
    if mode != "window5":
        raise ValueError(f"Modo temporal no soportado: {mode}")

    sample_features = []
    rows = []
    for video_id, group in metadata.groupby("video_id", sort=True):
        ordered = group.sort_values(["frame_number", "filename"])
        indices = ordered.index.to_numpy()
        for start in range(0, len(indices) - window_size + 1, window_size):
            selected = indices[start:start + window_size]
            values = features[selected]
            vector = np.concatenate([values.mean(axis=0), values.std(axis=0)])
            first = metadata.loc[selected[0]].to_dict()
            first.update({
                "sample_id": f"{video_id}__window_{start:03d}",
                "sample_frame_count": int(len(selected)),
                "frame_number": int(metadata.loc[selected, "frame_number"].min()),
                "window_last_frame": int(metadata.loc[selected, "frame_number"].max()),
            })
            sample_features.append(vector.astype(np.float32))
            rows.append(first)
    if not sample_features:
        return np.empty((0, features.shape[1] * 2), dtype=np.float32), pd.DataFrame()
    return np.stack(sample_features), pd.DataFrame(rows).reset_index(drop=True)


def clear_probabilities(model, features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Devuelve prediccion binaria y probabilidad comparable de clear."""
    classes = [str(value) for value in model.classes_]
    if "clear" not in classes or "blurry" not in classes:
        raise RuntimeError(f"Clases binarias incompletas: {classes}")
    if hasattr(model, "predict_proba"):
        probabilities = np.asarray(model.predict_proba(features), dtype=np.float64)
    elif hasattr(model, "decision_function"):
        scores = np.asarray(model.decision_function(features), dtype=np.float64)
        if scores.ndim == 1:
            scores = np.column_stack([-scores, scores])
        scores -= scores.max(axis=1, keepdims=True)
        exponentials = np.exp(scores)
        probabilities = exponentials / exponentials.sum(axis=1, keepdims=True)
    else:
        raise TypeError("El modelo no expone probabilidad ni decision_function.")
    clear_index = classes.index("clear")
    p_clear = probabilities[:, clear_index]
    predictions = np.where(p_clear >= 0.5, "clear", "blurry")
    return predictions, p_clear


def sample_prediction_table(
    metadata: pd.DataFrame,
    predictions: np.ndarray,
    p_clear: np.ndarray,
) -> pd.DataFrame:
    columns = [
        "patient", "role", "view", "quality", "video_id", "sample_id",
        "sample_frame_count", "frame_number", "filename", "image_path",
    ]
    available = [column for column in columns if column in metadata.columns]
    result = metadata[available].reset_index(drop=True).copy()
    result["true_quality"] = metadata["quality"].to_numpy()
    result["predicted_anchor"] = predictions
    result["probability_clear"] = p_clear
    return result


def aggregate_video_predictions(samples: pd.DataFrame) -> pd.DataFrame:
    """Promedia probabilidades dentro de cada video."""
    video = (
        samples.groupby("video_id", as_index=False)
        .agg(
            patient=("patient", "first"),
            role=("role", "first"),
            view=("view", "first"),
            true_quality=("true_quality", "first"),
            sample_count=("sample_id", "size"),
            source_frame_count=("sample_frame_count", "sum"),
            probability_clear=("probability_clear", "mean"),
            probability_clear_std=("probability_clear", "std"),
        )
    )
    video["probability_clear_std"] = video["probability_clear_std"].fillna(0.0)
    video["predicted_anchor"] = np.where(
        video["probability_clear"] >= 0.5, "clear", "blurry"
    )
    return video


def binary_metric_row(y_true, y_pred, prefix: str) -> dict:
    labels = list(ANCHOR_QUALITIES)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="macro", zero_division=0
    )
    result = {
        f"{prefix}_accuracy": float(accuracy_score(y_true, y_pred)),
        f"{prefix}_balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        f"{prefix}_precision_macro": float(macro_precision),
        f"{prefix}_recall_macro": float(macro_recall),
        f"{prefix}_f1_macro": float(macro_f1),
        f"{prefix}_samples": int(len(y_true)),
    }
    for index, label in enumerate(labels):
        result.update({
            f"{prefix}_{label}_precision": float(precision[index]),
            f"{prefix}_{label}_recall": float(recall[index]),
            f"{prefix}_{label}_f1": float(f1[index]),
            f"{prefix}_{label}_support": int(support[index]),
        })
    return result


def calibrate_abstention_thresholds(
    anchor_video_predictions: pd.DataFrame,
    minimum_action_precision: float = 0.90,
) -> dict:
    """Fija umbrales conservadores usando solo videos clear y blurry internos."""
    frame = anchor_video_predictions[
        anchor_video_predictions["true_quality"].isin(ANCHOR_QUALITIES)
    ].copy()
    if frame.empty:
        raise ValueError("No hay videos ancla para calibrar umbrales.")

    probabilities = frame["probability_clear"].to_numpy(dtype=float)
    truths = frame["true_quality"].to_numpy()
    grid = np.linspace(0.01, 0.99, 99)

    high_candidates = []
    for threshold in grid[grid >= 0.5]:
        acted = probabilities >= threshold
        if not acted.any():
            continue
        precision = float((truths[acted] == "clear").mean())
        clear_coverage = float(((truths == "clear") & acted).sum() / max((truths == "clear").sum(), 1))
        false_capture_rate = float(((truths == "blurry") & acted).sum() / max((truths == "blurry").sum(), 1))
        high_candidates.append((threshold, precision, clear_coverage, false_capture_rate))
    valid_high = [row for row in high_candidates if row[1] >= minimum_action_precision]
    if valid_high:
        high = sorted(valid_high, key=lambda row: (-row[2], row[0]))[0]
    else:
        high = sorted(high_candidates, key=lambda row: (row[3], -row[1], -row[2]))[0]

    low_candidates = []
    for threshold in grid[grid <= 0.5]:
        acted = probabilities <= threshold
        if not acted.any():
            continue
        precision = float((truths[acted] == "blurry").mean())
        blurry_coverage = float(((truths == "blurry") & acted).sum() / max((truths == "blurry").sum(), 1))
        false_adjust_rate = float(((truths == "clear") & acted).sum() / max((truths == "clear").sum(), 1))
        low_candidates.append((threshold, precision, blurry_coverage, false_adjust_rate))
    valid_low = [row for row in low_candidates if row[1] >= minimum_action_precision]
    if valid_low:
        low = sorted(valid_low, key=lambda row: (-row[2], -row[0]))[0]
    else:
        low = sorted(low_candidates, key=lambda row: (row[3], -row[1], -row[2]))[0]

    low_threshold = float(low[0])
    high_threshold = float(high[0])
    if low_threshold >= high_threshold:
        low_threshold, high_threshold = 0.35, 0.65

    return {
        "adjust_threshold": low_threshold,
        "capture_threshold": high_threshold,
        "minimum_action_precision": float(minimum_action_precision),
        "internal_adjust_precision": float(low[1]),
        "internal_blurry_coverage": float(low[2]),
        "internal_false_adjust_clear_rate": float(low[3]),
        "internal_capture_precision": float(high[1]),
        "internal_clear_coverage": float(high[2]),
        "internal_false_capture_blurry_rate": float(high[3]),
        "calibration_videos": int(len(frame)),
        "medium_used_for_calibration": False,
        "p005_used_for_calibration": False,
    }


def add_actions(
    predictions: pd.DataFrame,
    adjust_threshold: float,
    capture_threshold: float,
) -> pd.DataFrame:
    result = predictions.copy()
    probability = result["probability_clear"].to_numpy(dtype=float)
    result["action"] = np.select(
        [probability >= capture_threshold, probability <= adjust_threshold],
        ["capture", "adjust"],
        default="doubtful",
    )
    expected = result["true_quality"].map({
        "clear": "capture",
        "blurry": "adjust",
        "medium": "doubtful",
    })
    result["expected_action_for_analysis"] = expected
    result["action_matches_nominal_quality"] = (
        result["action"] == result["expected_action_for_analysis"]
    ).astype(int)
    return result


def action_metric_row(video_predictions: pd.DataFrame, prefix: str) -> dict:
    anchors = video_predictions[
        video_predictions["true_quality"].isin(ANCHOR_QUALITIES)
    ]
    medium = video_predictions[video_predictions["true_quality"] == "medium"]
    blurry = anchors[anchors["true_quality"] == "blurry"]
    clear = anchors[anchors["true_quality"] == "clear"]
    return {
        f"{prefix}_videos": int(len(video_predictions)),
        f"{prefix}_anchor_videos": int(len(anchors)),
        f"{prefix}_capture_rate_clear": float((clear["action"] == "capture").mean()) if len(clear) else np.nan,
        f"{prefix}_false_capture_rate_blurry": float((blurry["action"] == "capture").mean()) if len(blurry) else np.nan,
        f"{prefix}_adjust_rate_blurry": float((blurry["action"] == "adjust").mean()) if len(blurry) else np.nan,
        f"{prefix}_false_adjust_rate_clear": float((clear["action"] == "adjust").mean()) if len(clear) else np.nan,
        f"{prefix}_anchor_abstention_rate": float((anchors["action"] == "doubtful").mean()) if len(anchors) else np.nan,
        f"{prefix}_medium_doubtful_rate": float((medium["action"] == "doubtful").mean()) if len(medium) else np.nan,
    }