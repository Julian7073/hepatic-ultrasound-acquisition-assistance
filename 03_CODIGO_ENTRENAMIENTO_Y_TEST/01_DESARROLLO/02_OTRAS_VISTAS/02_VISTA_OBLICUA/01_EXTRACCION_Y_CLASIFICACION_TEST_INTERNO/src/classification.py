"""Clasificadores y metricas para embeddings DINOv2."""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from config_dino import QUALITIES


CLASSIFIER_NAMES = ("logistic_regression", "svm_rbf", "random_forest", "knn")


def create_classifier(name: str, seed: int):
    if name == "logistic_regression":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(
                C=1.0,
                max_iter=5000,
                class_weight="balanced",
                random_state=seed,
            )),
        ])
    if name == "svm_rbf":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", SVC(
                C=1.0,
                kernel="rbf",
                gamma="scale",
                class_weight="balanced",
                random_state=seed,
            )),
        ])
    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=seed,
            n_jobs=-1,
        )
    if name == "knn":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", KNeighborsClassifier(
                n_neighbors=7,
                weights="distance",
            )),
        ])
    raise ValueError(f"Clasificador no soportado: {name}")


def metric_row(y_true, y_pred, prefix: str) -> dict:
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=list(QUALITIES),
        zero_division=0,
    )
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=list(QUALITIES),
        average="macro",
        zero_division=0,
    )
    result = {
        f"{prefix}_accuracy": float(accuracy_score(y_true, y_pred)),
        f"{prefix}_precision_macro": float(macro_precision),
        f"{prefix}_recall_macro": float(macro_recall),
        f"{prefix}_f1_macro": float(macro_f1),
        f"{prefix}_samples": int(len(y_true)),
    }
    for index, quality in enumerate(QUALITIES):
        result.update({
            f"{prefix}_{quality}_precision": float(precision[index]),
            f"{prefix}_{quality}_recall": float(recall[index]),
            f"{prefix}_{quality}_f1": float(f1[index]),
            f"{prefix}_{quality}_support": int(support[index]),
        })
    return result


def predict_with_probabilities(
    model, features: np.ndarray
) -> tuple[np.ndarray, np.ndarray, list[str], float]:
    started = time.perf_counter()
    classes = [str(value) for value in model.classes_]
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(features)
        predictions = np.asarray(classes)[np.argmax(probabilities, axis=1)]
    elif hasattr(model, "decision_function"):
        scores = np.asarray(model.decision_function(features), dtype=np.float64)
        if scores.ndim == 1:
            scores = np.column_stack([-scores, scores])
        scores -= scores.max(axis=1, keepdims=True)
        exponentials = np.exp(scores)
        probabilities = exponentials / exponentials.sum(axis=1, keepdims=True)
        predictions = model.predict(features)
    else:
        raise TypeError("El clasificador no expone predict_proba ni decision_function.")
    elapsed_s = time.perf_counter() - started
    per_image_ms = elapsed_s * 1000.0 / max(len(features), 1)
    return predictions, probabilities, classes, per_image_ms


def frame_prediction_table(
    metadata: pd.DataFrame,
    predictions: np.ndarray,
    probabilities: np.ndarray,
    classes: list[str],
) -> pd.DataFrame:
    columns = [
        "patient", "role", "view", "quality", "video_id", "frame_number",
        "filename", "image_path",
    ]
    frame = metadata[columns].reset_index(drop=True).copy()
    frame["true_quality"] = metadata["quality"].to_numpy()
    frame["predicted_quality"] = predictions
    for index, class_name in enumerate(classes):
        frame[f"probability_{class_name}"] = probabilities[:, index]
    frame["correct"] = (frame["true_quality"] == frame["predicted_quality"]).astype(int)
    return frame


def aggregate_video_predictions(
    frame_predictions: pd.DataFrame,
    classes: list[str],
) -> pd.DataFrame:
    probability_columns = [f"probability_{class_name}" for class_name in classes]
    aggregations = {
        "patient": "first",
        "role": "first",
        "view": "first",
        "true_quality": "first",
        "filename": "size",
        **{column: "mean" for column in probability_columns},
    }
    video = (
        frame_predictions.groupby("video_id", as_index=False)
        .agg(aggregations)
        .rename(columns={"filename": "frame_count"})
    )
    values = video[probability_columns].to_numpy()
    video["predicted_quality"] = np.asarray(classes)[np.argmax(values, axis=1)]
    video["correct"] = (video["true_quality"] == video["predicted_quality"]).astype(int)
    return video
