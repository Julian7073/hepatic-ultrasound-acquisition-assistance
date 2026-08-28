"""Compara clasificadores con LOPO y evalua P005 tras seleccionar ganadores."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config_dino import (
    EMBEDDINGS_ROOT,
    FIGURES_ROOT,
    MODELS_ROOT,
    PATIENTS_DEVELOPMENT,
    QUALITIES,
    REPORTS_ROOT,
    VIEWS,
    ensure_directories,
)
from src.classification import (
    CLASSIFIER_NAMES,
    aggregate_video_predictions,
    create_classifier,
    frame_prediction_table,
    metric_row,
    predict_with_probabilities,
)


def table_markdown(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_Sin datos._"
    display = frame.copy().fillna("")
    lines = [
        "| " + " | ".join(display.columns) + " |",
        "| " + " | ".join(["---"] * len(display.columns)) + " |",
    ]
    for row in display.astype(str).itertuples(index=False, name=None):
        lines.append("| " + " | ".join(value.replace("|", "/") for value in row) + " |")
    return "\n".join(lines)


def load_embeddings(stride: int) -> tuple[np.ndarray, pd.DataFrame, dict]:
    prefix = EMBEDDINGS_ROOT / f"dinov2_small_stride{stride}"
    matrix = np.load(prefix.with_suffix(".npz"))["embeddings"]
    metadata = pd.read_csv(prefix.parent / f"{prefix.name}_metadata.csv")
    info = json.loads((prefix.parent / f"{prefix.name}_info.json").read_text(encoding="utf-8"))
    if len(matrix) != len(metadata):
        raise RuntimeError("Embeddings y metadatos no tienen la misma longitud.")
    return matrix, metadata, info


def save_confusion_figure(
    internal_predictions: pd.DataFrame,
    external_predictions: pd.DataFrame,
) -> Path:
    figure, axes = plt.subplots(2, 3, figsize=(13, 8), dpi=160)
    for column, view in enumerate(VIEWS):
        for row, (name, frame) in enumerate((
            ("LOPO interno", internal_predictions[internal_predictions["view"] == view]),
            ("P005 externo", external_predictions[external_predictions["view"] == view]),
        )):
            matrix = confusion_matrix(
                frame["true_quality"], frame["predicted_quality"], labels=list(QUALITIES)
            )
            display = ConfusionMatrixDisplay(matrix, display_labels=list(QUALITIES))
            display.plot(ax=axes[row, column], colorbar=False, values_format="d")
            axes[row, column].set_title(f"{view} - {name}")
    figure.tight_layout()
    path = FIGURES_ROOT / "05_dino_selected_confusion_matrices.png"
    figure.savefig(path)
    plt.close(figure)
    return path


def save_comparison_figure(selection: pd.DataFrame) -> Path:
    figure, axis = plt.subplots(figsize=(10, 5), dpi=160)
    pivot = selection.pivot(
        index="view", columns="classifier", values="mean_video_f1_macro"
    )
    pivot.plot(kind="bar", ax=axis)
    axis.set_ylim(0, 1)
    axis.set_ylabel("F1 macro por video (LOPO)")
    axis.set_xlabel("Vista")
    axis.set_title("Comparacion interna de clasificadores sobre DINOv2")
    axis.grid(True, axis="y", alpha=0.3)
    figure.tight_layout()
    path = FIGURES_ROOT / "04_dino_classifier_comparison.png"
    figure.savefig(path)
    plt.close(figure)
    return path


def save_pca_figure(features: np.ndarray, metadata: pd.DataFrame) -> Path:
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.8), dpi=160)
    colors = {"clear": "#2A9D8F", "medium": "#E9C46A", "blurry": "#C44536"}
    for axis, view in zip(axes, VIEWS):
        view_mask = metadata["view"] == view
        view_features = features[view_mask]
        view_metadata = metadata.loc[view_mask].reset_index(drop=True)
        development_mask = view_metadata["role"] == "development"
        scaler = StandardScaler().fit(view_features[development_mask])
        scaled_development = scaler.transform(view_features[development_mask])
        pca = PCA(n_components=2, random_state=42).fit(scaled_development)
        projected = pca.transform(scaler.transform(view_features))
        for quality in QUALITIES:
            for role, marker, alpha in (
                ("development", "o", 0.60),
                ("external_test", "x", 0.95),
            ):
                mask = (
                    (view_metadata["quality"] == quality)
                    & (view_metadata["role"] == role)
                ).to_numpy()
                axis.scatter(
                    projected[mask, 0], projected[mask, 1],
                    s=18, alpha=alpha, marker=marker, color=colors[quality],
                    label=f"{quality}-{role}" if view == VIEWS[0] else None,
                )
        axis.set_title(view)
        axis.set_xlabel("PCA 1")
        axis.set_ylabel("PCA 2")
        axis.grid(True, alpha=0.2)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=3, fontsize=8)
    figure.tight_layout(rect=(0, 0.12, 1, 1))
    path = FIGURES_ROOT / "05_dino_pca_by_view_quality.png"
    figure.savefig(path)
    plt.close(figure)
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    ensure_directories()
    features, metadata, embedding_info = load_embeddings(args.stride)
    development_mask = metadata["role"] == "development"
    external_mask = metadata["role"] == "external_test"
    if set(metadata.loc[external_mask, "patient"].unique()) != {"P005"}:
        raise RuntimeError("La prueba externa debe contener exclusivamente P005.")

    lopo_metrics = []
    lopo_frame_predictions = []
    lopo_video_predictions = []
    for view in VIEWS:
        view_mask = development_mask & (metadata["view"] == view)
        for held_out in PATIENTS_DEVELOPMENT:
            train_mask = view_mask & (metadata["patient"] != held_out)
            valid_mask = view_mask & (metadata["patient"] == held_out)
            x_train, y_train = features[train_mask], metadata.loc[train_mask, "quality"].to_numpy()
            x_valid, y_valid = features[valid_mask], metadata.loc[valid_mask, "quality"].to_numpy()
            valid_metadata = metadata.loc[valid_mask].reset_index(drop=True)
            for classifier_name in CLASSIFIER_NAMES:
                model = create_classifier(classifier_name, args.seed)
                started = time.perf_counter()
                model.fit(x_train, y_train)
                training_s = time.perf_counter() - started
                predictions, probabilities, classes, inference_ms = predict_with_probabilities(
                    model, x_valid
                )
                frame_table = frame_prediction_table(
                    valid_metadata, predictions, probabilities, classes
                )
                frame_table.insert(0, "classifier", classifier_name)
                frame_table.insert(1, "held_out_patient", held_out)
                video_table = aggregate_video_predictions(frame_table, classes)
                video_table.insert(0, "classifier", classifier_name)
                video_table.insert(1, "held_out_patient", held_out)
                row = {
                    "view": view,
                    "classifier": classifier_name,
                    "held_out_patient": held_out,
                    "train_images": len(x_train),
                    "valid_images": len(x_valid),
                    "train_videos": int(metadata.loc[train_mask, "video_id"].nunique()),
                    "valid_videos": int(valid_metadata["video_id"].nunique()),
                    "training_seconds": training_s,
                    "inference_ms_per_image": inference_ms,
                    **metric_row(y_valid, predictions, "frame"),
                    **metric_row(
                        video_table["true_quality"],
                        video_table["predicted_quality"],
                        "video",
                    ),
                }
                lopo_metrics.append(row)
                lopo_frame_predictions.append(frame_table)
                lopo_video_predictions.append(video_table)

    metrics = pd.DataFrame(lopo_metrics)
    frame_predictions = pd.concat(lopo_frame_predictions, ignore_index=True)
    video_predictions = pd.concat(lopo_video_predictions, ignore_index=True)
    metrics.to_csv(REPORTS_ROOT / "04_dino_lopo_metrics_by_fold.csv", index=False, encoding="utf-8-sig")
    frame_predictions.to_csv(REPORTS_ROOT / "04_dino_lopo_frame_predictions.csv", index=False, encoding="utf-8-sig")
    video_predictions.to_csv(REPORTS_ROOT / "04_dino_lopo_video_predictions.csv", index=False, encoding="utf-8-sig")

    selection = (
        metrics.groupby(["view", "classifier"], as_index=False)
        .agg(
            mean_video_f1_macro=("video_f1_macro", "mean"),
            std_video_f1_macro=("video_f1_macro", "std"),
            mean_video_accuracy=("video_accuracy", "mean"),
            mean_frame_f1_macro=("frame_f1_macro", "mean"),
            mean_frame_accuracy=("frame_accuracy", "mean"),
            mean_inference_ms=("inference_ms_per_image", "mean"),
        )
    )
    selection["rank"] = (
        selection.groupby("view")["mean_video_f1_macro"]
        .rank(method="min", ascending=False)
        .astype(int)
    )
    selection.to_csv(REPORTS_ROOT / "04_dino_classifier_selection.csv", index=False, encoding="utf-8-sig")
    winners = (
        selection.sort_values(
            ["view", "mean_video_f1_macro", "mean_frame_f1_macro", "mean_inference_ms"],
            ascending=[True, False, False, True],
        )
        .groupby("view", as_index=False)
        .first()
    )
    winners["selection_basis"] = "LOPO mean video macro F1; frame F1 and speed as tie-breakers"
    winners["p005_used_for_selection"] = False
    winners.to_csv(REPORTS_ROOT / "04_dino_best_classifier_by_view.csv", index=False, encoding="utf-8-sig")

    external_metrics = []
    external_frame_predictions = []
    external_video_predictions = []
    model_manifest = []
    for _, winner in winners.iterrows():
        view = winner["view"]
        classifier_name = winner["classifier"]
        train_mask = development_mask & (metadata["view"] == view)
        test_mask = external_mask & (metadata["view"] == view)
        x_train = features[train_mask]
        y_train = metadata.loc[train_mask, "quality"].to_numpy()
        x_test = features[test_mask]
        y_test = metadata.loc[test_mask, "quality"].to_numpy()
        test_metadata = metadata.loc[test_mask].reset_index(drop=True)
        model = create_classifier(classifier_name, args.seed)
        started = time.perf_counter()
        model.fit(x_train, y_train)
        training_s = time.perf_counter() - started
        predictions, probabilities, classes, inference_ms = predict_with_probabilities(model, x_test)
        frame_table = frame_prediction_table(test_metadata, predictions, probabilities, classes)
        frame_table.insert(0, "classifier", classifier_name)
        video_table = aggregate_video_predictions(frame_table, classes)
        video_table.insert(0, "classifier", classifier_name)
        external_metrics.append({
            "view": view,
            "classifier": classifier_name,
            "train_images": len(x_train),
            "test_images": len(x_test),
            "train_videos": int(metadata.loc[train_mask, "video_id"].nunique()),
            "test_videos": int(test_metadata["video_id"].nunique()),
            "training_seconds": training_s,
            "inference_ms_per_image": inference_ms,
            **metric_row(y_test, predictions, "frame"),
            **metric_row(video_table["true_quality"], video_table["predicted_quality"], "video"),
        })
        external_frame_predictions.append(frame_table)
        external_video_predictions.append(video_table)
        model_path = MODELS_ROOT / f"{view}__{classifier_name}__dinov2_stride{args.stride}.joblib"
        bundle = {
            "model": model,
            "view": view,
            "classifier": classifier_name,
            "qualities": list(QUALITIES),
            "embedding_model_id": embedding_info["model_id"],
            "embedding_dim": embedding_info["embedding_dim"],
            "stride": args.stride,
            "seed": args.seed,
            "selection_basis": winner["selection_basis"],
        }
        joblib.dump(bundle, model_path)
        model_manifest.append({
            "view": view,
            "classifier": classifier_name,
            "model_path": str(model_path),
            "selection_video_f1_macro": winner["mean_video_f1_macro"],
            "p005_used_for_selection": False,
        })

    external_metrics_frame = pd.DataFrame(external_metrics)
    external_frames = pd.concat(external_frame_predictions, ignore_index=True)
    external_videos = pd.concat(external_video_predictions, ignore_index=True)
    external_metrics_frame.to_csv(REPORTS_ROOT / "05_dino_p005_metrics.csv", index=False, encoding="utf-8-sig")
    external_frames.to_csv(REPORTS_ROOT / "05_dino_p005_frame_predictions.csv", index=False, encoding="utf-8-sig")
    external_videos.to_csv(REPORTS_ROOT / "05_dino_p005_video_predictions.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(model_manifest).to_csv(REPORTS_ROOT / "05_dino_model_manifest.csv", index=False, encoding="utf-8-sig")

    comparison_figure = save_comparison_figure(selection)
    selected_names = set(zip(winners["view"], winners["classifier"]))
    selected_internal_frames = frame_predictions[
        frame_predictions.apply(
            lambda row: (row["view"], row["classifier"]) in selected_names,
            axis=1,
        )
    ]
    confusion_figure = save_confusion_figure(selected_internal_frames, external_frames)
    pca_figure = save_pca_figure(features, metadata)

    report = [
        "# Clasificacion de calidad con DINOv2", "",
        "## Protocolo", "",
        "DINOv2-small permanece congelado. Los clasificadores se seleccionan por vista "
        "mediante leave-one-patient-out sobre P001-P003. La metrica primaria es F1 macro "
        "por video. P005 se evalua solo despues de fijar el ganador.", "",
        "Las etiquetas clear, medium y blurry representan calidad nominal de adquisicion, "
        "no una certificacion clinica independiente.", "",
        "## Comparacion interna", "", table_markdown(selection), "",
        "## Clasificadores seleccionados", "", table_markdown(winners), "",
        "## Evaluacion externa P005", "", table_markdown(external_metrics_frame), "",
        "P005 contiene solamente tres videos por vista, uno por calidad. Las metricas por "
        "video tienen alta incertidumbre y deben interpretarse como evidencia preliminar.", "",
        "## Figuras", "",
        f"- Comparacion: {comparison_figure}",
        f"- Matrices de confusion: {confusion_figure}",
        f"- PCA: {pca_figure}", "",
    ]
    report_path = REPORTS_ROOT / "05_dino_classification_report.md"
    report_path.write_text("\n".join(report), encoding="utf-8")
    print("\nSeleccion interna:")
    print(winners[["view", "classifier", "mean_video_f1_macro", "mean_frame_f1_macro"]].to_string(index=False))
    print("\nP005 externo:")
    print(external_metrics_frame[["view", "classifier", "video_accuracy", "video_f1_macro", "frame_accuracy", "frame_f1_macro"]].to_string(index=False))
    print(f"Reporte: {report_path}")


if __name__ == "__main__":
    main()
