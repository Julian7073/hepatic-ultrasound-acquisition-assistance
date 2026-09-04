"""Reproduce development only; never predicts on the independent cohort.

Derived from the archived 08 script; uses the same feature construction, classifiers,
LOPO comparisons, tie-breaking and threshold-selection functions. New outputs are
isolated from the frozen thesis artifacts. No embedding extractor is retrained.
"""

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
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config_dino import (
    BINARY_EMBEDDINGS_ROOT,
    BINARY_FIGURES_ROOT,
    BINARY_MODELS_ROOT,
    BINARY_REPORTS_ROOT,
    EMBEDDINGS_ROOT,
    PATIENTS_DEVELOPMENT,
    VIEWS,
    ensure_directories,
)
from src.binary_temporal import (
    ANCHOR_QUALITIES,
    TEMPORAL_MODES,
    action_metric_row,
    add_actions,
    aggregate_video_predictions,
    binary_metric_row,
    calibrate_abstention_thresholds,
    clear_probabilities,
    make_temporal_samples,
    sample_prediction_table,
)
from src.classification import CLASSIFIER_NAMES, create_classifier


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


def load_embedding_bundle(prefix: Path) -> tuple[np.ndarray, pd.DataFrame, dict]:
    matrix_path = prefix.with_suffix(".npz")
    metadata_path = prefix.parent / f"{prefix.name}_metadata.csv"
    info_path = prefix.parent / f"{prefix.name}_info.json"
    for path in (matrix_path, metadata_path, info_path):
        if not path.exists():
            raise FileNotFoundError(f"Falta {path}")
    matrix = np.load(matrix_path)["embeddings"]
    metadata = pd.read_csv(metadata_path).reset_index(drop=True)
    info = json.loads(info_path.read_text(encoding="utf-8"))
    if len(matrix) != len(metadata):
        raise RuntimeError(f"Longitudes incompatibles en {prefix.name}")
    return matrix, metadata, info


def embedding_variants(stride: int) -> dict[str, Path]:
    return {
        "small_full": EMBEDDINGS_ROOT / f"dinov2_small_stride{stride}",
        "small_fan_crop": BINARY_EMBEDDINGS_ROOT / f"dinov2_small_fan_crop_stride{stride}",
        "base_fan_crop": BINARY_EMBEDDINGS_ROOT / f"dinov2_base_fan_crop_stride{stride}",
    }


def add_identifiers(
    frame: pd.DataFrame,
    variant: str,
    temporal_mode: str,
    classifier: str,
    held_out: str,
) -> pd.DataFrame:
    result = frame.copy()
    result.insert(0, "embedding_variant", variant)
    result.insert(1, "temporal_mode", temporal_mode)
    result.insert(2, "classifier", classifier)
    result.insert(3, "held_out_patient", held_out)
    return result


def save_comparison_figure(comparison: pd.DataFrame) -> Path:
    figure, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=160)
    for axis, view in zip(axes, VIEWS):
        data = comparison[comparison["view"] == view].nlargest(
            10, "mean_video_f1_macro"
        ).sort_values("mean_video_f1_macro")
        labels = (
            data["embedding_variant"]
            + " / "
            + data["temporal_mode"]
            + " / "
            + data["classifier"]
        )
        axis.barh(labels, data["mean_video_f1_macro"], color="#2A9D8F")
        axis.set_xlim(0, 1)
        axis.set_title(view)
        axis.set_xlabel("F1 macro binario por video (LOPO)")
        axis.grid(True, axis="x", alpha=0.25)
    figure.suptitle("Mejores configuraciones DINOv2: clear vs blurry")
    figure.tight_layout()
    path = BINARY_FIGURES_ROOT / "08_binary_configuration_comparison.png"
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)
    return path


def save_confusion_figure(internal: pd.DataFrame, external: pd.DataFrame) -> Path:
    figure, axes = plt.subplots(2, 3, figsize=(13, 8), dpi=160)
    labels = list(ANCHOR_QUALITIES)
    for column, view in enumerate(VIEWS):
        for row, (title, source) in enumerate((
            ("LOPO interno", internal),
            ("P005 externo", external),
        )):
            data = source[
                (source["view"] == view)
                & source["true_quality"].isin(ANCHOR_QUALITIES)
            ]
            matrix = confusion_matrix(
                data["true_quality"], data["predicted_anchor"], labels=labels
            )
            display = ConfusionMatrixDisplay(matrix, display_labels=labels)
            display.plot(ax=axes[row, column], colorbar=False, values_format="d")
            axes[row, column].set_title(f"{view} - {title}")
    figure.tight_layout()
    path = BINARY_FIGURES_ROOT / "09_binary_selected_confusion_matrices.png"
    figure.savefig(path)
    plt.close(figure)
    return path


def save_threshold_figure(internal_actions: pd.DataFrame, thresholds: pd.DataFrame) -> Path:
    colors = {"clear": "#2A9D8F", "medium": "#E9C46A", "blurry": "#C44536"}
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5), dpi=160)
    for axis, view in zip(axes, VIEWS):
        data = internal_actions[internal_actions["view"] == view]
        threshold = thresholds[thresholds["view"] == view].iloc[0]
        for quality, group in data.groupby("true_quality"):
            x = np.full(len(group), list(colors).index(quality), dtype=float)
            if len(group) > 1:
                x += np.linspace(-0.08, 0.08, len(group))
            axis.scatter(
                x,
                group["probability_clear"],
                color=colors[quality],
                label=quality,
                s=45,
                alpha=0.8,
            )
        axis.axhline(threshold["capture_threshold"], color="#2A9D8F", linestyle="--")
        axis.axhline(threshold["adjust_threshold"], color="#C44536", linestyle="--")
        axis.set_xticks(range(3), list(colors))
        axis.set_ylim(-0.02, 1.02)
        axis.set_title(view)
        axis.set_ylabel("Probabilidad de frame/clip informativo")
        axis.grid(True, axis="y", alpha=0.25)
    figure.suptitle("Calibracion interna de captura, ajuste y abstencion")
    figure.tight_layout()
    path = BINARY_FIGURES_ROOT / "09_binary_abstention_thresholds.png"
    figure.savefig(path)
    plt.close(figure)
    return path


def save_action_figure(internal: pd.DataFrame, external: pd.DataFrame) -> Path:
    figure, axes = plt.subplots(2, 3, figsize=(14, 8), dpi=160)
    actions = ["adjust", "doubtful", "capture"]
    colors = ["#C44536", "#E9C46A", "#2A9D8F"]
    for column, view in enumerate(VIEWS):
        for row, (title, source) in enumerate((("LOPO interno", internal), ("P005", external))):
            data = source[source["view"] == view]
            counts = pd.crosstab(data["true_quality"], data["action"]).reindex(
                index=["clear", "medium", "blurry"], columns=actions, fill_value=0
            )
            counts.plot(kind="bar", stacked=True, ax=axes[row, column], color=colors)
            axes[row, column].set_title(f"{view} - {title}")
            axes[row, column].set_xlabel("Calidad nominal")
            axes[row, column].set_ylabel("Videos")
            axes[row, column].tick_params(axis="x", rotation=0)
            if not (row == 0 and column == 0):
                axes[row, column].get_legend().remove()
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=3)
    figure.tight_layout(rect=(0, 0.06, 1, 1))
    path = BINARY_FIGURES_ROOT / "09_binary_action_distribution.png"
    figure.savefig(path)
    plt.close(figure)
    return path


def main() -> None:
    global EMBEDDINGS_ROOT, BINARY_EMBEDDINGS_ROOT, BINARY_REPORTS_ROOT, BINARY_MODELS_ROOT, BINARY_FIGURES_ROOT
    parser = argparse.ArgumentParser()
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--minimum_action_precision", type=float, default=0.90)
    parser.add_argument("--views", nargs="+", choices=VIEWS, default=list(VIEWS))
    parser.add_argument("--artifact-root", type=Path, default=ROOT.parent)
    parser.add_argument("--output-root", type=Path, default=ROOT.parent / "outputs" / "stage_reproduction" / "development")
    args = parser.parse_args()
    artifact = args.artifact_root.resolve() / "outputs" / "dino_experimental"
    EMBEDDINGS_ROOT = artifact / "embeddings"
    BINARY_EMBEDDINGS_ROOT = artifact / "binary_improvement" / "embeddings"
    BINARY_REPORTS_ROOT = args.output_root / "reports"
    BINARY_MODELS_ROOT = args.output_root / "models"
    BINARY_FIGURES_ROOT = args.output_root / "figures"
    for folder in (BINARY_REPORTS_ROOT, BINARY_MODELS_ROOT, BINARY_FIGURES_ROOT):
        folder.mkdir(parents=True, exist_ok=True)

    bundles = {}
    reference_paths = None
    for variant, prefix in embedding_variants(args.stride).items():
        matrix, metadata, info = load_embedding_bundle(prefix)
        paths = metadata["image_path"].tolist()
        if reference_paths is None:
            reference_paths = paths
        elif paths != reference_paths:
            raise RuntimeError(f"El orden de imagenes difiere para {variant}.")
        keep = metadata["role"].eq("development")
        if metadata.loc[keep, "patient"].eq("P005").any():
            raise ValueError("P005 cannot enter development")
        bundles[variant] = (matrix[keep], metadata.loc[keep].reset_index(drop=True), info)

    fold_rows = []
    all_video_predictions = []
    all_sample_predictions = []
    prepared = {}
    for variant, (features, metadata, info) in bundles.items():
        for temporal_mode in TEMPORAL_MODES:
            sample_features, sample_metadata = make_temporal_samples(
                features, metadata, temporal_mode
            )
            prepared[(variant, temporal_mode)] = (
                sample_features, sample_metadata, info
            )
            development = sample_metadata["role"] == "development"
            for view in args.views:
                view_mask = development & (sample_metadata["view"] == view)
                for held_out in PATIENTS_DEVELOPMENT:
                    train_mask = (
                        view_mask
                        & (sample_metadata["patient"] != held_out)
                        & sample_metadata["quality"].isin(ANCHOR_QUALITIES)
                    )
                    valid_mask = view_mask & (sample_metadata["patient"] == held_out)
                    anchor_valid_mask = valid_mask & sample_metadata["quality"].isin(
                        ANCHOR_QUALITIES
                    )
                    x_train = sample_features[train_mask]
                    y_train = sample_metadata.loc[train_mask, "quality"].to_numpy()
                    x_valid = sample_features[valid_mask]
                    valid_metadata = sample_metadata.loc[valid_mask].reset_index(drop=True)
                    if set(y_train) != set(ANCHOR_QUALITIES):
                        raise RuntimeError(
                            f"Fold sin ambas anclas: {variant}/{temporal_mode}/{view}/{held_out}"
                        )
                    for classifier_name in CLASSIFIER_NAMES:
                        model = create_classifier(classifier_name, args.seed)
                        started = time.perf_counter()
                        model.fit(x_train, y_train)
                        training_seconds = time.perf_counter() - started
                        predictions, p_clear = clear_probabilities(model, x_valid)
                        sample_table = sample_prediction_table(
                            valid_metadata, predictions, p_clear
                        )
                        video_table = aggregate_video_predictions(sample_table)
                        sample_table = add_identifiers(
                            sample_table, variant, temporal_mode, classifier_name, held_out
                        )
                        video_table = add_identifiers(
                            video_table, variant, temporal_mode, classifier_name, held_out
                        )
                        all_sample_predictions.append(sample_table)
                        all_video_predictions.append(video_table)

                        anchor_samples = sample_table[
                            sample_table["true_quality"].isin(ANCHOR_QUALITIES)
                        ]
                        anchor_videos = video_table[
                            video_table["true_quality"].isin(ANCHOR_QUALITIES)
                        ]
                        fold_rows.append({
                            "view": view,
                            "embedding_variant": variant,
                            "backbone": info.get("backbone", "small"),
                            "preprocessing": info.get("preprocessing", "full"),
                            "embedding_dim": int(info["embedding_dim"]),
                            "temporal_mode": temporal_mode,
                            "classifier": classifier_name,
                            "held_out_patient": held_out,
                            "train_anchor_samples": int(len(x_train)),
                            "valid_all_samples": int(len(x_valid)),
                            "valid_anchor_samples": int(anchor_valid_mask.sum()),
                            "training_seconds": training_seconds,
                            **binary_metric_row(
                                anchor_samples["true_quality"],
                                anchor_samples["predicted_anchor"],
                                "sample",
                            ),
                            **binary_metric_row(
                                anchor_videos["true_quality"],
                                anchor_videos["predicted_anchor"],
                                "video",
                            ),
                        })

    fold_metrics = pd.DataFrame(fold_rows)
    sample_predictions = pd.concat(all_sample_predictions, ignore_index=True)
    video_predictions = pd.concat(all_video_predictions, ignore_index=True)
    fold_metrics.to_csv(
        BINARY_REPORTS_ROOT / "08_binary_lopo_metrics_by_fold.csv",
        index=False, encoding="utf-8-sig"
    )
    sample_predictions.to_csv(
        BINARY_REPORTS_ROOT / "08_binary_lopo_sample_predictions.csv",
        index=False, encoding="utf-8-sig"
    )
    video_predictions.to_csv(
        BINARY_REPORTS_ROOT / "08_binary_lopo_video_predictions.csv",
        index=False, encoding="utf-8-sig"
    )

    comparison = (
        fold_metrics.groupby(
            [
                "view", "embedding_variant", "backbone", "preprocessing",
                "embedding_dim", "temporal_mode", "classifier",
            ],
            as_index=False,
        )
        .agg(
            mean_video_f1_macro=("video_f1_macro", "mean"),
            std_video_f1_macro=("video_f1_macro", "std"),
            mean_video_balanced_accuracy=("video_balanced_accuracy", "mean"),
            mean_video_clear_precision=("video_clear_precision", "mean"),
            mean_video_clear_recall=("video_clear_recall", "mean"),
            mean_video_blurry_recall=("video_blurry_recall", "mean"),
            mean_sample_f1_macro=("sample_f1_macro", "mean"),
            mean_training_seconds=("training_seconds", "mean"),
        )
    )
    comparison["rank"] = (
        comparison.groupby("view")["mean_video_f1_macro"]
        .rank(method="min", ascending=False)
        .astype(int)
    )
    comparison.to_csv(
        BINARY_REPORTS_ROOT / "08_binary_configuration_comparison.csv",
        index=False, encoding="utf-8-sig"
    )
    comparison["backbone_priority"] = comparison["backbone"].map(
        {"small": 0, "base": 1}
    ).fillna(2)
    comparison["preprocessing_priority"] = comparison["preprocessing"].map(
        {"fan_crop": 0, "full": 1}
    ).fillna(2)
    comparison["temporal_priority"] = comparison["temporal_mode"].map(
        {"window5": 0, "frame": 1}
    ).fillna(2)
    comparison["classifier_priority"] = comparison["classifier"].map(
        {"logistic_regression": 0, "random_forest": 1, "svm_rbf": 2, "knn": 3}
    ).fillna(4)
    winners = (
        comparison.sort_values(
            [
                "view", "mean_video_f1_macro", "mean_sample_f1_macro",
                "mean_video_blurry_recall", "mean_video_clear_precision",
                "std_video_f1_macro", "backbone_priority",
                "preprocessing_priority", "temporal_priority",
                "classifier_priority",
            ],
            ascending=[
                True, False, False, False, False, True,
                True, True, True, True,
            ],
        )
        .groupby("view", as_index=False)
        .first()
    )
    winners["selection_basis"] = (
        "LOPO video macro F1; sample F1, safety metrics, stability and deployment cost as tie-breakers"
    )
    winners["medium_used_for_training"] = False
    winners["p005_used_for_selection"] = False
    winners.to_csv(
        BINARY_REPORTS_ROOT / "08_binary_winners_by_view.csv",
        index=False, encoding="utf-8-sig"
    )

    selected_oof_parts = []
    threshold_rows = []
    for _, winner in winners.iterrows():
        selected = video_predictions[
            (video_predictions["view"] == winner["view"])
            & (video_predictions["embedding_variant"] == winner["embedding_variant"])
            & (video_predictions["temporal_mode"] == winner["temporal_mode"])
            & (video_predictions["classifier"] == winner["classifier"])
        ].copy()
        threshold = calibrate_abstention_thresholds(
            selected, minimum_action_precision=args.minimum_action_precision
        )
        threshold["view"] = winner["view"]
        threshold["embedding_variant"] = winner["embedding_variant"]
        threshold["temporal_mode"] = winner["temporal_mode"]
        threshold["classifier"] = winner["classifier"]
        threshold_rows.append(threshold)
        selected = add_actions(
            selected,
            threshold["adjust_threshold"],
            threshold["capture_threshold"],
        )
        selected_oof_parts.append(selected)

    thresholds = pd.DataFrame(threshold_rows)
    selected_oof = pd.concat(selected_oof_parts, ignore_index=True)
    thresholds.to_csv(
        BINARY_REPORTS_ROOT / "09_binary_abstention_thresholds.csv",
        index=False, encoding="utf-8-sig"
    )
    selected_oof.to_csv(
        BINARY_REPORTS_ROOT / "09_binary_selected_oof_actions.csv",
        index=False, encoding="utf-8-sig"
    )


    # Fit deployment bundles on development anchors only. No external-test
    # mask or predictions exist in this entry point.
    manifest = []
    for _, winner in winners.iterrows():
        x, metadata, info = prepared[(winner["embedding_variant"], winner["temporal_mode"])]
        view = winner["view"]
        select = metadata["view"].eq(view) & metadata["quality"].isin(ANCHOR_QUALITIES)
        assert metadata.loc[select, "role"].eq("development").all()
        assert not metadata.loc[select, "patient"].eq("P005").any()
        model = create_classifier(winner["classifier"], args.seed)
        model.fit(x[select], metadata.loc[select, "quality"].to_numpy())
        threshold = thresholds[thresholds["view"].eq(view)].iloc[0]
        bundle = {
            "model": model, "view": view, "classifier": winner["classifier"],
            "embedding_variant": winner["embedding_variant"],
            "embedding_model_id": info["model_id"],
            "embedding_dim": int(info["embedding_dim"]),
            "preprocessing": info.get("preprocessing", "full"),
            "temporal_mode": winner["temporal_mode"],
            "window_size": 5 if winner["temporal_mode"] == "window5" else 1,
            "stride": args.stride, "seed": args.seed,
            "adjust_threshold": float(threshold["adjust_threshold"]),
            "capture_threshold": float(threshold["capture_threshold"]),
            "labels": list(ANCHOR_QUALITIES),
            "medium_role": "uncertainty_analysis_only",
            "selection_basis": winner["selection_basis"],
            "p005_used_for_selection": False,
        }
        path = BINARY_MODELS_ROOT / f"{view}__binary_dinov2.joblib"
        joblib.dump(bundle, path)
        manifest.append({"view": view, "model_path": str(path), "training_samples": int(select.sum()), "p005_used": False})
    pd.DataFrame(manifest).to_csv(BINARY_REPORTS_ROOT / "development_model_manifest.csv", index=False)
    print(f"Development-only results: {args.output_root}")

if __name__ == "__main__":
    main()
