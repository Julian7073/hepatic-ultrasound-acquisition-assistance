"""Compara entrada, backbone, clasificador y contexto temporal sin usar P005."""

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
    parser = argparse.ArgumentParser()
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--minimum_action_precision", type=float, default=0.90)
    args = parser.parse_args()
    ensure_directories()

    bundles = {}
    reference_paths = None
    for variant, prefix in embedding_variants(args.stride).items():
        matrix, metadata, info = load_embedding_bundle(prefix)
        paths = metadata["image_path"].tolist()
        if reference_paths is None:
            reference_paths = paths
        elif paths != reference_paths:
            raise RuntimeError(f"El orden de imagenes difiere para {variant}.")
        bundles[variant] = (matrix, metadata, info)

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
            for view in VIEWS:
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

    external_rows = []
    external_video_parts = []
    model_manifest = []
    for _, winner in winners.iterrows():
        key = (winner["embedding_variant"], winner["temporal_mode"])
        sample_features, sample_metadata, info = prepared[key]
        view = winner["view"]
        classifier_name = winner["classifier"]
        train_mask = (
            (sample_metadata["role"] == "development")
            & (sample_metadata["view"] == view)
            & sample_metadata["quality"].isin(ANCHOR_QUALITIES)
        )
        test_mask = (
            (sample_metadata["role"] == "external_test")
            & (sample_metadata["view"] == view)
        )
        model = create_classifier(classifier_name, args.seed)
        model.fit(
            sample_features[train_mask],
            sample_metadata.loc[train_mask, "quality"].to_numpy(),
        )
        predictions, p_clear = clear_probabilities(model, sample_features[test_mask])
        test_metadata = sample_metadata.loc[test_mask].reset_index(drop=True)
        test_samples = sample_prediction_table(test_metadata, predictions, p_clear)
        test_videos = aggregate_video_predictions(test_samples)
        threshold = thresholds[thresholds["view"] == view].iloc[0]
        test_videos = add_actions(
            test_videos,
            float(threshold["adjust_threshold"]),
            float(threshold["capture_threshold"]),
        )
        test_videos.insert(0, "embedding_variant", winner["embedding_variant"])
        test_videos.insert(1, "temporal_mode", winner["temporal_mode"])
        test_videos.insert(2, "classifier", classifier_name)
        external_video_parts.append(test_videos)
        anchor_videos = test_videos[
            test_videos["true_quality"].isin(ANCHOR_QUALITIES)
        ]
        external_rows.append({
            "view": view,
            "embedding_variant": winner["embedding_variant"],
            "temporal_mode": winner["temporal_mode"],
            "classifier": classifier_name,
            "test_anchor_videos": int(len(anchor_videos)),
            **binary_metric_row(
                anchor_videos["true_quality"],
                anchor_videos["predicted_anchor"],
                "video",
            ),
            **action_metric_row(test_videos, "action"),
        })
        model_path = BINARY_MODELS_ROOT / f"{view}__binary_dinov2.joblib"
        bundle = {
            "model": model,
            "view": view,
            "classifier": classifier_name,
            "embedding_variant": winner["embedding_variant"],
            "embedding_model_id": info["model_id"],
            "embedding_dim": int(info["embedding_dim"]),
            "preprocessing": info.get("preprocessing", "full"),
            "temporal_mode": winner["temporal_mode"],
            "window_size": 5 if winner["temporal_mode"] == "window5" else 1,
            "stride": args.stride,
            "adjust_threshold": float(threshold["adjust_threshold"]),
            "capture_threshold": float(threshold["capture_threshold"]),
            "seed": args.seed,
            "labels": list(ANCHOR_QUALITIES),
            "medium_role": "uncertainty_analysis_only",
            "selection_basis": winner["selection_basis"],
            "p005_used_for_selection": False,
        }
        joblib.dump(bundle, model_path)
        model_manifest.append({
            "view": view,
            "model_path": str(model_path),
            "embedding_variant": winner["embedding_variant"],
            "embedding_model_id": info["model_id"],
            "preprocessing": bundle["preprocessing"],
            "temporal_mode": bundle["temporal_mode"],
            "classifier": classifier_name,
            "adjust_threshold": bundle["adjust_threshold"],
            "capture_threshold": bundle["capture_threshold"],
            "p005_used_for_selection": False,
        })

    external_metrics = pd.DataFrame(external_rows)
    external_videos = pd.concat(external_video_parts, ignore_index=True)
    manifest = pd.DataFrame(model_manifest)
    external_metrics.to_csv(
        BINARY_REPORTS_ROOT / "09_binary_p005_metrics.csv",
        index=False, encoding="utf-8-sig"
    )
    external_videos.to_csv(
        BINARY_REPORTS_ROOT / "09_binary_p005_video_predictions.csv",
        index=False, encoding="utf-8-sig"
    )
    manifest.to_csv(
        BINARY_REPORTS_ROOT / "09_binary_model_manifest.csv",
        index=False, encoding="utf-8-sig"
    )

    internal_action_metrics = []
    for view in VIEWS:
        data = selected_oof[selected_oof["view"] == view]
        internal_action_metrics.append({
            "view": view,
            **action_metric_row(data, "internal"),
        })
    internal_action_frame = pd.DataFrame(internal_action_metrics)
    internal_action_frame.to_csv(
        BINARY_REPORTS_ROOT / "09_binary_internal_action_metrics.csv",
        index=False, encoding="utf-8-sig"
    )

    comparison_figure = save_comparison_figure(comparison)
    confusion_figure = save_confusion_figure(selected_oof, external_videos)
    threshold_figure = save_threshold_figure(selected_oof, thresholds)
    action_figure = save_action_figure(selected_oof, external_videos)

    report = [
        "# Mejora binaria y temporal de DINOv2", "",
        "## Cambio metodologico", "",
        "El objetivo se reformula como una decision operativa: clear y blurry se usan "
        "como anclas para aprender informativo frente a no informativo. Medium no se "
        "fuerza como tercera clase; se analiza como region de incertidumbre. Esta "
        "equivalencia sigue siendo nominal y no sustituye una etiqueta clinica independiente.", "",
        "P001-P003 se usan mediante leave-one-patient-out. P005 permanece fuera de la "
        "seleccion de backbone, recorte, clasificador, modo temporal y umbrales.", "",
        "## Comparaciones realizadas", "",
        "- DINOv2-Small con frame completo.",
        "- DINOv2-Small con campo ecografico recortado.",
        "- DINOv2-Base con el mismo recorte.",
        "- Prediccion por frame frente a ventanas no solapadas de cinco embeddings.",
        "- Logistic Regression, SVM RBF, Random Forest y k-NN.", "",
        "## Configuraciones seleccionadas", "", table_markdown(winners), "",
        "## Umbrales conservadores", "", table_markdown(thresholds), "",
        "Los umbrales exigen una precision interna minima para emitir capture o adjust. "
        "Cuando la probabilidad queda entre ambos valores, el sistema se abstiene y "
        "devuelve doubtful. Medium no se usa para calcular los umbrales.", "",
        "## Comportamiento interno de acciones", "", table_markdown(internal_action_frame), "",
        "## Evaluacion externa P005", "", table_markdown(external_metrics), "",
        "P005 contiene solo un video clear, uno medium y uno blurry por vista. Cada error "
        "cambia las metricas de video en gran magnitud; el resultado sigue siendo preliminar.", "",
        "## Modelos listos para inferencia", "", table_markdown(manifest), "",
        "## Figuras", "",
        f"- Comparacion: {comparison_figure}",
        f"- Matrices de confusion: {confusion_figure}",
        f"- Umbrales: {threshold_figure}",
        f"- Distribucion de acciones: {action_figure}", "",
        "## Limitacion central", "",
        "Las etiquetas proceden de carpetas de calidad del video y no certifican por frame "
        "la presencia de referencias anatomicas. La mejora reduce ambiguedad y agrega "
        "contexto temporal, pero una futura validacion clinica requiere mas pacientes y "
        "etiquetas independientes de relevancia por clip o frame.", "",
    ]
    report_path = BINARY_REPORTS_ROOT / "10_binary_dino_technical_report.md"
    report_path.write_text("\n".join(report), encoding="utf-8")

    print("\nConfiguraciones seleccionadas:")
    print(winners[[
        "view", "embedding_variant", "temporal_mode", "classifier",
        "mean_video_f1_macro", "mean_video_blurry_recall",
    ]].to_string(index=False))
    print("\nUmbrales:")
    print(thresholds[[
        "view", "adjust_threshold", "capture_threshold",
        "internal_capture_precision", "internal_adjust_precision",
    ]].to_string(index=False))
    print("\nP005:")
    print(external_metrics[[
        "view", "video_f1_macro", "video_accuracy",
        "action_false_capture_rate_blurry", "action_medium_doubtful_rate",
    ]].to_string(index=False))
    print(f"\nReporte: {report_path}")


if __name__ == "__main__":
    main()