"""Entrenamiento, validacion y test compartidos por las tres arquitecturas."""

from __future__ import annotations

import time
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
from tqdm import tqdm

from src.losses import combined_loss
from src.metrics import logits_to_mask, per_image_rows, summarize_rows


def resolve_checkpoint_metric(requested: str, class_name: str) -> str:
    """Selecciona Dice para ROI/Higado y combined_la_score para LA."""
    if requested != "auto":
        return requested
    return "combined_la_score" if class_name == "LA" else "dice"


def metric_key(metric: str) -> str:
    """Mapea el nombre CLI a la columna de validacion."""
    return {
        "dice": "valid_dice",
        "positive_dice": "valid_positive_dice",
        "combined_la_score": "valid_combined_la_score",
    }[metric]


def run_loader(model, loader, device, optimizer=None, prefix: str = "valid") -> tuple[dict, list[dict]]:
    """Ejecuta un loader y devuelve resumen mas metricas por imagen."""
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_images = 0
    rows = []
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch in tqdm(loader, desc=prefix, leave=False):
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            logits = model(images)
            loss = combined_loss(logits, masks)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            batch_size = images.shape[0]
            total_loss += float(loss.detach().cpu()) * batch_size
            total_images += batch_size
            predictions = logits_to_mask(logits)
            rows.extend(per_image_rows(predictions, masks, list(batch["filename"])))
    summary = summarize_rows(rows, prefix)
    summary[f"{prefix}_loss"] = total_loss / max(total_images, 1)
    return summary, rows


def plot_curves(history: pd.DataFrame, output_path: Path) -> None:
    """Genera curvas de loss, Dice e IoU."""
    figure, axes = plt.subplots(1, 3, figsize=(13, 4), dpi=150)
    for axis, metric in zip(axes, ("loss", "dice", "iou")):
        axis.plot(history["epoch"], history[f"train_{metric}"], label="train")
        axis.plot(history["epoch"], history[f"valid_{metric}"], label="valid")
        axis.set_title(metric)
        axis.set_xlabel("Epoca")
        axis.grid(True, alpha=0.3)
        axis.legend()
    figure.tight_layout()
    figure.savefig(output_path)
    plt.close(figure)


def fit_model(model, train_loader, valid_loader, device, config: dict, experiment_dir: Path) -> tuple[Path, pd.DataFrame, dict, list[dict]]:
    """Entrena y conserva el checkpoint con mejor metrica de validacion."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    selected_metric = resolve_checkpoint_metric(config["checkpoint_metric"], config["class_name"])
    selected_key = metric_key(selected_metric)
    checkpoint_path = experiment_dir / "best_model.pth"
    history_rows = []
    best_score = float("-inf")
    best_validation_loss = float("inf")
    patience_counter = 0
    best_validation = {}
    best_validation_rows = []

    for epoch in range(1, config["epochs"] + 1):
        print(f"\nEpoca {epoch}/{config['epochs']}")
        train_metrics, _ = run_loader(model, train_loader, device, optimizer=optimizer, prefix="train")
        valid_metrics, valid_rows = run_loader(model, valid_loader, device, optimizer=None, prefix="valid")
        row = {"epoch": epoch, **train_metrics, **valid_metrics}
        history_rows.append(row)
        score = float(valid_metrics[selected_key])
        print(
            f"train_loss={train_metrics['train_loss']:.4f} "
            f"valid_loss={valid_metrics['valid_loss']:.4f} "
            f"valid_dice={valid_metrics['valid_dice']:.4f} "
            f"valid_iou={valid_metrics['valid_iou']:.4f} "
            f"selection={selected_metric}:{score:.4f}"
        )
        valid_loss = float(valid_metrics["valid_loss"])
        improved_score = score > best_score + config["checkpoint_min_delta"]
        near_tie_lower_loss = abs(score - best_score) <= config["checkpoint_min_delta"] and valid_loss < best_validation_loss
        if improved_score or near_tie_lower_loss:
            best_score = score
            best_validation_loss = valid_loss
            patience_counter = 0
            best_validation = valid_metrics.copy()
            best_validation_rows = valid_rows
            torch.save({
                "model_state_dict": model.state_dict(),
                "architecture": config["architecture"],
                "class_name": config["class_name"],
                "image_size": config["image_size"],
                "resize_mode": config["resize_mode"],
                "pretrained": config["pretrained"],
                "model_metadata": config["model_metadata"],
                "checkpoint_metric": selected_metric,
                "checkpoint_score": best_score,
                "valid_loss": best_validation_loss,
                "checkpoint_min_delta": config["checkpoint_min_delta"],
                "epoch": epoch,
                "config": config,
            }, checkpoint_path)
            print(f"Nuevo mejor checkpoint: {selected_metric}={best_score:.4f}")
        else:
            patience_counter += 1

        if config["early_stopping_patience"] > 0 and patience_counter >= config["early_stopping_patience"]:
            print(f"Early stopping en epoca {epoch}; sin mejora relevante durante {patience_counter} epocas.")
            break

    history = pd.DataFrame(history_rows)
    history.to_csv(experiment_dir / "train_log.csv", index=False, encoding="utf-8-sig")
    plot_curves(history, experiment_dir / "curves.png")
    validation_df = pd.DataFrame([{
        "best_epoch": torch.load(checkpoint_path, map_location="cpu", weights_only=False)["epoch"],
        "checkpoint_metric": selected_metric,
        "checkpoint_score": best_score,
        **best_validation,
    }])
    validation_df.to_csv(experiment_dir / "validation_metrics.csv", index=False, encoding="utf-8-sig")
    return checkpoint_path, history, validation_df.iloc[0].to_dict(), best_validation_rows


def load_best_weights(model, checkpoint_path: Path, device) -> dict:
    """Restaura el mejor checkpoint del experimento."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    return checkpoint


@torch.no_grad()
def evaluate_model(model, loader, device, prefix: str) -> tuple[dict, list[dict]]:
    """Evalua sin actualizar pesos."""
    return run_loader(model, loader, device, optimizer=None, prefix=prefix)
