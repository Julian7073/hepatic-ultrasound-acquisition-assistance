"""Entrenamiento reutilizable para una arquitectura y clase."""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from config_segmentation import (
    CHECKPOINTS_ROOT,
    DATASET_ROOTS,
    DEFAULT_BATCH_SIZE,
    DEFAULT_EPOCHS,
    DEFAULT_IMAGE_SIZE,
    DEFAULT_LR,
    DEFAULT_NUM_WORKERS,
    DEFAULT_SEED,
    FIGURES_ROOT,
    LOGS_ROOT,
    METRICS_ROOT,
    OVERLAYS_ROOT,
    TARGET_CLASSES,
    ensure_output_dirs,
)
from src.augmentations import get_transforms
from src.coco_dataset import BinaryCocoSegmentationDataset
from src.metrics import binary_stats, dice_loss_from_logits, logits_to_mask
from src.models import count_parameters, create_model
from src.visualization import save_prediction_panel


def seed_everything(seed: int) -> None:
    """Fija semillas basicas."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_loader(class_name: str, split: str, image_size: int, batch_size: int, num_workers: int) -> DataLoader:
    """Construye DataLoader para un split."""
    dataset = BinaryCocoSegmentationDataset(
        dataset_root=DATASET_ROOTS[class_name],
        split=split,
        class_name=class_name,
        transform=get_transforms(class_name, split, image_size=image_size),
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def compute_loss(logits: torch.Tensor, masks: torch.Tensor, bce_loss: nn.Module) -> torch.Tensor:
    """Combina BCE y Dice loss."""
    return bce_loss(logits, masks) + dice_loss_from_logits(logits, masks)


def run_epoch(model, loader, optimizer, device, train: bool) -> dict:
    """Ejecuta una epoca de train o valid."""
    model.train(train)
    bce_loss = nn.BCEWithLogitsLoss()
    totals = {"loss": 0.0, "dice": 0.0, "iou": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}
    positive_totals = {"dice": 0.0, "iou": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}
    positive_count = 0
    empty_count = 0
    empty_false_positive_count = 0
    count = 0

    progress = tqdm(loader, desc="train" if train else "valid", leave=False)
    for batch in progress:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)

        with torch.set_grad_enabled(train):
            logits = model(images)
            loss = compute_loss(logits, masks, bce_loss)
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        pred = logits_to_mask(logits)
        stats = binary_stats(pred, masks)
        batch_size = images.shape[0]
        totals["loss"] += float(loss.detach().cpu()) * batch_size
        for key in ["dice", "iou", "precision", "recall", "f1"]:
            totals[key] += float(stats[key].detach().cpu()) * batch_size

        dims = tuple(range(1, pred.ndim))
        target_area = torch.sum(masks, dim=dims)
        pred_area = torch.sum(pred, dim=dims)
        positive = target_area > 0
        empty = target_area == 0

        if positive.any():
            pred_pos = pred[positive]
            masks_pos = masks[positive]
            pos_stats = binary_stats(pred_pos, masks_pos)
            pos_count = int(positive.sum().detach().cpu())
            for key in ["dice", "iou", "precision", "recall", "f1"]:
                positive_totals[key] += float(pos_stats[key].detach().cpu()) * pos_count
            positive_count += pos_count

        if empty.any():
            empty_count += int(empty.sum().detach().cpu())
            empty_false_positive_count += int((pred_area[empty] > 0).sum().detach().cpu())
        count += batch_size

    results = {key: value / max(count, 1) for key, value in totals.items()}
    for key, value in positive_totals.items():
        results[f"positive_{key}"] = value / max(positive_count, 1)
    results["positive_count"] = positive_count
    results["empty_count"] = empty_count
    results["empty_false_positive_rate"] = empty_false_positive_count / max(empty_count, 1)
    results["combined_la_score"] = results["positive_dice"] - results["empty_false_positive_rate"]
    return results


@torch.no_grad()
def estimate_inference_time(model, loader, device, max_batches: int = 10) -> float:
    """Estima tiempo promedio de inferencia por frame en segundos."""
    model.eval()
    total_time = 0.0
    total_frames = 0
    for batch_index, batch in enumerate(loader):
        if batch_index >= max_batches:
            break
        images = batch["image"].to(device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        _ = model(images)
        if device.type == "cuda":
            torch.cuda.synchronize()
        total_time += time.perf_counter() - start
        total_frames += images.shape[0]
    return total_time / max(total_frames, 1)


@torch.no_grad()
def save_validation_overlays(model, loader, device, output_dir: Path, max_images: int = 8) -> None:
    """Guarda ejemplos visuales de inferencia."""
    model.eval()
    saved = 0
    for batch in loader:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)
        preds = logits_to_mask(model(images))
        for index in range(images.shape[0]):
            filename = Path(batch["filename"][index]).stem[:80]
            save_prediction_panel(
                images[index],
                masks[index],
                preds[index],
                output_dir / f"{filename}_panel.png",
            )
            saved += 1
            if saved >= max_images:
                return


def plot_curves(history: list[dict], output_path: Path) -> None:
    """Guarda curvas de loss, Dice e IoU."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(history)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), dpi=150)
    for axis, metric in zip(axes, ["loss", "dice", "iou"]):
        axis.plot(df["epoch"], df[f"train_{metric}"], label=f"train_{metric}")
        axis.plot(df["epoch"], df[f"valid_{metric}"], label=f"valid_{metric}")
        axis.set_title(metric)
        axis.set_xlabel("epoch")
        axis.grid(True, alpha=0.3)
        axis.legend()
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)



def resolve_checkpoint_metric(metric: str, class_name: str) -> str:
    """Define la metrica usada para seleccionar el mejor checkpoint."""
    if metric != "auto":
        return metric
    if class_name == "LA":
        return "combined_la_score"
    return "dice"

def train_model(args: argparse.Namespace) -> Path:
    """Entrena un modelo y guarda pesos/logs."""
    ensure_output_dirs()
    seed_everything(args.seed)

    class_name = args.class_name
    architecture = args.architecture.lower()
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    train_loader = make_loader(class_name, "train", args.image_size, args.batch_size, args.num_workers)
    valid_loader = make_loader(class_name, "valid", args.image_size, args.batch_size, args.num_workers)

    model = create_model(architecture, image_size=args.image_size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    checkpoint_dir = CHECKPOINTS_ROOT / class_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"{architecture}_{class_name.lower()}_best.pth"

    log_path = LOGS_ROOT / class_name / f"{architecture}_{class_name.lower()}_training_log.csv"
    figure_path = FIGURES_ROOT / class_name / f"{architecture}_{class_name.lower()}_curves.png"
    metrics_path = METRICS_ROOT / class_name / f"{architecture}_{class_name.lower()}_valid_metrics.csv"
    overlays_dir = OVERLAYS_ROOT / class_name / architecture

    history = []
    best_valid_dice = -1.0
    best_checkpoint_score = -1.0
    best_epoch = 0
    patience_counter = 0
    parameter_count = count_parameters(model)
    checkpoint_metric = resolve_checkpoint_metric(args.checkpoint_metric, class_name)

    print(f"Arquitectura: {architecture}")
    print(f"Clase: {class_name}")
    print(f"Device: {device}")
    print(f"Parametros entrenables: {parameter_count}")
    print(f"Metrica de seleccion checkpoint: {checkpoint_metric}")
    print(f"Checkpoint: {checkpoint_path}")

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        train_metrics = run_epoch(model, train_loader, optimizer, device, train=True)
        valid_metrics = run_epoch(model, valid_loader, optimizer, device, train=False)

        row = {"epoch": epoch}
        row.update({f"train_{key}": value for key, value in train_metrics.items()})
        row.update({f"valid_{key}": value for key, value in valid_metrics.items()})
        row["parameter_count"] = parameter_count
        history.append(row)

        print(
            f"train_loss={train_metrics['loss']:.4f} valid_loss={valid_metrics['loss']:.4f} "
            f"valid_dice={valid_metrics['dice']:.4f} valid_iou={valid_metrics['iou']:.4f} "
            f"valid_positive_dice={valid_metrics['positive_dice']:.4f} "
            f"valid_empty_fp_rate={valid_metrics['empty_false_positive_rate']:.4f}"
        )

        checkpoint_score = valid_metrics[checkpoint_metric]
        if checkpoint_score > best_checkpoint_score:
            best_checkpoint_score = checkpoint_score
            best_valid_dice = valid_metrics["dice"]
            best_epoch = epoch
            patience_counter = 0
            torch.save(
                {
                    "architecture": architecture,
                    "class_name": class_name,
                    "image_size": args.image_size,
                    "model_state_dict": model.state_dict(),
                    "valid_dice": valid_metrics["dice"],
                    "valid_positive_dice": valid_metrics["positive_dice"],
                    "valid_empty_false_positive_rate": valid_metrics["empty_false_positive_rate"],
                    "checkpoint_metric": checkpoint_metric,
                    "checkpoint_score": best_checkpoint_score,
                    "epoch": epoch,
                    "parameter_count": parameter_count,
                },
                checkpoint_path,
            )
            print(
                f"Nuevo mejor checkpoint guardado: {checkpoint_metric}={best_checkpoint_score:.4f} "
                f"Dice={valid_metrics['dice']:.4f} positive_Dice={valid_metrics['positive_dice']:.4f}"
            )
        else:
            patience_counter += 1

        if args.early_stopping_patience > 0 and patience_counter >= args.early_stopping_patience:
            print(f"Early stopping activado en epoch {epoch}. Mejor epoch: {best_epoch}")
            break

    pd.DataFrame(history).to_csv(log_path, index=False, encoding="utf-8-sig")
    pd.DataFrame(history).tail(1).to_csv(metrics_path, index=False, encoding="utf-8-sig")
    plot_curves(history, figure_path)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    inference_time = estimate_inference_time(model, valid_loader, device)
    save_validation_overlays(model, valid_loader, device, overlays_dir, max_images=args.overlay_examples)

    summary_path = METRICS_ROOT / class_name / f"{architecture}_{class_name.lower()}_summary.csv"
    pd.DataFrame(
        [
            {
                "architecture": architecture,
                "class_name": class_name,
                "best_epoch": best_epoch,
                "best_valid_dice": best_valid_dice,
                "checkpoint_metric": checkpoint_metric,
                "best_checkpoint_score": best_checkpoint_score,
                "inference_time_s_per_frame_valid": inference_time,
                "parameter_count": parameter_count,
                "checkpoint_path": str(checkpoint_path),
            }
        ]
    ).to_csv(summary_path, index=False, encoding="utf-8-sig")

    print(f"\nEntrenamiento terminado.")
    print(f"Mejor checkpoint: {checkpoint_metric}={best_checkpoint_score:.4f} en epoch {best_epoch}")
    print(f"Dice valid del checkpoint seleccionado: {best_valid_dice:.4f}")
    print(f"Tiempo inferencia valid/frame: {inference_time:.6f} s")
    print(f"Log: {log_path}")
    print(f"Curvas: {figure_path}")
    print(f"Overlays: {overlays_dir}")
    return checkpoint_path


def build_parser(default_architecture: str) -> argparse.ArgumentParser:
    """Parser comun para scripts de entrenamiento."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--architecture", default=default_architecture)
    parser.add_argument("--class_name", choices=TARGET_CLASSES, required=True)
    parser.add_argument("--image_size", type=int, default=DEFAULT_IMAGE_SIZE)
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=DEFAULT_NUM_WORKERS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--early_stopping_patience", type=int, default=8)
    parser.add_argument("--overlay_examples", type=int, default=8)
    parser.add_argument(
        "--checkpoint_metric",
        choices=["auto", "dice", "positive_dice", "positive_iou", "combined_la_score"],
        default="auto",
        help="Metrica para seleccionar checkpoint. auto usa combined_la_score en LA y dice en ROI/Higado.",
    )
    parser.add_argument("--cpu", action="store_true")
    return parser


def main(default_architecture: str) -> None:
    """Entrada reusable."""
    parser = build_parser(default_architecture)
    args = parser.parse_args()
    train_model(args)
