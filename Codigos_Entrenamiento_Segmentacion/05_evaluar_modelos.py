"""Evalua checkpoints entrenados en test."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from config_segmentation import (
    CHECKPOINTS_ROOT,
    DATASET_ROOTS,
    DEFAULT_BATCH_SIZE,
    DEFAULT_NUM_WORKERS,
    METRICS_ROOT,
    OVERLAYS_ROOT,
    TARGET_CLASSES,
    ensure_output_dirs,
)
from src.augmentations import get_transforms
from src.coco_dataset import BinaryCocoSegmentationDataset
from src.metrics import binary_stats, logits_to_mask
from src.models import create_model
from src.visualization import save_prediction_panel


@torch.no_grad()
def evaluate_checkpoint(checkpoint_path: Path, batch_size: int, num_workers: int, cpu: bool, overlay_examples: int) -> dict:
    """Evalua un checkpoint en test."""
    device = torch.device("cuda" if torch.cuda.is_available() and not cpu else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    architecture = checkpoint["architecture"]
    class_name = checkpoint["class_name"]
    image_size = int(checkpoint.get("image_size", 512))

    dataset = BinaryCocoSegmentationDataset(
        dataset_root=DATASET_ROOTS[class_name],
        split="test",
        class_name=class_name,
        transform=get_transforms(class_name, "test", image_size=image_size),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    model = create_model(architecture, image_size=image_size).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    totals = {"dice": 0.0, "iou": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}
    positive_totals = {"dice": 0.0, "iou": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}
    count = 0
    positive_count = 0
    empty_gt_count = 0
    empty_gt_false_positive_count = 0
    empty_gt_pred_area_total = 0.0
    positive_gt_area_total = 0.0
    positive_pred_area_total = 0.0
    inference_time = 0.0
    saved = 0
    overlay_dir = OVERLAYS_ROOT / class_name / f"{architecture}_test"

    for batch in tqdm(loader, desc=f"test {architecture} {class_name}"):
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)

        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        logits = model(images)
        if device.type == "cuda":
            torch.cuda.synchronize()
        inference_time += time.perf_counter() - start

        preds = logits_to_mask(logits)
        stats = binary_stats(preds, masks)
        batch_n = images.shape[0]
        for key in totals:
            totals[key] += float(stats[key].detach().cpu()) * batch_n
        count += batch_n

        flat_preds = preds.reshape(batch_n, -1)
        flat_masks = masks.reshape(batch_n, -1)
        pred_area = flat_preds.sum(dim=1)
        gt_area = flat_masks.sum(dim=1)
        tp = (flat_preds * flat_masks).sum(dim=1)
        fp = (flat_preds * (1.0 - flat_masks)).sum(dim=1)
        fn = ((1.0 - flat_preds) * flat_masks).sum(dim=1)
        eps = 1e-7
        positive_mask = gt_area > 0
        empty_mask = gt_area == 0

        if positive_mask.any():
            pos_n = int(positive_mask.sum().detach().cpu())
            positive_count += pos_n
            positive_gt_area_total += float(gt_area[positive_mask].sum().detach().cpu())
            positive_pred_area_total += float(pred_area[positive_mask].sum().detach().cpu())
            pos_dice = (2.0 * tp[positive_mask] + eps) / (2.0 * tp[positive_mask] + fp[positive_mask] + fn[positive_mask] + eps)
            pos_iou = (tp[positive_mask] + eps) / (tp[positive_mask] + fp[positive_mask] + fn[positive_mask] + eps)
            pos_precision = (tp[positive_mask] + eps) / (tp[positive_mask] + fp[positive_mask] + eps)
            pos_recall = (tp[positive_mask] + eps) / (tp[positive_mask] + fn[positive_mask] + eps)
            positive_totals["dice"] += float(pos_dice.sum().detach().cpu())
            positive_totals["iou"] += float(pos_iou.sum().detach().cpu())
            positive_totals["precision"] += float(pos_precision.sum().detach().cpu())
            positive_totals["recall"] += float(pos_recall.sum().detach().cpu())
            positive_totals["f1"] += float(pos_dice.sum().detach().cpu())

        if empty_mask.any():
            empty_n = int(empty_mask.sum().detach().cpu())
            empty_gt_count += empty_n
            empty_pred_area = pred_area[empty_mask]
            empty_gt_pred_area_total += float(empty_pred_area.sum().detach().cpu())
            empty_gt_false_positive_count += int((empty_pred_area > 0).sum().detach().cpu())

        for index in range(images.shape[0]):
            if saved >= overlay_examples:
                break
            filename = Path(batch["filename"][index]).stem[:80]
            save_prediction_panel(
                images[index],
                masks[index],
                preds[index],
                overlay_dir / f"{filename}_test_panel.png",
            )
            saved += 1

    row = {
        "architecture": architecture,
        "class_name": class_name,
        "checkpoint_path": str(checkpoint_path),
        "test_images": count,
        "test_positive_images": positive_count,
        "test_empty_gt_images": empty_gt_count,
        "empty_gt_false_positive_images": empty_gt_false_positive_count,
        "empty_gt_false_positive_rate": empty_gt_false_positive_count / max(empty_gt_count, 1),
        "empty_gt_mean_pred_area_px": empty_gt_pred_area_total / max(empty_gt_count, 1),
        "positive_mean_gt_area_px": positive_gt_area_total / max(positive_count, 1),
        "positive_mean_pred_area_px": positive_pred_area_total / max(positive_count, 1),
        "inference_time_s_per_frame": inference_time / max(count, 1),
        "parameter_count": checkpoint.get("parameter_count", None),
    }
    row.update({f"test_{key}": value / max(count, 1) for key, value in totals.items()})
    row.update({f"test_positive_{key}": value / max(positive_count, 1) for key, value in positive_totals.items()})
    return row


def find_checkpoints(class_name: str | None, architecture: str | None) -> list[Path]:
    """Busca solo checkpoints canonicos; los respaldos se evaluan con --checkpoint."""
    roots = [CHECKPOINTS_ROOT / class_name] if class_name else [path for path in CHECKPOINTS_ROOT.iterdir() if path.is_dir()]
    checkpoints = []
    candidate_architectures = [architecture.lower()] if architecture else ["unet", "deeplabv3", "segformer"]
    for root in roots:
        if not root.exists():
            continue
        current_class = root.name
        for model_name in candidate_architectures:
            canonical_path = root / f"{model_name}_{current_class.lower()}_best.pth"
            if canonical_path.exists():
                checkpoints.append(canonical_path)
    return sorted(checkpoints)

def main() -> None:
    """Entrada CLI."""
    ensure_output_dirs()
    parser = argparse.ArgumentParser()
    parser.add_argument("--class_name", choices=TARGET_CLASSES, default=None)
    parser.add_argument("--architecture", choices=["unet", "deeplabv3", "segformer"], default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--num_workers", type=int, default=DEFAULT_NUM_WORKERS)
    parser.add_argument("--overlay_examples", type=int, default=8)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    checkpoints = [Path(args.checkpoint)] if args.checkpoint else find_checkpoints(args.class_name, args.architecture)
    if not checkpoints:
        raise FileNotFoundError("No se encontraron checkpoints para evaluar.")

    rows = [evaluate_checkpoint(path, args.batch_size, args.num_workers, args.cpu, args.overlay_examples) for path in checkpoints]
    df_latest = pd.DataFrame(rows)

    latest_path = METRICS_ROOT / "test_metrics_latest_run.csv"
    output_path = METRICS_ROOT / "test_metrics_all_available.csv"
    df_latest.to_csv(latest_path, index=False, encoding="utf-8-sig")

    if output_path.exists():
        previous = pd.read_csv(output_path)
        combined = pd.concat([previous, df_latest], ignore_index=True)
        combined = combined.drop_duplicates(
            subset=["architecture", "class_name", "checkpoint_path"],
            keep="last",
        )
    else:
        combined = df_latest

    combined = combined.sort_values(["class_name", "architecture"]).reset_index(drop=True)
    combined.to_csv(output_path, index=False, encoding="utf-8-sig")

    print("Metricas de esta ejecucion:")
    print(df_latest.to_string(index=False))
    print("\nMetricas acumuladas disponibles:")
    print(combined.to_string(index=False))
    print(f"Metricas ultima ejecucion: {latest_path}")
    print(f"Metricas acumuladas: {output_path}")


if __name__ == "__main__":
    main()
