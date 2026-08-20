"""Punto unico de entrada para experimentos de segmentacion longitudinal."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

PIPELINE_ROOT = Path(__file__).resolve().parent
if str(PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(PIPELINE_ROOT))

from config_experimental import (
    ARCHITECTURES,
    CLASS_NAMES,
    DATASET_ROOTS,
    EXPERIMENTS_ROOT,
    RESIZE_MODES,
    ensure_directories,
)
from src.benchmark import benchmark_model
from src.dataset_coco import (
    BinaryCocoDataset,
    assign_grouped_splits,
    build_balanced_sampler,
    build_roi_mask_index,
    load_records,
)
from src.models import count_parameters, create_model
from src.overlays import save_selected_overlays
from src.reports import update_global_reports, write_experiment_summary
from src.split_audit import run_audit
from src.train_eval import evaluate_model, fit_model, load_best_weights
from src.transforms import build_transforms
from src.utils import environment_info, parse_bool, save_json, seed_everything, unique_directory


def build_parser() -> argparse.ArgumentParser:
    """Define la interfaz reproducible del pipeline."""
    parser = argparse.ArgumentParser(description="Pipeline experimental de segmentacion longitudinal")
    parser.add_argument("--class_name", choices=CLASS_NAMES, required=True)
    parser.add_argument("--architecture", choices=ARCHITECTURES, required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--resize_mode", choices=RESIZE_MODES, default="full_resize")
    parser.add_argument("--augmentation", choices=("none", "x4", "positive_x4"), default="none")
    parser.add_argument("--sampling_strategy", choices=("natural", "balanced_la"), default="natural")
    parser.add_argument("--pretrained", type=parse_bool, default=False)
    parser.add_argument(
        "--checkpoint_metric",
        choices=("auto", "dice", "positive_dice", "combined_la_score"),
        default="auto",
    )
    parser.add_argument("--experiment_name", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--save_overlays", action="store_true")
    parser.add_argument("--max_overlay_samples", type=int, default=8)
    parser.add_argument("--run_test", type=parse_bool, default=True)
    parser.add_argument("--run_benchmark", type=parse_bool, default=True)
    parser.add_argument("--split_strategy", choices=("coco", "group_video", "group_patient"), default="coco")
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--early_stopping_patience", type=int, default=10)
    parser.add_argument("--checkpoint_min_delta", type=float, default=1e-4)
    parser.add_argument("--cpu", action="store_true")
    return parser


def make_loader(dataset, batch_size: int, shuffle: bool, num_workers: int, sampler=None) -> DataLoader:
    """Construye DataLoader compatible con Windows."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def main() -> None:
    """Audita, entrena, valida, prueba, mide y reporta un experimento."""
    args = build_parser().parse_args()
    if args.augmentation == "positive_x4" and args.class_name != "LA":
        raise ValueError("positive_x4 solo puede usarse con class_name LA.")
    if args.augmentation == "positive_x4" and args.sampling_strategy != "natural":
        raise ValueError(
            "Use positive_x4 con sampling_strategy natural para evitar doble rebalanceo."
        )
    ensure_directories()
    seed_everything(args.seed)

    print("Ejecutando auditoria previa de datasets y leakage...")
    audit_df, leakage_df = run_audit()
    invalid = audit_df[
        (~audit_df["coco_exists"])
        | (~audit_df["single_expected_class"])
        | (audit_df["missing_image_files"] > 0)
    ]
    if not invalid.empty:
        print(invalid.to_string(index=False))
        raise RuntimeError("La auditoria encontro problemas estructurales. No se inicia entrenamiento.")

    experiment_dir = unique_directory(EXPERIMENTS_ROOT / args.experiment_name)
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    records = load_records(DATASET_ROOTS[args.class_name], args.class_name)
    split_records = assign_grouped_splits(records, args.split_strategy, args.seed)
    for split, items in split_records.items():
        if not items:
            raise ValueError(f"El split {split} quedo vacio con estrategia {args.split_strategy}.")

    base_transform, augmented_transform, multiplier = build_transforms(
        args.class_name,
        args.image_size,
        args.resize_mode,
        args.augmentation,
    )
    roi_index = build_roi_mask_index() if args.resize_mode == "roi_crop_resize" else None

    datasets = {}
    for split in ("train", "valid", "test"):
        datasets[split] = BinaryCocoDataset(
            records=split_records[split],
            class_name=args.class_name,
            split=split,
            transform_base=base_transform,
            transform_augmented=augmented_transform if split == "train" else None,
            augmentation_multiplier=multiplier if split == "train" else 1,
            augmentation_scope=(
                "positive_only" if split == "train" and args.augmentation == "positive_x4"
                else "all"
            ),
            resize_mode=args.resize_mode,
            roi_index=roi_index,
        )

    train_sampler = None
    sampling_metadata = {"strategy": args.sampling_strategy}
    if args.sampling_strategy == "balanced_la":
        if args.class_name != "LA":
            raise ValueError("balanced_la solo puede usarse con class_name LA.")
        train_sampler, sampler_details = build_balanced_sampler(datasets["train"], args.seed)
        sampling_metadata.update(sampler_details)

    train_loader = make_loader(datasets["train"], args.batch_size, True, args.num_workers, sampler=train_sampler)
    valid_loader = make_loader(datasets["valid"], args.batch_size, False, args.num_workers)
    test_loader = make_loader(datasets["test"], args.batch_size, False, args.num_workers)
    benchmark_loader = make_loader(datasets["test"], 1, False, args.num_workers)

    model, model_metadata = create_model(args.architecture, args.pretrained)
    model = model.to(device)
    config = {
        **vars(args),
        "experiment_name_requested": args.experiment_name,
        "experiment_name_actual": experiment_dir.name,
        "experiment_path": str(experiment_dir),
        "dataset_root": str(DATASET_ROOTS[args.class_name]),
        "split_sizes": {key: len(value) for key, value in split_records.items()},
        "effective_train_samples": len(datasets["train"]),
        "augmentation_multiplier": multiplier,
        "augmentation_metadata": datasets["train"].augmentation_metadata(),
        "sampling_metadata": sampling_metadata,
        "device": str(device),
        "parameter_count": count_parameters(model),
        "model_metadata": model_metadata,
        "environment": environment_info(),
        "possible_leakage_rows": int(leakage_df["possible_leakage"].sum()) if not leakage_df.empty else 0,
    }
    save_json(config, experiment_dir / "config.json")

    print(f"\nExperimento: {experiment_dir.name}")
    print(f"Clase: {args.class_name} | Arquitectura: {args.architecture}")
    print(f"Device: {device} | Parametros: {config['parameter_count']}")
    print(f"Splits: {config['split_sizes']} | Train efectivo: {config['effective_train_samples']}")
    print(f"Sampling: {args.sampling_strategy} | {sampling_metadata}")
    print(f"Augmentation: {args.augmentation} | {config['augmentation_metadata']}")

    checkpoint_path, _, validation_metrics, validation_rows = fit_model(
        model,
        train_loader,
        valid_loader,
        device,
        config,
        experiment_dir,
    )
    load_best_weights(model, checkpoint_path, device)

    test_metrics = None
    test_rows = None
    if args.run_test:
        test_metrics, test_rows = evaluate_model(model, test_loader, device, "test")
        test_metrics.update({
            "architecture": args.architecture,
            "class_name": args.class_name,
            "parameter_count": config["parameter_count"],
            "checkpoint_path": str(checkpoint_path),
        })
        pd.DataFrame([test_metrics]).to_csv(experiment_dir / "test_metrics.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame(test_rows).to_csv(experiment_dir / "test_per_image_metrics.csv", index=False, encoding="utf-8-sig")

    if args.save_overlays:
        rows = test_rows if test_rows is not None else validation_rows
        loader = test_loader if test_rows is not None else valid_loader
        save_selected_overlays(
            model,
            loader,
            device,
            experiment_dir / "overlays",
            rows,
            args.max_overlay_samples,
        )

    benchmark_metrics = None
    if args.run_benchmark:
        benchmark_metrics = benchmark_model(
            model,
            benchmark_loader,
            device,
            args.architecture,
            args.class_name,
            args.image_size,
        )
        pd.DataFrame([benchmark_metrics]).to_csv(
            experiment_dir / "benchmark_single_model.csv",
            index=False,
            encoding="utf-8-sig",
        )

    write_experiment_summary(experiment_dir, config, validation_metrics, test_metrics, benchmark_metrics)
    update_global_reports()

    print("\nPipeline finalizado.")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Resumen: {experiment_dir / 'summary.md'}")
    if test_metrics:
        print(f"Test Dice: {test_metrics['test_dice']:.4f} | Test IoU: {test_metrics['test_iou']:.4f}")
    if benchmark_metrics:
        print(f"Benchmark: {benchmark_metrics['mean_ms_per_frame']:.3f} ms/frame | {benchmark_metrics['fps']:.2f} FPS")


if __name__ == "__main__":
    main()
