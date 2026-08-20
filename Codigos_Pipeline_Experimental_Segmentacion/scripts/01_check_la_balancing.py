"""Verifica la distribucion producida por balanced_la sin entrenar."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config_experimental import DATASET_ROOTS
from src.dataset_coco import (
    BinaryCocoDataset,
    assign_grouped_splits,
    build_balanced_sampler,
    load_records,
)
from src.transforms import build_transforms


if __name__ == "__main__":
    records = load_records(DATASET_ROOTS["LA"], "LA")
    train_records = assign_grouped_splits(records, "group_video", 42)["train"]
    base, augmented, multiplier = build_transforms("LA", 512, "full_resize", "none")
    dataset = BinaryCocoDataset(
        train_records,
        "LA",
        "train",
        base,
        augmented,
        multiplier,
        "full_resize",
        None,
    )
    sampler, metadata = build_balanced_sampler(dataset, 42)
    sampled_indices = list(iter(sampler))
    sampled_positive = sum(
        bool(dataset.records[index % len(dataset.records)].annotations)
        for index in sampled_indices
    )
    print(metadata)
    print({
        "sampled_total": len(sampled_indices),
        "sampled_positive": sampled_positive,
        "sampled_empty": len(sampled_indices) - sampled_positive,
        "sampled_positive_fraction": sampled_positive / len(sampled_indices),
    })
