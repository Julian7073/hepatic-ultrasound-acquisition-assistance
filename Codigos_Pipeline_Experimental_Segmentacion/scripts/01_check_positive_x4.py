"""Verifica positive_x4 para LA sin iniciar entrenamiento."""

import sys
from pathlib import Path

from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config_experimental import DATASET_ROOTS
from src.dataset_coco import BinaryCocoDataset, assign_grouped_splits, load_records
from src.transforms import build_transforms


if __name__ == "__main__":
    records = load_records(DATASET_ROOTS["LA"], "LA")
    train_records = assign_grouped_splits(records, "group_video", 42)["train"]
    base, augmented, multiplier = build_transforms(
        "LA", 512, "full_resize", "positive_x4"
    )
    dataset = BinaryCocoDataset(
        records=train_records,
        class_name="LA",
        split="train",
        transform_base=base,
        transform_augmented=augmented,
        augmentation_multiplier=multiplier,
        resize_mode="full_resize",
        roi_index=None,
        augmentation_scope="positive_only",
    )
    metadata = dataset.augmentation_metadata()
    expected_positive = metadata["base_positive_samples"] * 4
    expected_empty = metadata["base_empty_samples"]
    assert metadata["effective_positive_samples"] == expected_positive
    assert metadata["effective_empty_samples"] == expected_empty
    assert metadata["effective_total_samples"] == expected_positive + expected_empty

    loader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0)
    batch = next(iter(loader))
    assert tuple(batch["image"].shape[1:]) == (3, 512, 512)
    assert tuple(batch["mask"].shape[1:]) == (1, 512, 512)

    print("positive_x4 OK")
    print(metadata)
    print(f"Batch image tensor: {tuple(batch['image'].shape)}")
    print(f"Batch mask tensor: {tuple(batch['mask'].shape)}")
