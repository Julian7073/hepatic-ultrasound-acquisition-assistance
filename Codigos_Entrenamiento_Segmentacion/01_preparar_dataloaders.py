"""Prueba rapida de Dataset y DataLoader antes de entrenar."""

from __future__ import annotations

import argparse

from torch.utils.data import DataLoader

from config_segmentation import DATASET_ROOTS, DEFAULT_IMAGE_SIZE, TARGET_CLASSES
from src.augmentations import get_transforms
from src.coco_dataset import BinaryCocoSegmentationDataset


def main() -> None:
    """Carga un batch y muestra dimensiones."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--class_name", choices=TARGET_CLASSES, default="ROI")
    parser.add_argument("--split", choices=["train", "valid", "test"], default="train")
    parser.add_argument("--image_size", type=int, default=DEFAULT_IMAGE_SIZE)
    parser.add_argument("--batch_size", type=int, default=2)
    args = parser.parse_args()

    dataset = BinaryCocoSegmentationDataset(
        dataset_root=DATASET_ROOTS[args.class_name],
        split=args.split,
        class_name=args.class_name,
        transform=get_transforms(args.class_name, args.split, args.image_size),
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    batch = next(iter(loader))

    print(f"Dataset: {args.class_name} | split={args.split}")
    print(f"Imagenes: {len(dataset)}")
    print(f"Batch image tensor: {tuple(batch['image'].shape)}")
    print(f"Batch mask tensor: {tuple(batch['mask'].shape)}")
    print(f"Mask min/max: {batch['mask'].min().item()} / {batch['mask'].max().item()}")
    print("DataLoader OK")


if __name__ == "__main__":
    main()
