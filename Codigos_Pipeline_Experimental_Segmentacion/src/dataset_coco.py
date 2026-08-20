"""Dataset COCO binario con splits originales o agrupados."""

from __future__ import annotations

import json
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, WeightedRandomSampler

from config_experimental import DATASET_ROOTS, SPLITS
from src.split_audit import infer_metadata, normalize_class_name


@dataclass(frozen=True)
class CocoRecord:
    """Referencia inmutable a una imagen y sus anotaciones."""
    image_path: Path
    filename: str
    width: int
    height: int
    annotations: tuple
    original_split: str
    patient: str
    video_id: str
    source_key: str


def polygon_to_mask(segmentation, height: int, width: int) -> np.ndarray:
    """Convierte poligonos COCO o RLE a mascara binaria."""
    mask = np.zeros((height, width), dtype=np.uint8)
    if isinstance(segmentation, list):
        for polygon in segmentation:
            if not isinstance(polygon, list) or len(polygon) < 6:
                continue
            points = np.asarray(polygon, dtype=np.float32).reshape(-1, 2)
            points[:, 0] = np.clip(points[:, 0], 0, width - 1)
            points[:, 1] = np.clip(points[:, 1], 0, height - 1)
            cv2.fillPoly(mask, [points.astype(np.int32)], 1)
        return mask
    if isinstance(segmentation, dict):
        try:
            from pycocotools import mask as mask_utils
            decoded = mask_utils.decode(segmentation)
            if decoded.ndim == 3:
                decoded = np.any(decoded, axis=2)
            return (decoded > 0).astype(np.uint8)
        except Exception as error:
            raise ValueError(f"No se pudo decodificar RLE COCO: {error}") from error
    return mask


def load_records(dataset_root: Path, class_name: str) -> list[CocoRecord]:
    """Carga registros de todos los splits sin copiar imagenes."""
    records = []
    expected = normalize_class_name(class_name)
    for split in SPLITS:
        split_dir = dataset_root / split
        coco_path = split_dir / "_annotations.coco.json"
        if not coco_path.exists():
            raise FileNotFoundError(f"No existe COCO: {coco_path}")
        with coco_path.open("r", encoding="utf-8") as file:
            coco = json.load(file)
        target_ids = {item["id"] for item in coco.get("categories", []) if normalize_class_name(item.get("name", "")) == expected}
        if len(target_ids) != 1:
            raise ValueError(f"Se esperaba una clase {class_name} en {coco_path}")
        annotations_by_image = defaultdict(list)
        for annotation in coco.get("annotations", []):
            if annotation.get("category_id") in target_ids:
                annotations_by_image[annotation.get("image_id")].append(annotation)
        for image in coco.get("images", []):
            metadata = infer_metadata(image["file_name"])
            records.append(CocoRecord(
                image_path=split_dir / image["file_name"],
                filename=image["file_name"],
                width=int(image.get("width", 0)),
                height=int(image.get("height", 0)),
                annotations=tuple(annotations_by_image.get(image.get("id"), [])),
                original_split=split,
                patient=metadata["patient"],
                video_id=metadata["video_id"],
                source_key=metadata["source_key"],
            ))
    return records


def assign_grouped_splits(records: list[CocoRecord], strategy: str, seed: int) -> dict[str, list[CocoRecord]]:
    """Reasigna grupos completos a train/valid/test de forma determinista."""
    if strategy == "coco":
        return {split: [record for record in records if record.original_split == split] for split in SPLITS}
    key_name = "patient" if strategy == "group_patient" else "video_id"
    groups = sorted({getattr(record, key_name) for record in records})
    if "unknown" in groups or "" in groups:
        raise ValueError(f"No se puede usar {strategy}: existen grupos no identificados.")
    random.Random(seed).shuffle(groups)
    if len(groups) < 3:
        raise ValueError(f"No se puede usar {strategy}: se requieren al menos 3 grupos y hay {len(groups)}.")
    if len(groups) == 3:
        train_end, valid_end = 1, 2
    else:
        train_end = max(1, round(len(groups) * 0.70))
        valid_end = min(max(train_end + 1, round(len(groups) * 0.85)), len(groups) - 1)
    assignment = {group: "train" if i < train_end else "valid" if i < valid_end else "test" for i, group in enumerate(groups)}
    return {split: [record for record in records if assignment[getattr(record, key_name)] == split] for split in SPLITS}


def build_roi_mask_index() -> dict[str, np.ndarray]:
    """Indexa mascaras ROI por nombre fuente para recortar otras clases."""
    index = {}
    for record in load_records(DATASET_ROOTS["ROI"], "ROI"):
        mask = np.zeros((record.height, record.width), dtype=np.uint8)
        for annotation in record.annotations:
            mask = np.maximum(mask, polygon_to_mask(annotation.get("segmentation", []), record.height, record.width))
        index[record.source_key] = mask
    return index


def crop_to_mask(image: np.ndarray, mask: np.ndarray, roi_mask: np.ndarray, margin_ratio: float = 0.03):
    """Recorta imagen y mascara a la caja de ROI con margen pequeno."""
    points = cv2.findNonZero((roi_mask > 0).astype(np.uint8))
    if points is None:
        return image, mask, False
    x, y, width, height = cv2.boundingRect(points)
    margin = int(round(max(width, height) * margin_ratio))
    x0, y0 = max(0, x - margin), max(0, y - margin)
    x1, y1 = min(image.shape[1], x + width + margin), min(image.shape[0], y + height + margin)
    return image[y0:y1, x0:x1], mask[y0:y1, x0:x1], True



def build_balanced_sampler(dataset: "BinaryCocoDataset", seed: int) -> tuple[WeightedRandomSampler, dict]:
    """Equilibra imagenes positivas y vacias sin duplicar archivos fisicos."""
    labels = [dataset.is_positive(index) for index in range(len(dataset))]
    positive_count = sum(labels)
    empty_count = len(labels) - positive_count
    if positive_count == 0 or empty_count == 0:
        raise ValueError(
            f"No se puede balancear: positives={positive_count}, empty={empty_count}."
        )
    weights = [
        0.5 / positive_count if label else 0.5 / empty_count
        for label in labels
    ]
    generator = torch.Generator()
    generator.manual_seed(seed)
    sampler = WeightedRandomSampler(
        weights=torch.as_tensor(weights, dtype=torch.double),
        num_samples=len(labels),
        replacement=True,
        generator=generator,
    )
    metadata = {
        "original_positive_samples": positive_count,
        "original_empty_samples": empty_count,
        "target_positive_fraction": 0.5,
        "samples_per_epoch": len(labels),
    }
    return sampler, metadata

class BinaryCocoDataset(Dataset):
    """Dataset binario con transformaciones sincronizadas y augmentation x4."""

    def __init__(self, records, class_name, split, transform_base, transform_augmented=None, augmentation_multiplier=1, resize_mode="full_resize", roi_index=None, augmentation_scope="all"):
        self.records = records
        self.class_name = class_name
        self.split = split
        self.transform_base = transform_base
        self.transform_augmented = transform_augmented
        self.multiplier = augmentation_multiplier if split == "train" else 1
        self.augmentation_scope = augmentation_scope
        self.resize_mode = resize_mode
        self.roi_index = roi_index or {}
        self.virtual_indices = []
        for record_index, record in enumerate(self.records):
            repeat_count = self.multiplier
            if self.augmentation_scope == "positive_only" and not record.annotations:
                repeat_count = 1
            self.virtual_indices.extend(
                (record_index, repeat_index)
                for repeat_index in range(repeat_count)
            )

    def __len__(self) -> int:
        return len(self.virtual_indices)

    def is_positive(self, index: int) -> bool:
        record_index, _ = self.virtual_indices[index]
        return bool(self.records[record_index].annotations)

    def augmentation_metadata(self) -> dict:
        """Resume registros base y muestras virtuales efectivas."""
        base_positive = sum(bool(record.annotations) for record in self.records)
        effective_positive = sum(self.is_positive(index) for index in range(len(self)))
        return {
            "scope": self.augmentation_scope,
            "base_positive_samples": base_positive,
            "base_empty_samples": len(self.records) - base_positive,
            "effective_positive_samples": effective_positive,
            "effective_empty_samples": len(self) - effective_positive,
            "effective_total_samples": len(self),
            "effective_positive_fraction": effective_positive / max(len(self), 1),
        }

    def __getitem__(self, index: int) -> dict:
        record_index, repeat_index = self.virtual_indices[index]
        record = self.records[record_index]
        image = cv2.imread(str(record.image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"No se pudo leer: {record.image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width = image.shape[:2]
        mask = np.zeros((height, width), dtype=np.uint8)
        for annotation in record.annotations:
            mask = np.maximum(mask, polygon_to_mask(annotation.get("segmentation", []), height, width))
        roi_crop_found = True
        if self.resize_mode == "roi_crop_resize":
            roi_mask = self.roi_index.get(record.source_key)
            if roi_mask is None and self.class_name == "ROI" and mask.any():
                roi_mask = mask
            if roi_mask is not None:
                if roi_mask.shape != mask.shape:
                    roi_mask = cv2.resize(roi_mask, (width, height), interpolation=cv2.INTER_NEAREST)
                image, mask, roi_crop_found = crop_to_mask(image, mask, roi_mask)
            else:
                roi_crop_found = False
        transform = self.transform_augmented if repeat_index > 0 and self.transform_augmented else self.transform_base
        transformed = transform(image=image, mask=mask)
        image = transformed["image"].astype(np.float32) / 255.0
        mask = (transformed["mask"] > 0).astype(np.float32)
        return {
            "image": torch.from_numpy(image.transpose(2, 0, 1)).float(),
            "mask": torch.from_numpy(mask).unsqueeze(0).float(),
            "filename": record.filename,
            "image_path": str(record.image_path),
            "source_key": record.source_key,
            "patient": record.patient,
            "video_id": record.video_id,
            "original_split": record.original_split,
            "roi_crop_found": roi_crop_found,
        }
