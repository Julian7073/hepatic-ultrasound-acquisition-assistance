"""Dataset PyTorch para COCO segmentation binario."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


def normalize_name(value: str) -> str:
    """Normaliza nombres de clases para comparacion robusta."""
    return str(value).strip().lower().replace("í", "i").replace("ı", "i")


def load_coco(path: Path) -> dict:
    """Carga un archivo COCO JSON."""
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def polygon_to_mask(segmentation, height: int, width: int) -> np.ndarray:
    """Convierte poligonos COCO a mascara binaria 0/1."""
    mask = np.zeros((height, width), dtype=np.uint8)
    if not isinstance(segmentation, list):
        return mask

    for polygon in segmentation:
        if not isinstance(polygon, list) or len(polygon) < 6:
            continue
        points = np.array(polygon, dtype=np.float32).reshape(-1, 2)
        points[:, 0] = np.clip(points[:, 0], 0, width - 1)
        points[:, 1] = np.clip(points[:, 1], 0, height - 1)
        cv2.fillPoly(mask, [points.astype(np.int32)], 1)
    return mask


class BinaryCocoSegmentationDataset(Dataset):
    """Dataset para entrenar una clase COCO como segmentacion binaria."""

    def __init__(
        self,
        dataset_root: Path,
        split: str,
        class_name: str,
        transform=None,
    ) -> None:
        self.dataset_root = Path(dataset_root)
        self.split = split
        self.class_name = class_name
        self.transform = transform
        self.split_dir = self.dataset_root / split
        self.coco_path = self.split_dir / "_annotations.coco.json"

        if not self.coco_path.exists():
            raise FileNotFoundError(f"No existe COCO: {self.coco_path}")

        self.coco = load_coco(self.coco_path)
        self.images = self.coco.get("images", [])
        categories = self.coco.get("categories", [])

        expected = normalize_name(class_name)
        target_ids = [cat["id"] for cat in categories if normalize_name(cat.get("name", "")) == expected]
        if len(target_ids) != 1:
            raise ValueError(f"No se encontro una clase unica {class_name} en {self.coco_path}: {categories}")
        self.target_category_id = target_ids[0]

        self.annotations_by_image = defaultdict(list)
        for annotation in self.coco.get("annotations", []):
            if annotation.get("category_id") == self.target_category_id:
                self.annotations_by_image[annotation.get("image_id")].append(annotation)

    def __len__(self) -> int:
        """Numero de imagenes."""
        return len(self.images)

    def __getitem__(self, index: int) -> dict:
        """Devuelve imagen, mascara y metadatos."""
        image_info = self.images[index]
        filename = image_info["file_name"]
        image_path = self.split_dir / filename

        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"No se pudo leer imagen: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width = image.shape[:2]

        mask = np.zeros((height, width), dtype=np.uint8)
        for annotation in self.annotations_by_image.get(image_info["id"], []):
            mask = np.maximum(mask, polygon_to_mask(annotation.get("segmentation", []), height, width))

        if self.transform is not None:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]

        image = image.astype(np.float32) / 255.0
        if image.ndim == 2:
            image = np.repeat(image[..., None], 3, axis=2)
        image_tensor = torch.from_numpy(image.transpose(2, 0, 1)).float()
        mask_tensor = torch.from_numpy((mask > 0).astype(np.float32)).unsqueeze(0)

        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "filename": filename,
            "image_path": str(image_path),
            "split": self.split,
            "class_name": self.class_name,
        }
