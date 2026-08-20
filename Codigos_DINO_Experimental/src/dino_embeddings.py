"""Extraccion congelada de embeddings DINOv2."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

try:
    from ..config_dino import DINOV2_MODEL_ID
except ImportError:
    from config_dino import DINOV2_MODEL_ID


def select_by_video_stride(index: pd.DataFrame, stride: int) -> pd.DataFrame:
    """Muestrea cada video independientemente conservando el primer frame."""
    if stride < 1:
        raise ValueError("stride debe ser >= 1.")
    selected = []
    for _, group in index.groupby("video_id", sort=True):
        ordered = group.sort_values(["frame_number", "filename"])
        selected.append(ordered.iloc[::stride])
    return pd.concat(selected, ignore_index=True)


def balanced_smoke_sample(index: pd.DataFrame, maximum: int, seed: int) -> pd.DataFrame:
    """Selecciona una muestra pequena distribuida por vista y calidad."""
    if maximum <= 0 or len(index) <= maximum:
        return index.reset_index(drop=True)
    selected = []
    groups = list(index.groupby(["view", "quality"], sort=True))
    per_group = max(1, maximum // max(len(groups), 1))
    selected_indices = set()
    for group_index, (_, group) in enumerate(groups):
        sample = group.sample(
            min(per_group, len(group)), random_state=seed + group_index
        )
        selected.append(sample)
        selected_indices.update(sample.index)
    frame = pd.concat(selected, ignore_index=False)
    if len(frame) < maximum:
        remaining = index.loc[~index.index.isin(selected_indices)]
        if not remaining.empty:
            frame = pd.concat([
                frame,
                remaining.sample(
                    min(maximum - len(frame), len(remaining)), random_state=seed
                ),
            ])
    return frame.head(maximum).reset_index(drop=True)


class DinoV2Extractor:
    """Carga DINOv2 una sola vez y devuelve el token CLS."""

    def __init__(self, device: torch.device, model_id: str = DINOV2_MODEL_ID) -> None:
        from transformers import AutoImageProcessor, AutoModel

        self.device = device
        self.model_id = model_id
        try:
            self.processor = AutoImageProcessor.from_pretrained(
                model_id, local_files_only=True
            )
            self.model = AutoModel.from_pretrained(
                model_id, local_files_only=True
            ).to(device).eval()
            self.loaded_from_local_cache = True
        except OSError:
            self.processor = AutoImageProcessor.from_pretrained(model_id)
            self.model = AutoModel.from_pretrained(model_id).to(device).eval()
            self.loaded_from_local_cache = False
        self.embedding_dim = int(self.model.config.hidden_size)

    @torch.inference_mode()
    def extract_images(self, images: list[Image.Image]) -> np.ndarray:
        """Extrae embeddings de imagenes ya cargadas para inferencia en video."""
        if not images:
            return np.empty((0, self.embedding_dim), dtype=np.float32)
        inputs = self.processor(images=images, return_tensors="pt")
        outputs = self.model(
            pixel_values=inputs["pixel_values"].to(self.device)
        )
        return (
            outputs.last_hidden_state[:, 0, :]
            .cpu().numpy().astype(np.float32)
        )

    @torch.inference_mode()
    def extract(
        self,
        paths: list[Path],
        batch_size: int = 16,
        image_transform: Callable[[Image.Image], Image.Image] | None = None,
    ) -> tuple[np.ndarray, dict]:
        embeddings = []
        batch_times = []
        for start in tqdm(
            range(0, len(paths), batch_size), desc="DINOv2 embeddings"
        ):
            batch_paths = paths[start:start + batch_size]
            images = []
            for path in batch_paths:
                with Image.open(path) as source:
                    image = source.convert("RGB")
                    if image_transform is not None:
                        image = image_transform(image)
                    images.append(image)
            inputs = self.processor(images=images, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(self.device)
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            started = time.perf_counter()
            outputs = self.model(pixel_values=pixel_values)
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            batch_times.append(time.perf_counter() - started)
            embeddings.append(
                outputs.last_hidden_state[:, 0, :]
                .cpu().numpy().astype(np.float32)
            )
        matrix = (
            np.concatenate(embeddings, axis=0)
            if embeddings
            else np.empty((0, self.embedding_dim), dtype=np.float32)
        )
        total_s = float(sum(batch_times))
        metadata = {
            "model_id": self.model_id,
            "embedding_dim": self.embedding_dim,
            "images": len(paths),
            "batch_size": batch_size,
            "device": str(self.device),
            "model_inference_seconds": total_s,
            "model_images_per_second": len(paths) / total_s if total_s > 0 else 0.0,
        }
        return matrix, metadata


def save_embeddings(
    matrix: np.ndarray,
    metadata_frame: pd.DataFrame,
    output_prefix: Path,
    extraction_metadata: dict,
) -> tuple[Path, Path, Path]:
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    embeddings_path = output_prefix.with_suffix(".npz")
    metadata_path = output_prefix.parent / f"{output_prefix.name}_metadata.csv"
    info_path = output_prefix.parent / f"{output_prefix.name}_info.json"
    np.savez_compressed(embeddings_path, embeddings=matrix)
    metadata_frame.to_csv(metadata_path, index=False, encoding="utf-8-sig")
    info_path.write_text(
        json.dumps(extraction_metadata, indent=2), encoding="utf-8"
    )
    return embeddings_path, metadata_path, info_path
