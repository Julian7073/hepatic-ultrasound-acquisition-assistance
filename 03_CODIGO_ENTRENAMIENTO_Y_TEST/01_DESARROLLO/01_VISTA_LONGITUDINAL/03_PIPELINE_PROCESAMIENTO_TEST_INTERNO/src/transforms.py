"""Transformaciones sincronizadas para imagen y mascara."""

from __future__ import annotations

import inspect
import os

os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

import albumentations as A
import cv2


def affine_transform(class_name: str):
    """Crea una transformacion anatomica conservadora compatible con Albumentations 1/2."""
    rotate = 3 if class_name == "LA" else 5
    scale = (0.98, 1.02) if class_name == "LA" else (0.96, 1.04)
    params = {
        "scale": scale,
        "translate_percent": (-0.02, 0.02),
        "rotate": (-rotate, rotate),
        "p": 0.65,
    }
    accepted = inspect.signature(A.Affine).parameters
    if "border_mode" in accepted:
        params["border_mode"] = cv2.BORDER_CONSTANT
    elif "mode" in accepted:
        params["mode"] = cv2.BORDER_CONSTANT
    if "fill" in accepted:
        params["fill"] = 0
    elif "cval" in accepted:
        params["cval"] = 0
    if "fill_mask" in accepted:
        params["fill_mask"] = 0
    elif "cval_mask" in accepted:
        params["cval_mask"] = 0
    return A.Affine(**params)


def spatial_resize(image_size: int, resize_mode: str):
    """Implementa resize directo o letterbox conservando proporcion."""
    if resize_mode in {"full_resize", "roi_crop_resize"}:
        return [A.Resize(image_size, image_size)]
    return [
        A.LongestMaxSize(max_size=image_size),
        A.PadIfNeeded(
            min_height=image_size,
            min_width=image_size,
            border_mode=cv2.BORDER_CONSTANT,
            fill=0,
            fill_mask=0,
        ),
    ]


def build_transforms(class_name: str, image_size: int, resize_mode: str, augmentation: str):
    """Devuelve transformacion base y variante aumentada para train."""
    resize = spatial_resize(image_size, resize_mode)
    base = A.Compose(resize)
    if augmentation == "none":
        return base, None, 1

    intensity = 0.04 if class_name == "LA" else 0.08
    noise_range = (0.004, 0.020) if class_name == "LA" else (0.008, 0.030)
    transforms = [
        A.RandomBrightnessContrast(brightness_limit=intensity, contrast_limit=intensity, p=0.55),
        A.RandomGamma(gamma_limit=(95, 105) if class_name == "LA" else (90, 110), p=0.30),
        affine_transform(class_name),
        A.GaussNoise(std_range=noise_range, p=0.20),
        *resize,
    ]
    return base, A.Compose(transforms), 4
