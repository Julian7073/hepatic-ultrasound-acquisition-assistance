"""Augmentations locales conservadoras para segmentacion ecografica."""

from __future__ import annotations

import inspect
import os

# Evita warnings por chequeo online de version en entornos sin conexion estable.
os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

import albumentations as A
import cv2


def affine_conservative(scale, translate_percent, rotate, p):
    """Crea Affine compatible con Albumentations 1.x y 2.x.

    Algunas versiones usan `mode`; otras usan `border_mode`. Esta funcion evita
    warnings por argumentos ignorados y mantiene borde/mask fill en 0.
    """
    params = {
        "scale": scale,
        "translate_percent": translate_percent,
        "rotate": rotate,
        "p": p,
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


def get_transforms(class_name: str, split: str, image_size: int = 512):
    """Devuelve transformaciones conjuntas para imagen y mascara."""
    if split != "train":
        return A.Compose([A.Resize(image_size, image_size)])

    class_key = class_name.lower()
    rotate_limit = 3 if class_key == "la" else 5

    if class_key == "la":
        brightness_limit = 0.05
        contrast_limit = 0.05
        noise_p = 0.15
    elif class_key == "higado":
        brightness_limit = 0.08
        contrast_limit = 0.08
        noise_p = 0.20
    else:
        brightness_limit = 0.08
        contrast_limit = 0.08
        noise_p = 0.10

    return A.Compose(
        [
            A.Resize(image_size, image_size),
            A.RandomBrightnessContrast(
                brightness_limit=brightness_limit,
                contrast_limit=contrast_limit,
                p=0.45,
            ),
            affine_conservative(
                scale=(0.96, 1.04),
                translate_percent=(-0.03, 0.03),
                rotate=(-rotate_limit, rotate_limit),
                p=0.55,
            ),
            A.GaussNoise(p=noise_p),
        ]
    )
