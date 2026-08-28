"""Motor compartido para inferencia longitudinal en imagenes y video."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
import torch

from src.longitudinal_quality_rules import (
    compute_lumen_quality_metrics,
    final_longitudinal_decision,
    load_decision_config,
)
from src.models import create_model


@dataclass
class LoadedBinaryModel:
    class_name: str
    model: torch.nn.Module
    architecture: str
    image_size: int
    resize_mode: str
    checkpoint_path: Path


@dataclass
class FrameInference:
    decision: str
    decision_reason: str
    message: str
    has_roi: int
    has_higado: int
    has_la: int
    raw_area_roi_px: int
    raw_area_higado_px: int
    raw_area_la_px: int
    area_roi_px: int
    area_higado_px: int
    area_la_px: int
    roi_frame_ratio: float
    higado_roi_ratio: float
    la_roi_ratio: float
    min_la_area_px_scaled: int
    la_presence_threshold_scaled: int
    la_area_ok: int
    la_std_ok: int
    la_entropy_ok: int
    base_lumen_rule: int
    final_lumen_rule: int
    valid_lumen_area_px: int
    la_mean_intensity: float | None
    la_std_intensity: float | None
    glcm_contrast: float | None
    glcm_entropy: float | None
    glcm_homogeneity: float | None
    glcm_energy: float | None
    glcm_valid_pairs: int
    border_status: str
    border_ring_area_px: int
    border_bright_delta: float | None
    border_gradient_p75: float | None
    border_ring_bright_ratio: float | None
    border_p90_minus_la_p10: float | None
    bright_border_ok: int
    gradient_border_ok: int
    local_contrast_ok: int
    border_evidence: int
    border_all_evidence: int
    rule_mode: str
    glcm_region_mode: str
    use_border_rule: int
    mostly_black: int
    inference_ms_roi: float
    inference_ms_higado: float
    inference_ms_la: float
    inference_ms_models_total: float
    masks: dict[str, np.ndarray] | None = None

    def to_row(self) -> dict:
        row = asdict(self)
        row.pop("masks", None)
        return row


def load_binary_checkpoint(
    checkpoint_path: Path,
    class_name: str,
    device: torch.device,
) -> LoadedBinaryModel:
    """Carga arquitectura y pesos sin volver a descargar pretraining."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint.get("config", {})
    architecture = checkpoint.get("architecture") or config.get("architecture")
    if not architecture:
        raise ValueError(f"Checkpoint sin arquitectura: {checkpoint_path}")
    model, _ = create_model(str(architecture), pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()
    return LoadedBinaryModel(
        class_name=class_name,
        model=model,
        architecture=str(architecture),
        image_size=int(checkpoint.get("image_size", config.get("image_size", 512))),
        resize_mode=str(
            checkpoint.get("resize_mode", config.get("resize_mode", "full_resize"))
        ),
        checkpoint_path=checkpoint_path,
    )


def load_selected_models(
    checkpoint_paths: dict[str, Path],
    device: torch.device,
) -> dict[str, LoadedBinaryModel]:
    loaded = {}
    for class_name in ("ROI", "Higado", "LA"):
        path = checkpoint_paths[class_name]
        if not path.exists():
            raise FileNotFoundError(f"Falta checkpoint {class_name}: {path}")
        loaded[class_name] = load_binary_checkpoint(path, class_name, device)
    return loaded


def preprocess_frame(frame_bgr: np.ndarray, model_info: LoadedBinaryModel) -> torch.Tensor:
    """Replica full_resize y normalizacion [0,1] usadas en entrenamiento."""
    if model_info.resize_mode != "full_resize":
        raise NotImplementedError(
            f"Inferencia externa soporta full_resize; recibido {model_info.resize_mode}."
        )
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(
        rgb,
        (model_info.image_size, model_info.image_size),
        interpolation=cv2.INTER_AREA,
    )
    return torch.from_numpy(
        resized.transpose(2, 0, 1).copy()
    ).float().unsqueeze(0) / 255.0


@torch.inference_mode()
def predict_mask(
    model_info: LoadedBinaryModel,
    frame_bgr: np.ndarray,
    device: torch.device,
    threshold: float,
) -> tuple[np.ndarray, float]:
    tensor = preprocess_frame(frame_bgr, model_info).to(device)
    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    logits = model_info.model(tensor)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    probability = torch.sigmoid(logits)[0, 0].detach().cpu().numpy()
    height, width = frame_bgr.shape[:2]
    probability = cv2.resize(
        probability, (width, height), interpolation=cv2.INTER_LINEAR
    )
    return (probability >= threshold).astype(np.uint8), elapsed_ms


def largest_component(mask: np.ndarray, minimum_area: int = 1) -> np.ndarray:
    binary = (mask > 0).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if count <= 1:
        return np.zeros_like(binary)
    areas = stats[1:, cv2.CC_STAT_AREA]
    selected = int(np.argmax(areas)) + 1
    if int(stats[selected, cv2.CC_STAT_AREA]) < minimum_area:
        return np.zeros_like(binary)
    return (labels == selected).astype(np.uint8)


def mask_area(mask: np.ndarray) -> int:
    return int(np.count_nonzero(mask))


def postprocess_masks(
    raw_masks: dict[str, np.ndarray],
    frame_shape: tuple[int, ...],
    decision_config: dict,
) -> dict[str, np.ndarray]:
    """Elimina componentes aislados y restringe Higado y LA a la ROI."""
    height, width = frame_shape[:2]
    scale = (height * width) / float(decision_config["reference_area"])
    roi = largest_component(raw_masks["ROI"], max(16, round(500 * scale)))
    higado = largest_component(
        raw_masks["Higado"] & roi,
        max(16, round(200 * scale)),
    )
    la = largest_component(
        raw_masks["LA"] & roi,
        max(4, round(8 * scale)),
    )
    return {"ROI": roi, "Higado": higado, "LA": la}


def guidance_decision(
    frame_bgr: np.ndarray,
    raw_masks: dict[str, np.ndarray],
    masks: dict[str, np.ndarray],
    timings: dict[str, float],
    decision_config: dict,
) -> FrameInference:
    """Aplica la unica regla longitudinal usada por scripts y GUI."""
    height, width = frame_bgr.shape[:2]
    frame_area = height * width
    area_scale = frame_area / float(decision_config["reference_area"])
    raw_areas = {name: mask_area(mask) for name, mask in raw_masks.items()}
    areas = {name: mask_area(mask) for name, mask in masks.items()}
    roi_frame_ratio = areas["ROI"] / max(frame_area, 1)
    higado_roi_ratio = areas["Higado"] / max(areas["ROI"], 1)

    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    mostly_black = int(
        float(np.mean(gray < int(decision_config["mostly_black_intensity"])))
        > float(decision_config["mostly_black_fraction"])
    )
    has_roi = int(
        roi_frame_ratio >= float(decision_config["roi_min_frame_ratio"])
        and not mostly_black
    )
    minimum_liver_area = max(
        1, round(float(decision_config["liver_min_area_px_ref"]) * area_scale)
    )
    has_higado = int(areas["Higado"] >= minimum_liver_area)

    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    quality = compute_lumen_quality_metrics(
        frame_rgb,
        masks["ROI"],
        masks["LA"],
        decision_config,
    )
    quality.update({
        "roi_present": has_roi,
        "liver_present": has_higado,
        "liver_roi_ratio": higado_roi_ratio,
    })
    final = final_longitudinal_decision(quality, decision_config)

    return FrameInference(
        decision=final["decision"],
        decision_reason=final["decision_reason"],
        message=final["message"],
        has_roi=has_roi,
        has_higado=has_higado,
        has_la=int(final["la_present"]),
        raw_area_roi_px=raw_areas["ROI"],
        raw_area_higado_px=raw_areas["Higado"],
        raw_area_la_px=raw_areas["LA"],
        area_roi_px=areas["ROI"],
        area_higado_px=areas["Higado"],
        area_la_px=int(quality["la_area_px"]),
        roi_frame_ratio=roi_frame_ratio,
        higado_roi_ratio=higado_roi_ratio,
        la_roi_ratio=float(quality["la_roi_ratio"]),
        min_la_area_px_scaled=int(quality["la_area_threshold_scaled"]),
        la_presence_threshold_scaled=int(
            quality["la_presence_threshold_scaled"]
        ),
        la_area_ok=int(final["la_area_ok"]),
        la_std_ok=int(final["la_std_ok"]),
        la_entropy_ok=int(final["la_entropy_ok"]),
        base_lumen_rule=int(final["base_lumen_rule"]),
        final_lumen_rule=int(final["final_lumen_rule"]),
        valid_lumen_area_px=int(quality["valid_lumen_area_px"]),
        la_mean_intensity=quality["la_mean"],
        la_std_intensity=quality["la_std"],
        glcm_contrast=quality["glcm_contrast"],
        glcm_entropy=quality["glcm_entropy"],
        glcm_homogeneity=quality["glcm_homogeneity"],
        glcm_energy=quality["glcm_energy"],
        glcm_valid_pairs=int(quality["glcm_valid_pairs"]),
        border_status=str(quality["border_status"]),
        border_ring_area_px=int(quality["border_ring_area_px"]),
        border_bright_delta=quality["border_bright_delta"],
        border_gradient_p75=quality["border_gradient_p75"],
        border_ring_bright_ratio=quality["border_ring_bright_ratio"],
        border_p90_minus_la_p10=quality["border_p90_minus_la_p10"],
        bright_border_ok=int(quality["bright_border_ok"]),
        gradient_border_ok=int(quality["gradient_border_ok"]),
        local_contrast_ok=int(quality["local_contrast_ok"]),
        border_evidence=int(quality["border_evidence"]),
        border_all_evidence=int(quality["border_all_evidence"]),
        rule_mode=str(final["rule_mode"]),
        glcm_region_mode=str(quality["glcm_region_mode"]),
        use_border_rule=int(final["use_border_rule"]),
        mostly_black=mostly_black,
        inference_ms_roi=timings["ROI"],
        inference_ms_higado=timings["Higado"],
        inference_ms_la=timings["LA"],
        inference_ms_models_total=sum(timings.values()),
        masks=masks,
    )


def infer_frame(
    frame_bgr: np.ndarray,
    models: dict[str, LoadedBinaryModel],
    device: torch.device,
    threshold: float = 0.5,
    decision_config: dict | None = None,
) -> FrameInference:
    config = decision_config or load_decision_config()
    raw_masks = {}
    timings = {}
    for class_name in ("ROI", "Higado", "LA"):
        raw_masks[class_name], timings[class_name] = predict_mask(
            models[class_name], frame_bgr, device, threshold
        )
    masks = postprocess_masks(raw_masks, frame_bgr.shape, config)
    return guidance_decision(frame_bgr, raw_masks, masks, timings, config)


def create_overlay(frame_bgr: np.ndarray, result: FrameInference) -> np.ndarray:
    overlay = frame_bgr.copy()
    color_layer = np.zeros_like(frame_bgr)
    masks = result.masks or {}
    colors = {"ROI": (0, 180, 0), "Higado": (255, 70, 40), "LA": (0, 0, 255)}
    for class_name in ("ROI", "Higado", "LA"):
        mask = masks.get(class_name)
        if mask is not None:
            color_layer[mask > 0] = colors[class_name]
    overlay = cv2.addWeighted(overlay, 0.74, color_layer, 0.26, 0)
    for class_name in ("ROI", "Higado", "LA"):
        mask = masks.get(class_name)
        if mask is None:
            continue
        contours, _ = cv2.findContours(
            (mask > 0).astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        cv2.drawContours(overlay, contours, -1, colors[class_name], 2)
    cv2.putText(
        overlay,
        result.decision,
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return overlay
