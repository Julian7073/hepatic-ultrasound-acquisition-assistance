"""Inferencia binaria DINOv2 para videos nuevos."""

from __future__ import annotations

from collections import deque
from pathlib import Path

import cv2
import joblib
import numpy as np
import pandas as pd
import torch
from PIL import Image

try:
    from ..config_dino import BINARY_MODELS_ROOT
    from .binary_temporal import clear_probabilities
    from .dino_embeddings import DinoV2Extractor
    from .ultrasound_preprocessing import fan_crop_pil
except ImportError:
    from config_dino import BINARY_MODELS_ROOT
    from src.binary_temporal import clear_probabilities
    from src.dino_embeddings import DinoV2Extractor
    from src.ultrasound_preprocessing import fan_crop_pil


MESSAGES = {
    "capture": (
        "Imagen potencialmente informativa. Mantenga la posicion y capture."
    ),
    "adjust": (
        "La imagen aun no es suficientemente informativa. "
        "Ajuste suavemente la posicion o inclinacion de la sonda."
    ),
    "doubtful": (
        "Calidad incierta. Continue ajustando la sonda hasta obtener "
        "una visualizacion mas estable."
    ),
    "warming_up": (
        "Analizando una secuencia corta. Mantenga el movimiento suave."
    ),
}


class BinaryDinoVideoPredictor:
    """Carga el modelo seleccionado de una vista y mantiene su contexto temporal."""

    def __init__(self, view: str, device: str | None = None) -> None:
        model_path = BINARY_MODELS_ROOT / f"{view}__binary_dinov2.joblib"
        if not model_path.exists():
            raise FileNotFoundError(f"Falta el modelo {model_path}")
        self.bundle = joblib.load(model_path)
        self.view = view
        requested = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(requested)
        self.extractor = DinoV2Extractor(
            device=self.device,
            model_id=self.bundle["embedding_model_id"],
        )
        self.window_size = int(self.bundle.get("window_size", 1))
        self.buffer: deque[np.ndarray] = deque(maxlen=self.window_size)

    def reset(self) -> None:
        self.buffer.clear()

    def _prepare(self, rgb: np.ndarray) -> Image.Image:
        image = Image.fromarray(rgb)
        if self.bundle.get("preprocessing") == "fan_crop":
            return fan_crop_pil(image)
        return image

    def predict_rgb(self, rgb: np.ndarray) -> dict:
        image = self._prepare(rgb)
        embedding = self.extractor.extract_images([image])[0]
        self.buffer.append(embedding)
        if self.bundle["temporal_mode"] == "window5":
            if len(self.buffer) < self.window_size:
                return {
                    "probability_clear": np.nan,
                    "action": "warming_up",
                    "message": MESSAGES["warming_up"],
                    "buffer_frames": len(self.buffer),
                }
            values = np.stack(self.buffer)
            feature = np.concatenate([values.mean(axis=0), values.std(axis=0)])
        else:
            feature = embedding

        _, p_clear = clear_probabilities(
            self.bundle["model"], feature.reshape(1, -1)
        )
        probability = float(p_clear[0])
        if probability >= float(self.bundle["capture_threshold"]):
            action = "capture"
        elif probability <= float(self.bundle["adjust_threshold"]):
            action = "adjust"
        else:
            action = "doubtful"
        return {
            "probability_clear": probability,
            "action": action,
            "message": MESSAGES[action],
            "buffer_frames": len(self.buffer),
        }


def process_video(
    video_path: Path,
    view: str,
    output_csv: Path,
    inference_stride: int | None = None,
    device: str | None = None,
) -> pd.DataFrame:
    video_path = Path(video_path).expanduser()
    if not video_path.exists():
        raise FileNotFoundError(
            f"No existe el video: {video_path}. "
            "Reemplace la ruta de ejemplo por la ruta real del archivo."
        )
    if not video_path.is_file():
        raise FileNotFoundError(f"La ruta no es un archivo de video: {video_path}")
    predictor = BinaryDinoVideoPredictor(view=view, device=device)
    stride = int(inference_stride or predictor.bundle.get("stride", 5))
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"No se pudo abrir el video: {video_path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    rows = []
    last_result = {
        "probability_clear": np.nan,
        "action": "warming_up",
        "message": MESSAGES["warming_up"],
        "buffer_frames": 0,
    }
    frame_id = 0
    while True:
        readable, bgr = capture.read()
        if not readable:
            break
        evaluated = int(frame_id % stride == 0)
        if evaluated:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            last_result = predictor.predict_rgb(rgb)
        rows.append({
            "frame_id": frame_id,
            "timestamp_s": frame_id / fps if fps > 0 else np.nan,
            "view": view,
            "evaluated": evaluated,
            "probability_informative": last_result["probability_clear"],
            "decision": last_result["action"],
            "message": last_result["message"],
            "temporal_buffer_frames": last_result["buffer_frames"],
            "inference_stride": stride,
            "embedding_model_id": predictor.bundle["embedding_model_id"],
            "preprocessing": predictor.bundle["preprocessing"],
            "temporal_mode": predictor.bundle["temporal_mode"],
        })
        frame_id += 1
    capture.release()
    result = pd.DataFrame(rows)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_csv, index=False, encoding="utf-8-sig")
    return result