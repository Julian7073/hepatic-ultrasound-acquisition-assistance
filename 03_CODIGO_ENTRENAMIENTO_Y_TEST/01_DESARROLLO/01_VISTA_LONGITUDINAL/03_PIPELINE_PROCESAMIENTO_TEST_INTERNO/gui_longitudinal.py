"""GUI Streamlit para adquisición longitudinal asistida."""

from __future__ import annotations

import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import cv2
import pandas as pd
import streamlit as st
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from config_experimental import (
    FINAL_MODELS_ROOT,
    GUI_OUTPUT_ROOT,
    ensure_directories,
)
from src.longitudinal_inference import (
    create_overlay,
    infer_frame,
    load_selected_models,
)
from src.temporal_guidance import TemporalGuidance


st.set_page_config(
    page_title="Adquisición ecográfica hepática",
    layout="wide",
)


def checkpoint_paths() -> dict[str, Path]:
    return {
        "ROI": FINAL_MODELS_ROOT / "best_roi_model.pth",
        "Higado": FINAL_MODELS_ROOT / "best_higado_model.pth",
        "LA": FINAL_MODELS_ROOT / "best_la_model.pth",
    }


@st.cache_resource(show_spinner="Cargando modelos de segmentación...")
def load_models_cached(use_cpu: bool):
    device = torch.device(
        "cpu" if use_cpu or not torch.cuda.is_available() else "cuda"
    )
    return load_selected_models(checkpoint_paths(), device), device


def resolve_input_video(uploaded_file, local_path: str, session_dir: Path) -> Path:
    if uploaded_file is not None:
        suffix = Path(uploaded_file.name).suffix.lower() or ".mp4"
        destination = session_dir / f"input_video{suffix}"
        destination.write_bytes(uploaded_file.getbuffer())
        return destination
    path = Path(local_path.strip().strip('"'))
    if not path.exists():
        raise FileNotFoundError(f"No existe el video: {path}")
    return path


def show_guidance(decision: str, message: str) -> None:
    if decision == "capture":
        st.success(message)
    elif decision == "no_structure":
        st.error(message)
    elif decision == "partial_liver":
        st.warning(message)
    else:
        st.info(message)


def save_summary(
    session_dir: Path,
    video_path: Path,
    results: pd.DataFrame,
    source_fps: float,
    elapsed_s: float,
) -> None:
    counts = Counter(results["stable_decision"])
    summary = pd.DataFrame([{
        "video_path": str(video_path),
        "processed_frames": len(results),
        "source_fps": source_fps,
        "elapsed_s": elapsed_s,
        "mean_processing_ms": results["processing_ms"].mean(),
        "estimated_processing_fps": 1000.0 / results["processing_ms"].mean(),
        "saved_capture_frames": int(results["capture_saved"].sum()),
        **{f"decision_{key}": value for key, value in counts.items()},
    }])
    summary.to_csv(
        session_dir / "session_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    lines = [
        "# Resumen de sesión longitudinal",
        "",
        f"- Video: {video_path}",
        f"- Frames procesados: {len(results)}",
        f"- FPS fuente: {source_fps:.3f}",
        f"- Capturas guardadas: {int(results['capture_saved'].sum())}",
        "",
        "Los mensajes son asistencia experimental y no constituyen diagnóstico médico.",
    ]
    (session_dir / "session_summary.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


ensure_directories()
st.title("Adquisición ecográfica hepática")
st.caption("Procesamiento longitudinal frame por frame con ROI, hígado y lumen anecoico.")

with st.sidebar:
    st.header("Configuración")
    view = st.selectbox("Vista ecográfica", ["Longitudinal"])
    frame_stride = st.number_input(
        "Procesar 1 de cada N frames",
        min_value=1,
        max_value=30,
        value=3,
        step=1,
        help="Con el video actual, N=3 permite aproximarse al ritmo de adquisición.",
    )
    threshold = st.slider(
        "Umbral de segmentación",
        min_value=0.10,
        max_value=0.90,
        value=0.50,
        step=0.05,
    )
    temporal_window = st.number_input(
        "Ventana temporal",
        min_value=1,
        max_value=15,
        value=5,
        step=2,
    )
    capture_streak = st.number_input(
        "Frames aceptables consecutivos",
        min_value=1,
        max_value=10,
        value=3,
        step=1,
    )
    capture_cooldown = st.number_input(
        "Intervalo entre capturas guardadas",
        min_value=1,
        max_value=100,
        value=10,
        step=1,
    )
    max_processed = st.number_input(
        "Máximo de frames procesados (0 = todo)",
        min_value=0,
        max_value=100000,
        value=0,
        step=10,
    )
    display_every = st.number_input(
        "Actualizar pantalla cada N resultados",
        min_value=1,
        max_value=30,
        value=1,
        step=1,
    )
    use_cpu = st.checkbox("Forzar CPU", value=False)

input_tab, path_tab = st.tabs(["Cargar video", "Usar ruta local"])
with input_tab:
    uploaded_file = st.file_uploader(
        "Video ecográfico",
        type=["mp4", "avi", "mov", "mkv"],
    )
with path_tab:
    local_path = st.text_input(
        "Ruta del video",
        placeholder=r"C:\ruta\video_longitudinal.mp4",
    )

process_button = st.button(
    "Procesar video",
    type="primary",
    disabled=(uploaded_file is None and not local_path.strip()),
)

if process_button:
    session_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    session_dir = GUI_OUTPUT_ROOT / session_id
    capture_dir = session_dir / "captured_frames"
    session_dir.mkdir(parents=True, exist_ok=False)
    capture_dir.mkdir(parents=True, exist_ok=True)

    try:
        video_path = resolve_input_video(uploaded_file, local_path, session_dir)
        models, device = load_models_cached(use_cpu)
    except Exception as error:
        st.error(str(error))
        st.stop()

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        st.error(f"No se pudo abrir el video: {video_path}")
        st.stop()

    source_fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    total_source_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    expected = (
        (total_source_frames + int(frame_stride) - 1) // int(frame_stride)
        if total_source_frames > 0 else 0
    )
    if max_processed > 0:
        expected = min(expected, int(max_processed)) if expected else int(max_processed)

    ok, warmup_frame = capture.read()
    if not ok:
        st.error("No se pudo leer el primer frame.")
        st.stop()
    with st.spinner("Preparando GPU y modelos..."):
        for _ in range(3):
            infer_frame(warmup_frame, models, device, float(threshold))
    capture.set(cv2.CAP_PROP_POS_FRAMES, 0)

    temporal = TemporalGuidance(
        window_size=int(temporal_window),
        required_capture_streak=int(capture_streak),
    )
    progress = st.progress(0.0)
    status = st.empty()
    metric_cols = st.columns(4)
    original_slot, overlay_slot = st.columns(2)
    original_image = original_slot.empty()
    overlay_image = overlay_slot.empty()
    guidance_slot = st.empty()

    rows = []
    frame_id = 0
    processed = 0
    saved_captures = 0
    last_saved_at = -int(capture_cooldown)
    started = time.perf_counter()

    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if frame_id % int(frame_stride) != 0:
            frame_id += 1
            continue
        if max_processed > 0 and processed >= int(max_processed):
            break

        started_frame = time.perf_counter()
        inference = infer_frame(frame, models, device, float(threshold))
        temporal_result = temporal.update(inference.decision)
        processing_ms = (time.perf_counter() - started_frame) * 1000.0

        capture_saved = 0
        capture_path = ""
        if (
            temporal_result.capture_confirmed
            and processed - last_saved_at >= int(capture_cooldown)
        ):
            destination = (
                capture_dir
                / f"frame_{frame_id:06d}_timestamp_{frame_id / source_fps if source_fps else 0:.3f}.png"
            )
            cv2.imwrite(str(destination), frame)
            capture_path = str(destination)
            capture_saved = 1
            saved_captures += 1
            last_saved_at = processed

        row = {
            "frame_id": frame_id,
            "timestamp_s": frame_id / source_fps if source_fps > 0 else None,
            "view": view.lower(),
            "source_video": str(video_path),
            "source_fps": source_fps,
            "frame_stride": int(frame_stride),
            "raw_decision": temporal_result.raw_decision,
            "stable_decision": temporal_result.stable_decision,
            "stable_message": temporal_result.stable_message,
            "capture_streak": temporal_result.capture_streak,
            "capture_confirmed": temporal_result.capture_confirmed,
            "capture_saved": capture_saved,
            "capture_path": capture_path,
            "processing_ms": processing_ms,
            **inference.to_row(),
        }
        rows.append(row)
        processed += 1

        if processed % int(display_every) == 0:
            original_image.image(
                cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                caption=f"Frame original {frame_id}",
                width="stretch",
            )
            overlay = create_overlay(frame, inference)
            overlay_image.image(
                cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB),
                caption="ROI verde, hígado azul, LA roja",
                width="stretch",
            )
            with guidance_slot.container():
                show_guidance(
                    temporal_result.stable_decision,
                    temporal_result.stable_message,
                )
            metric_cols[0].metric("Frame", frame_id)
            metric_cols[1].metric("LA área", inference.area_la_px)
            metric_cols[2].metric("Racha válida", temporal_result.capture_streak)
            metric_cols[3].metric("Capturas", saved_captures)

        if expected > 0:
            progress.progress(min(processed / expected, 1.0))
        status.caption(
            f"Procesados: {processed} | Device: {device} | "
            f"Tiempo/frame: {processing_ms:.1f} ms"
        )
        frame_id += 1

    capture.release()
    elapsed_s = time.perf_counter() - started
    progress.progress(1.0)
    results = pd.DataFrame(rows)
    if results.empty:
        st.error("No se procesaron frames.")
        st.stop()

    csv_path = session_dir / "frame_results.csv"
    results.to_csv(csv_path, index=False, encoding="utf-8-sig")
    save_summary(session_dir, video_path, results, source_fps, elapsed_s)

    st.subheader("Resumen")
    final_metrics = st.columns(4)
    final_metrics[0].metric("Frames procesados", len(results))
    final_metrics[1].metric("Capturas guardadas", saved_captures)
    final_metrics[2].metric(
        "FPS de procesamiento",
        f"{1000.0 / results['processing_ms'].mean():.2f}",
    )
    final_metrics[3].metric(
        "LA detectada",
        int(results["has_la"].sum()),
    )
    decision_counts = (
        results["stable_decision"]
        .value_counts()
        .rename_axis("decision")
        .to_frame("frames")
    )
    st.bar_chart(decision_counts)
    st.dataframe(
        results[
            [
                "frame_id",
                "timestamp_s",
                "stable_decision",
                "has_roi",
                "has_higado",
                "has_la",
                "area_la_px",
                "la_std_intensity",
                "glcm_entropy",
                "capture_saved",
            ]
        ],
        width="stretch",
        hide_index=True,
    )
    st.download_button(
        "Descargar CSV de resultados",
        data=csv_path.read_bytes(),
        file_name=f"{session_id}_frame_results.csv",
        mime="text/csv",
    )
    st.caption(f"Resultados locales: {session_dir}")
