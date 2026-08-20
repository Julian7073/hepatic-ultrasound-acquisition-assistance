"""GUI unificada para adquisicion ecografica hepatica asistida."""

from __future__ import annotations

import html
import sys
import time
import zipfile
from collections import Counter
from datetime import datetime
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import torch

ROOT = Path(__file__).resolve().parent
TESIS_ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(TESIS_ROOT))

from config_experimental import (
    FINAL_MODELS_ROOT,
    UNIFIED_GUI_OUTPUT_ROOT,
    ensure_directories,
)
from src.longitudinal_inference import (
    create_overlay,
    infer_frame,
    load_selected_models,
)
from src.longitudinal_quality_rules import load_decision_config
from src.temporal_guidance import TemporalGuidance

from Codigos_DINO_Experimental.src.binary_inference import (
    BinaryDinoVideoPredictor,
)
from Codigos_DINO_Experimental.src.ultrasound_preprocessing import (
    isolate_ultrasound_fan,
)


VIEW_LABELS = {
    "longitudinal": "Longitudinal",
    "transversal": "Transversal",
    "oblicua": "Oblicua",
    "hepatorrenal": "Hepatorrenal",
}
DINO_VIEWS = ("transversal", "oblicua", "hepatorrenal")


st.set_page_config(
    page_title="Adquisición ecográfica hepática",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container {padding-top: 1rem; padding-bottom: 1rem; max-width: 1500px;}
    [data-testid="stSidebar"] .block-container {padding-top: 1rem;}
    .status-banner {
        display: flex; align-items: center; gap: 0.65rem;
        border: 1px solid; border-radius: 8px;
        padding: 0.6rem 0.8rem; margin: 0.15rem 0 0.55rem 0;
    }
    .status-dot {width: 14px; height: 14px; border-radius: 50%; flex: 0 0 14px;}
    .status-copy {line-height: 1.2;}
    .status-copy strong {display: block; font-size: 0.95rem;}
    .status-copy span {font-size: 0.84rem;}
    .completion-banner {
        border: 1px solid #94a3b8; border-radius: 8px;
        background: #f8fafc; color: #334155;
        padding: 0.65rem 0.8rem; margin: 0.55rem 0;
        font-size: 0.92rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


STATE_META = {
    "informative": {
        "label": "INFORMATIVA",
        "background": "#dcfce7",
        "border": "#16a34a",
        "text": "#166534",
    },
    "not_informative": {
        "label": "NO INFORMATIVA",
        "background": "#fee2e2",
        "border": "#dc2626",
        "text": "#991b1b",
    },
    "analyzing": {
        "label": "ANALIZANDO",
        "background": "#f3f4f6",
        "border": "#6b7280",
        "text": "#374151",
    },
}


def user_state_from_decision(decision: str) -> str:
    if decision == "capture":
        return "informative"
    if decision in {"warming_up", "analyzing", ""}:
        return "analyzing"
    return "not_informative"


def binary_guidance_message(decision: str, fallback: str = "") -> str:
    messages = {
        "capture": (
            "Imagen informativa. Mantenga la posición del transductor."
        ),
        "no_structure": (
            "Imagen no informativa. Ajuste el contacto y la posición del "
            "transductor para recuperar el campo ecográfico."
        ),
        "partial_liver": (
            "Imagen no informativa. Ajuste la orientación o profundidad "
            "hasta incluir una mayor porción del hígado."
        ),
        "liver_without_reference": (
            "Imagen no informativa. Ajuste ligeramente la inclinación para "
            "visualizar la referencia anatomica requerida."
        ),
        "adjust": (
            "Imagen no informativa. Corrija la posición y orientación del "
            "transductor."
        ),
        "doubtful": (
            "Imagen no informativa. Mantenga la posición y realice un ajuste "
            "fino para confirmar la vista."
        ),
        "warming_up": "Analizando la secuencia antes de emitir una decisión.",
    }
    return messages.get(decision, fallback or "Analizando la secuencia.")


def render_status_state(state: str, message: str, target=None) -> None:
    meta = STATE_META.get(state, STATE_META["analyzing"])
    safe_message = html.escape(str(message))
    markup = f"""
    <div class="status-banner" style="
        background:{meta['background']};
        border-color:{meta['border']};
        color:{meta['text']};">
        <span class="status-dot" style="background:{meta['border']};"></span>
        <div class="status-copy">
            <strong>{meta['label']}</strong>
            <span>{safe_message}</span>
        </div>
    </div>
    """
    renderer = target if target is not None else st
    renderer.markdown(markup, unsafe_allow_html=True)


def render_status_indicator(decision: str, message: str, target=None) -> None:
    state = "analyzing" if decision == "warming_up" else user_state_from_decision(decision)
    render_status_state(
        state,
        binary_guidance_message(decision, message),
        target,
    )


def user_state_series(results: pd.DataFrame) -> pd.Series:
    return results["stable_decision"].map(user_state_from_decision)


def annotated_analysis_image(rgb: np.ndarray, decision: str) -> np.ndarray:
    """Superpone una unica decision visual verde, roja o neutral."""
    output = np.ascontiguousarray(rgb.copy())
    state = user_state_from_decision(decision)
    colors = {
        "informative": (22, 163, 74),
        "not_informative": (220, 38, 38),
        "analyzing": (107, 114, 128),
    }
    labels = {
        "informative": "INFORMATIVA",
        "not_informative": "NO INFORMATIVA",
        "analyzing": "ANALIZANDO",
    }
    color = colors[state]
    label = labels[state]
    height, width = output.shape[:2]
    thickness = max(4, int(round(min(height, width) * 0.008)))
    cv2.rectangle(
        output,
        (thickness // 2, thickness // 2),
        (width - thickness // 2 - 1, height - thickness // 2 - 1),
        color,
        thickness,
    )
    font_scale = max(0.55, min(1.0, width / 1100.0))
    text_thickness = max(1, int(round(font_scale * 2)))
    (text_width, text_height), baseline = cv2.getTextSize(
        label,
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        text_thickness,
    )
    pad = max(8, thickness)
    cv2.rectangle(
        output,
        (thickness, thickness),
        (
            min(width - thickness, thickness + text_width + 2 * pad),
            thickness + text_height + baseline + 2 * pad,
        ),
        color,
        -1,
    )
    cv2.putText(
        output,
        label,
        (thickness + pad, thickness + text_height + pad),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (255, 255, 255),
        text_thickness,
        cv2.LINE_AA,
    )
    return output


def informative_archive(capture_dir: Path) -> tuple[bytes | None, int]:
    images = sorted(capture_dir.glob("*.png"))
    if not images:
        return None, 0
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for image_path in images:
            archive.write(image_path, arcname=image_path.name)
    return buffer.getvalue(), len(images)


def render_informative_download(
    capture_dir: Path,
    view: str,
    key: str,
    target=None,
) -> int:
    renderer = target if target is not None else st
    archive_data, image_count = informative_archive(capture_dir)
    if archive_data is None:
        renderer.caption(
            "No se confirmaron imágenes informativas para descargar en esta sesión."
        )
        return 0
    renderer.download_button(
        "Descargar imágenes informativas",
        data=archive_data,
        file_name=f"{view}_imagenes_informativas.zip",
        mime="application/zip",
        key=key,
        width="stretch",
    )
    return image_count


def render_completion_notice(target=None) -> None:
    renderer = target if target is not None else st
    renderer.markdown(
        """
        <div class="completion-banner">
            <strong>Análisis finalizado.</strong>
            Revise la decisión binaria y descargue las imágenes informativas confirmadas.
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_compact_summary(
    results: pd.DataFrame,
    view: str,
    capture_dir: Path,
    csv_path: Path,
) -> None:
    states = user_state_series(results)
    informative = int((states == "informative").sum())
    final_state = "informative" if informative > 0 else "not_informative"
    final_message = (
        "Se confirmaron imágenes informativas durante el análisis."
        if informative > 0
        else "No se confirmaron imágenes informativas durante el análisis."
    )
    render_status_state(final_state, final_message)
    render_completion_notice()
    render_informative_download(
        capture_dir,
        view,
        key=f"summary_images_{csv_path.parent.name}",
    )
    with st.expander("Descargar evidencia tecnica", expanded=False):
        st.download_button(
            "Descargar resultados CSV",
            data=csv_path.read_bytes(),
            file_name=f"{csv_path.parent.name}_{view}.csv",
            mime="text/csv",
            key=f"summary_csv_{csv_path.parent.name}",
        )

def checkpoint_paths() -> dict[str, Path]:
    return {
        "ROI": FINAL_MODELS_ROOT / "best_roi_model.pth",
        "Higado": FINAL_MODELS_ROOT / "best_higado_model.pth",
        "LA": FINAL_MODELS_ROOT / "best_la_model.pth",
    }


@st.cache_resource(show_spinner="Cargando modelos longitudinales...")
def load_longitudinal_models_cached(use_cpu: bool):
    device = torch.device(
        "cpu" if use_cpu or not torch.cuda.is_available() else "cuda"
    )
    return load_selected_models(checkpoint_paths(), device), device


@st.cache_resource(show_spinner="Cargando DINOv2 y clasificador...")
def load_dino_predictor_cached(view: str, use_cpu: bool):
    device = "cpu" if use_cpu or not torch.cuda.is_available() else "cuda"
    return BinaryDinoVideoPredictor(view=view, device=device)


def resolve_input_video(uploaded_file, local_path: str, session_dir: Path) -> Path:
    if uploaded_file is not None:
        suffix = Path(uploaded_file.name).suffix.lower() or ".mp4"
        destination = session_dir / f"input_video{suffix}"
        destination.write_bytes(uploaded_file.getbuffer())
        return destination
    path = Path(local_path.strip().strip('"')).expanduser()
    if not path.exists():
        raise FileNotFoundError(
            f"No existe el video: {path}. Use la ruta real, no una ruta de ejemplo."
        )
    if not path.is_file():
        raise FileNotFoundError(f"La ruta no corresponde a un archivo: {path}")
    return path


def show_longitudinal_guidance(decision: str, message: str) -> None:
    if decision == "capture":
        st.success(binary_guidance_message(decision, message))
    else:
        st.error(binary_guidance_message(decision, message))


def show_dino_guidance(decision: str, message: str) -> None:
    if decision == "capture":
        st.success(binary_guidance_message(decision, message))
    else:
        st.error(binary_guidance_message(decision, message))


def write_session_summary(
    session_dir: Path,
    video_path: Path,
    view: str,
    results: pd.DataFrame,
    source_fps: float,
    elapsed_s: float,
) -> None:
    counts = Counter(results["stable_decision"])
    mean_ms = float(results["processing_ms"].mean())
    summary = pd.DataFrame([{
        "video_path": str(video_path),
        "view": view,
        "processed_frames": len(results),
        "source_fps": source_fps,
        "elapsed_s": elapsed_s,
        "mean_processing_ms": mean_ms,
        "estimated_processing_fps": 1000.0 / mean_ms if mean_ms > 0 else 0.0,
        "saved_capture_frames": int(results["capture_saved"].sum()),
        **{f"decision_{key}": value for key, value in counts.items()},
    }])
    summary.to_csv(
        session_dir / "session_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    lines = [
        "# Resumen de sesion de adquisicion",
        "",
        f"- Video: {video_path}",
        f"- Vista: {view}",
        f"- Frames evaluados: {len(results)}",
        f"- FPS fuente: {source_fps:.3f}",
        f"- Capturas guardadas: {int(results['capture_saved'].sum())}",
        f"- Tiempo medio por evaluacion: {mean_ms:.3f} ms",
        "",
        "Los mensajes son asistencia experimental y no constituyen diagnostico medico.",
    ]
    (session_dir / "session_summary.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def open_video(video_path: Path):
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"No se pudo abrir el video: {video_path}")
    source_fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    return capture, source_fps, total_frames


def expected_evaluations(total_frames: int, stride: int, maximum: int) -> int:
    expected = (
        (total_frames + stride - 1) // stride if total_frames > 0 else 0
    )
    if maximum > 0:
        expected = min(expected, maximum) if expected else maximum
    return expected


def clean_dino_display(rgb: np.ndarray) -> np.ndarray:
    """Devuelve el campo ecografico sin encabezado para revision visual."""
    result = isolate_ultrasound_fan(rgb)
    if result.detected:
        return result.image

    # Si el fan no se detecta, elimina cabecera y margenes de la interfaz.
    height, width = rgb.shape[:2]
    y0 = int(round(height * 0.10))
    x0 = int(round(width * 0.07))
    x1 = int(round(width * 0.93))
    return rgb[y0:, x0:x1].copy()


def read_video_frame(video_path: Path, frame_id: int) -> np.ndarray:
    """Lee un frame concreto sin mantener el video completo en memoria."""
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"No se pudo abrir el video: {video_path}")
    capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_id))
    ok, frame = capture.read()
    capture.release()
    if not ok:
        raise RuntimeError(f"No se pudo leer el frame {frame_id}.")
    return frame


def render_dino_frame_browser(view: str) -> None:
    """Permite revisar cualquier frame DINO evaluado en la sesion actual."""
    sessions = st.session_state.get("dino_review_sessions", {})
    review = sessions.get(view)
    if not review:
        return

    session_dir = Path(review["session_dir"])
    video_path = Path(review["video_path"])
    results_path = session_dir / "frame_results.csv"
    if not results_path.exists() or not video_path.exists():
        return

    results = pd.read_csv(results_path)
    if results.empty or "frame_id" not in results:
        return
    results["frame_id"] = pd.to_numeric(
        results["frame_id"], errors="coerce"
    ).astype("Int64")
    results = results.dropna(subset=["frame_id"]).copy()
    results["frame_id"] = results["frame_id"].astype(int)
    frame_ids = results["frame_id"].tolist()
    if not frame_ids:
        return

    with st.expander("Revisar cualquier frame analizado", expanded=False):
        selected_frame = st.select_slider(
            "Frame evaluado",
            options=frame_ids,
            value=frame_ids[0],
            key=f"dino_review_{view}_{session_dir.name}",
        )
        row = results.loc[results["frame_id"] == selected_frame].iloc[0]
        frame_bgr = read_video_frame(video_path, selected_frame)
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        clean_rgb = clean_dino_display(frame_rgb)

        original_column, clean_column = st.columns(2)
        original_column.image(
            frame_rgb,
            caption="Imagen adquirida",
            width=360,
        )
        stable_decision = str(row.get("stable_decision", "doubtful"))
        clean_column.image(
            annotated_analysis_image(clean_rgb, stable_decision),
            caption="Imagen analizada",
            width=360,
        )
        message = str(row.get("stable_message", "")).strip()
        if message:
            render_status_indicator(
                stable_decision,
                message,
            )

        if str(row.get("preprocessing", "")) == "full":
            st.caption(
                "La vista limpia facilita la revision; el modelo uso "
                "internamente el frame completo."
            )

def render_longitudinal_frame_browser(settings: dict) -> None:
    """Permite revisar cualquier frame longitudinal evaluado."""
    sessions = st.session_state.get("dino_review_sessions", {})
    review = sessions.get("longitudinal")
    if not review:
        return

    session_dir = Path(review["session_dir"])
    video_path = Path(review["video_path"])
    results_path = session_dir / "frame_results.csv"
    if not results_path.exists() or not video_path.exists():
        return

    results = pd.read_csv(results_path)
    if results.empty or "frame_id" not in results:
        return
    results["frame_id"] = pd.to_numeric(
        results["frame_id"], errors="coerce"
    ).astype("Int64")
    results = results.dropna(subset=["frame_id"]).copy()
    results["frame_id"] = results["frame_id"].astype(int)
    frame_ids = results["frame_id"].tolist()
    if not frame_ids:
        return

    with st.expander("Revisar cualquier frame analizado", expanded=False):
        selected_frame = st.select_slider(
            "Frame evaluado",
            options=frame_ids,
            value=frame_ids[0],
            key=f"longitudinal_review_{session_dir.name}",
        )
        row = results.loc[results["frame_id"] == selected_frame].iloc[0]
        frame_bgr = read_video_frame(video_path, selected_frame)
        models, device = load_longitudinal_models_cached(
            bool(review.get("use_cpu", settings["use_cpu"]))
        )
        inference = infer_frame(
            frame_bgr,
            models,
            device,
            float(review.get("threshold", settings["threshold"])),
            decision_config=load_decision_config(),
        )
        overlay = create_overlay(frame_bgr, inference)

        original_column, overlay_column = st.columns(2)
        original_column.image(
            cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB),
            caption="Imagen adquirida",
            width=360,
        )
        stable_decision = str(
            row.get("stable_decision", "liver_without_reference")
        )
        overlay_column.image(
            annotated_analysis_image(
                cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB),
                stable_decision,
            ),
            caption="Imagen analizada",
            width=360,
        )
        message = str(row.get("stable_message", "")).strip()
        if message:
            render_status_indicator(
                stable_decision,
                message,
            )


def run_longitudinal(
    video_path: Path,
    session_dir: Path,
    settings: dict,
) -> None:
    capture_dir = session_dir / "captured_frames"
    capture_dir.mkdir(parents=True, exist_ok=True)
    models, device = load_longitudinal_models_cached(settings["use_cpu"])
    decision_config = load_decision_config()
    capture, source_fps, total_frames = open_video(video_path)
    stride = settings["frame_stride"]
    expected = expected_evaluations(total_frames, stride, settings["max_processed"])

    ok, warmup_frame = capture.read()
    if not ok:
        capture.release()
        raise RuntimeError("No se pudo leer el primer frame.")
    with st.spinner("Preparando modelos longitudinales..."):
        for _ in range(3):
            infer_frame(
                warmup_frame,
                models,
                device,
                settings["threshold"],
                decision_config=decision_config,
            )
    capture.set(cv2.CAP_PROP_POS_FRAMES, 0)

    temporal = TemporalGuidance(
        window_size=settings["temporal_window"],
        required_capture_streak=settings["capture_streak"],
    )
    analysis_tab, summary_tab = st.tabs(["Análisis", "Resumen"])
    with analysis_tab:
        status_indicator = st.empty()
        render_status_state(
            "analyzing",
            "Preparando el análisis del video.",
            status_indicator,
        )
        original_slot, overlay_slot = st.columns(2)
        original_image = original_slot.empty()
        overlay_image = overlay_slot.empty()
        progress = st.progress(0.0)
        progress_text = st.empty()
        progress_text.caption("Análisis en curso...")
        completion_slot = st.empty()
        download_slot = st.empty()

    rows = []
    frame_id = 0
    processed = 0
    saved_captures = 0
    last_saved_at = -settings["capture_cooldown"]
    started = time.perf_counter()

    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if frame_id % stride != 0:
            frame_id += 1
            continue
        if settings["max_processed"] > 0 and processed >= settings["max_processed"]:
            break

        started_frame = time.perf_counter()
        inference = infer_frame(
            frame,
            models,
            device,
            settings["threshold"],
            decision_config=decision_config,
        )
        temporal_result = temporal.update(inference.decision)
        processing_ms = (time.perf_counter() - started_frame) * 1000.0

        capture_saved = 0
        capture_path = ""
        if (
            temporal_result.capture_confirmed
            and processed - last_saved_at >= settings["capture_cooldown"]
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

        rows.append({
            "frame_id": frame_id,
            "timestamp_s": frame_id / source_fps if source_fps > 0 else None,
            "view": "longitudinal",
            "source_video": str(video_path),
            "source_fps": source_fps,
            "frame_stride": stride,
            "raw_decision": temporal_result.raw_decision,
            "stable_decision": temporal_result.stable_decision,
            "stable_message": temporal_result.stable_message,
            "capture_streak": temporal_result.capture_streak,
            "capture_confirmed": temporal_result.capture_confirmed,
            "capture_saved": capture_saved,
            "capture_path": capture_path,
            "processing_ms": processing_ms,
            **inference.to_row(),
        })
        processed += 1

        if processed % settings["display_every"] == 0:
            original_image.image(
                cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                caption="Imagen adquirida",
                width=360,
            )
            overlay = create_overlay(frame, inference)
            analyzed_rgb = annotated_analysis_image(
                cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB),
                temporal_result.stable_decision,
            )
            overlay_image.image(
                analyzed_rgb,
                caption="Imagen analizada",
                width=360,
            )
            render_status_indicator(
                temporal_result.stable_decision,
                temporal_result.stable_message,
                status_indicator,
            )

        if expected > 0:
            progress.progress(min(processed / expected, 1.0))
        frame_id += 1

    capture.release()
    elapsed_s = time.perf_counter() - started
    progress.progress(1.0)
    results = pd.DataFrame(rows)
    if results.empty:
        raise RuntimeError("No se procesaron frames.")

    csv_path = session_dir / "frame_results.csv"
    results.to_csv(csv_path, index=False, encoding="utf-8-sig")
    review_sessions = dict(
        st.session_state.get("dino_review_sessions", {})
    )
    review_sessions["longitudinal"] = {
        "session_dir": str(session_dir),
        "video_path": str(video_path),
        "threshold": float(settings["threshold"]),
        "use_cpu": bool(settings["use_cpu"]),
    }
    st.session_state["dino_review_sessions"] = review_sessions
    write_session_summary(
        session_dir, video_path, "longitudinal", results, source_fps, elapsed_s
    )
    progress_text.caption("Análisis finalizado.")
    render_completion_notice(completion_slot)
    render_informative_download(
        capture_dir,
        "longitudinal",
        key=f"analysis_images_{session_dir.name}",
        target=download_slot,
    )

    with summary_tab:
        render_compact_summary(
            results,
            "longitudinal",
            capture_dir,
            csv_path,
        )


def run_dino(
    video_path: Path,
    session_dir: Path,
    view: str,
    settings: dict,
) -> None:
    capture_dir = session_dir / "captured_frames"
    capture_dir.mkdir(parents=True, exist_ok=True)
    predictor = load_dino_predictor_cached(view, settings["use_cpu"])
    predictor.reset()
    capture, source_fps, total_frames = open_video(video_path)
    stride = settings["frame_stride"]
    expected = expected_evaluations(total_frames, stride, settings["max_processed"])

    analysis_tab, summary_tab = st.tabs(["Análisis", "Resumen"])
    with analysis_tab:
        status_indicator = st.empty()
        render_status_state(
            "analyzing",
            "Analizando una secuencia corta del video.",
            status_indicator,
        )
        original_slot, analyzed_slot = st.columns(2)
        original_image = original_slot.empty()
        analyzed_image = analyzed_slot.empty()
        progress = st.progress(0.0)
        progress_text = st.empty()
        progress_text.caption("Análisis en curso...")
        completion_slot = st.empty()
        download_slot = st.empty()

    rows = []
    frame_id = 0
    processed = 0
    saved_captures = 0
    capture_streak = 0
    last_saved_at = -settings["capture_cooldown"]
    best_overall = None
    best_confirmed = None
    started = time.perf_counter()

    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if frame_id % stride != 0:
            frame_id += 1
            continue
        if settings["max_processed"] > 0 and processed >= settings["max_processed"]:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if predictor.device.type == "cuda":
            torch.cuda.synchronize()
        started_frame = time.perf_counter()
        prediction = predictor.predict_rgb(rgb)
        if predictor.device.type == "cuda":
            torch.cuda.synchronize()
        processing_ms = (time.perf_counter() - started_frame) * 1000.0

        raw_action = prediction["action"]
        if raw_action == "capture":
            capture_streak += 1
        else:
            capture_streak = 0
        capture_confirmed = int(
            raw_action == "capture"
            and capture_streak >= settings["capture_streak"]
        )
        if raw_action == "capture" and not capture_confirmed:
            stable_decision = "doubtful"
            stable_message = (
                "Imagen prometedora. Mantenga la posicion mientras se confirma "
                "la estabilidad temporal."
            )
        else:
            stable_decision = raw_action
            stable_message = prediction["message"]

        capture_saved = 0
        capture_path = ""
        if (
            capture_confirmed
            and processed - last_saved_at >= settings["capture_cooldown"]
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

        probability = prediction["probability_clear"]
        if not pd.isna(probability):
            candidate = {
                "frame": frame.copy(),
                "frame_id": frame_id,
                "timestamp_s": frame_id / source_fps if source_fps > 0 else None,
                "probability": float(probability),
            }
            if (
                best_overall is None
                or candidate["probability"] > best_overall["probability"]
            ):
                best_overall = candidate
            if capture_confirmed and (
                best_confirmed is None
                or candidate["probability"] > best_confirmed["probability"]
            ):
                best_confirmed = candidate

        rows.append({
            "frame_id": frame_id,
            "timestamp_s": frame_id / source_fps if source_fps > 0 else None,
            "view": view,
            "source_video": str(video_path),
            "source_fps": source_fps,
            "frame_stride": stride,
            "raw_decision": raw_action,
            "stable_decision": stable_decision,
            "stable_message": stable_message,
            "probability_informative": probability,
            "capture_streak": capture_streak,
            "capture_confirmed": capture_confirmed,
            "capture_saved": capture_saved,
            "capture_path": capture_path,
            "temporal_buffer_frames": prediction["buffer_frames"],
            "embedding_model_id": predictor.bundle["embedding_model_id"],
            "preprocessing": predictor.bundle["preprocessing"],
            "temporal_mode": predictor.bundle["temporal_mode"],
            "classifier": predictor.bundle["classifier"],
            "adjust_threshold": predictor.bundle["adjust_threshold"],
            "capture_threshold": predictor.bundle["capture_threshold"],
            "processing_ms": processing_ms,
        })
        processed += 1

        if processed % settings["display_every"] == 0:
            original_image.image(
                rgb,
                caption="Imagen adquirida",
                width=360,
            )
            analyzed = clean_dino_display(rgb)
            analyzed_image.image(
                annotated_analysis_image(analyzed, stable_decision),
                caption="Imagen analizada",
                width=360,
            )
            render_status_indicator(
                stable_decision,
                stable_message,
                status_indicator,
            )

        if expected > 0:
            progress.progress(min(processed / expected, 1.0))
        frame_id += 1

    capture.release()
    elapsed_s = time.perf_counter() - started
    progress.progress(1.0)
    results = pd.DataFrame(rows)
    if results.empty:
        raise RuntimeError("No se procesaron frames.")

    results["is_best_frame"] = 0
    results["best_frame_selection"] = ""
    results["best_frame_path"] = ""
    selected_best = best_confirmed or best_overall
    if selected_best is not None:
        selection = (
            "confirmed_capture"
            if best_confirmed is not None
            else "candidate_only"
        )
        filename = (
            "best_informative_frame.png"
            if selection == "confirmed_capture"
            else "best_candidate_frame.png"
        )
        best_path = session_dir / filename
        cv2.imwrite(str(best_path), selected_best["frame"])
        best_mask = results["frame_id"] == selected_best["frame_id"]
        results.loc[best_mask, "is_best_frame"] = 1
        results.loc[best_mask, "best_frame_selection"] = selection
        results.loc[best_mask, "best_frame_path"] = str(best_path)
        best_summary = pd.DataFrame([{
            "view": view,
            "frame_id": selected_best["frame_id"],
            "timestamp_s": selected_best["timestamp_s"],
            "probability_informative": selected_best["probability"],
            "selection": selection,
            "best_frame_path": str(best_path),
            "best_overall_frame_id": (
                best_overall["frame_id"] if best_overall is not None else None
            ),
            "best_overall_probability": (
                best_overall["probability"] if best_overall is not None else None
            ),
        }])
        best_summary.to_csv(
            session_dir / "best_frame_summary.csv",
            index=False,
            encoding="utf-8-sig",
        )
    csv_path = session_dir / "frame_results.csv"
    results.to_csv(csv_path, index=False, encoding="utf-8-sig")
    review_sessions = dict(
        st.session_state.get("dino_review_sessions", {})
    )
    review_sessions[view] = {
        "session_dir": str(session_dir),
        "video_path": str(video_path),
    }
    st.session_state["dino_review_sessions"] = review_sessions
    write_session_summary(
        session_dir, video_path, view, results, source_fps, elapsed_s
    )
    progress_text.caption("Análisis finalizado.")
    render_completion_notice(completion_slot)
    render_informative_download(
        capture_dir,
        view,
        key=f"analysis_images_{session_dir.name}",
        target=download_slot,
    )

    with summary_tab:
        render_compact_summary(
            results,
            view,
            capture_dir,
            csv_path,
        )


ensure_directories()
UNIFIED_GUI_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

st.title("Asistente de adquisición hepática")
st.caption(
    "Seleccione la vista, cargue un video y siga la retroalimentación binaria "
    "durante el análisis."
)

uploaded_file = None
local_path = ""

with st.sidebar:
    st.header("Nueva evaluacion")
    view_label = st.selectbox(
        "Vista ecográfica",
        options=list(VIEW_LABELS.values()),
    )
    view = next(
        key for key, label in VIEW_LABELS.items() if label == view_label
    )

    source_mode = st.radio(
        "Fuente del video",
        options=["Subir archivo", "Ruta local"],
    )
    if source_mode == "Subir archivo":
        uploaded_file = st.file_uploader(
            "Video ecografico",
            type=["mp4", "avi", "mov", "mkv"],
            key=f"upload_{view}",
        )
    else:
        local_path = st.text_input(
            "Ruta del video",
            placeholder=r"path\to\ultrasound_video.mp4",
            key=f"path_{view}",
        )

    with st.expander("Opciones avanzadas", expanded=False):
        if view == "longitudinal":
            frame_stride = int(st.number_input(
                "Procesar 1 de cada N frames",
                min_value=1, max_value=30, value=3, step=1,
                key="longitudinal_stride",
            ))
            threshold = float(st.slider(
                "Umbral de segmentacion",
                min_value=0.10, max_value=0.90, value=0.50, step=0.05,
            ))
            temporal_window = int(st.number_input(
                "Ventana temporal",
                min_value=1, max_value=15, value=5, step=2,
            ))
            capture_streak = int(st.number_input(
                "Frames informativos consecutivos",
                min_value=1, max_value=10, value=3, step=1,
                key="longitudinal_capture_streak",
            ))
        else:
            frame_stride = int(st.number_input(
                "Procesar 1 de cada N frames",
                min_value=1, max_value=30, value=5, step=1,
                key="dino_stride",
            ))
            threshold = 0.50
            temporal_window = 5
            capture_streak = int(st.number_input(
                "Decisiones informativas consecutivas",
                min_value=1, max_value=10, value=3, step=1,
                key="dino_capture_streak",
            ))

        capture_cooldown = int(st.number_input(
            "Intervalo entre capturas guardadas",
            min_value=1, max_value=100, value=10, step=1,
        ))
        max_processed = int(st.number_input(
            "Máximo de frames evaluados (0 = todo)",
            min_value=0, max_value=100000, value=0, step=10,
        ))
        display_every = int(st.number_input(
            "Actualizar pantalla cada N resultados",
            min_value=1, max_value=30, value=1, step=1,
        ))
        use_cpu = st.checkbox("Forzar CPU", value=False)

    has_video = uploaded_file is not None or bool(local_path.strip())
    process_button = st.button(
        "Analizar video",
        type="primary",
        width="stretch",
        disabled=not has_video,
    )
    st.caption("Uso experimental. No realiza diagnóstico médico.")

if view == "longitudinal":
    st.caption(
        "Vista longitudinal: segmenta ROI, hígado y lumen anecoico, "
        "y evalúa área, textura y borde."
    )
else:
    st.caption(
        "Vista DINOv2: analiza una ventana de frames y presenta una "
        "decisión binaria."
    )

if not process_button:
    render_status_state(
        "analyzing",
        "Cargue un video y pulse Analizar video para comenzar.",
    )

if process_button:
    session_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    session_dir = UNIFIED_GUI_OUTPUT_ROOT / f"{session_id}_{view}"
    session_dir.mkdir(parents=True, exist_ok=False)
    settings = {
        "frame_stride": frame_stride,
        "threshold": threshold,
        "temporal_window": temporal_window,
        "capture_streak": capture_streak,
        "capture_cooldown": capture_cooldown,
        "max_processed": max_processed,
        "display_every": display_every,
        "use_cpu": use_cpu,
    }

    try:
        video_path = resolve_input_video(uploaded_file, local_path, session_dir)
        if view == "longitudinal":
            run_longitudinal(video_path, session_dir, settings)
        else:
            run_dino(video_path, session_dir, view, settings)
        st.caption(f"Resultados guardados en: {session_dir}")
    except Exception as error:
        st.error(str(error))
        st.exception(error)
