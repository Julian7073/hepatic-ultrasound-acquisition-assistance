"""Auditoria reproducible de frames para DINOv2."""

from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from config_dino import (
    FRAMES_ROOT,
    IMAGE_EXTENSIONS,
    PATIENT_EXTERNAL,
    PATIENTS_DEVELOPMENT,
    QUALITIES,
    REPORTS_ROOT,
    VIEWS,
    ensure_directories,
)


FRAME_PATTERN = re.compile(r"^(?P<video>.+?)-converted_frame_(?P<frame>\d+)$", re.IGNORECASE)
EMBEDDED_PATIENT_PATTERN = re.compile(r"PACIENTE[ _-]*(?P<patient>\d{3})", re.IGNORECASE)


@dataclass(frozen=True)
class FrameRecord:
    patient: str
    role: str
    view: str
    quality: str
    video_id: str
    frame_number: int
    filename: str
    image_path: str
    width: int
    height: int
    readable: int
    file_size_bytes: int
    sha256: str
    embedded_patient: str
    patient_name_mismatch: int


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_filename(path: Path) -> tuple[str, int, str]:
    match = FRAME_PATTERN.match(path.stem)
    if match:
        video_id = match.group("video")
        frame_number = int(match.group("frame"))
    else:
        video_id = path.stem
        frame_number = -1
    patient_match = EMBEDDED_PATIENT_PATTERN.search(path.name)
    embedded_patient = (
        f"P{patient_match.group('patient')}" if patient_match else "unknown"
    )
    return video_id, frame_number, embedded_patient


def scan_frames() -> pd.DataFrame:
    rows = []
    patients = (*PATIENTS_DEVELOPMENT, PATIENT_EXTERNAL)
    for patient in patients:
        for view in VIEWS:
            for quality in QUALITIES:
                directory = FRAMES_ROOT / patient / view / quality
                paths = sorted(
                    path
                    for path in directory.glob("*")
                    if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
                ) if directory.exists() else []
                for path in paths:
                    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
                    readable = int(image is not None)
                    height, width = image.shape[:2] if readable else (0, 0)
                    video_id, frame_number, embedded_patient = parse_filename(path)
                    rows.append(asdict(FrameRecord(
                        patient=patient,
                        role="external_test" if patient == PATIENT_EXTERNAL else "development",
                        view=view,
                        quality=quality,
                        video_id=video_id,
                        frame_number=frame_number,
                        filename=path.name,
                        image_path=str(path),
                        width=int(width),
                        height=int(height),
                        readable=readable,
                        file_size_bytes=int(path.stat().st_size),
                        sha256=sha256(path),
                        embedded_patient=embedded_patient,
                        patient_name_mismatch=int(embedded_patient not in {patient, "unknown"}),
                    )))
    return pd.DataFrame(rows)


def summarize_groups(index: pd.DataFrame) -> pd.DataFrame:
    expected = pd.MultiIndex.from_product(
        [(*PATIENTS_DEVELOPMENT, PATIENT_EXTERNAL), VIEWS, QUALITIES],
        names=["patient", "view", "quality"],
    ).to_frame(index=False)
    observed = (
        index.groupby(["patient", "view", "quality"], as_index=False)
        .agg(
            image_count=("filename", "size"),
            unique_videos=("video_id", "nunique"),
            unreadable_images=("readable", lambda values: int((values == 0).sum())),
            width=("width", "median"),
            height=("height", "median"),
            patient_name_mismatches=("patient_name_mismatch", "sum"),
        )
    )
    summary = expected.merge(observed, on=["patient", "view", "quality"], how="left")
    numeric = [
        "image_count", "unique_videos", "unreadable_images", "width", "height",
        "patient_name_mismatches",
    ]
    summary[numeric] = summary[numeric].fillna(0).astype(int)
    summary["group_exists"] = (summary["image_count"] > 0).astype(int)
    summary["expected_frames_per_video"] = 101
    summary["count_consistent_with_101_per_video"] = (
        summary["image_count"] == summary["unique_videos"] * 101
    ).astype(int)
    return summary


def summarize_videos(index: pd.DataFrame) -> pd.DataFrame:
    return (
        index.groupby(
            ["patient", "role", "view", "quality", "video_id"],
            as_index=False,
        )
        .agg(
            frame_count=("filename", "size"),
            minimum_frame=("frame_number", "min"),
            maximum_frame=("frame_number", "max"),
            unreadable_images=("readable", lambda values: int((values == 0).sum())),
            width=("width", "median"),
            height=("height", "median"),
        )
        .sort_values(["patient", "view", "quality", "video_id"])
        .reset_index(drop=True)
    )


def exact_duplicate_report(index: pd.DataFrame) -> pd.DataFrame:
    rows = []
    duplicate_groups = index.groupby("sha256").filter(lambda group: len(group) > 1)
    if duplicate_groups.empty:
        return pd.DataFrame(columns=[
            "sha256", "duplicate_count", "patients", "videos", "paths",
            "cross_patient", "cross_video",
        ])
    for digest, group in duplicate_groups.groupby("sha256"):
        patients = sorted(group["patient"].unique())
        videos = sorted(group["video_id"].unique())
        rows.append({
            "sha256": digest,
            "duplicate_count": len(group),
            "patients": ";".join(patients),
            "videos": ";".join(videos),
            "paths": ";".join(group["image_path"]),
            "cross_patient": int(len(patients) > 1),
            "cross_video": int(len(videos) > 1),
        })
    return pd.DataFrame(rows)


def temporal_redundancy(index: pd.DataFrame, threshold: float = 0.005) -> pd.DataFrame:
    rows = []
    grouped = index[index["readable"] == 1].groupby(
        ["patient", "role", "view", "quality", "video_id"],
        sort=True,
    )
    for keys, group in grouped:
        group = group.sort_values(["frame_number", "filename"])
        differences = []
        previous = None
        for path_text in group["image_path"]:
            image = cv2.imread(path_text, cv2.IMREAD_GRAYSCALE)
            if image is None:
                continue
            small = cv2.resize(image, (64, 64), interpolation=cv2.INTER_AREA)
            normalized = small.astype(np.float32) / 255.0
            if previous is not None:
                differences.append(float(np.mean(np.abs(normalized - previous))))
            previous = normalized
        patient, role, view, quality, video_id = keys
        values = np.asarray(differences, dtype=np.float64)
        rows.append({
            "patient": patient,
            "role": role,
            "view": view,
            "quality": quality,
            "video_id": video_id,
            "adjacent_pairs": int(len(values)),
            "mean_adjacent_mad": float(values.mean()) if len(values) else np.nan,
            "median_adjacent_mad": float(np.median(values)) if len(values) else np.nan,
            "p95_adjacent_mad": float(np.percentile(values, 95)) if len(values) else np.nan,
            "near_identical_threshold": threshold,
            "near_identical_pairs": int((values <= threshold).sum()) if len(values) else 0,
            "near_identical_rate": float((values <= threshold).mean()) if len(values) else np.nan,
        })
    return pd.DataFrame(rows)


def table_markdown(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_Sin registros._"
    display = frame.copy().fillna("")
    lines = [
        "| " + " | ".join(display.columns) + " |",
        "| " + " | ".join(["---"] * len(display.columns)) + " |",
    ]
    for row in display.astype(str).itertuples(index=False, name=None):
        lines.append("| " + " | ".join(value.replace("|", "/") for value in row) + " |")
    return "\n".join(lines)


def run_audit() -> dict[str, pd.DataFrame]:
    ensure_directories()
    index = scan_frames()
    summary = summarize_groups(index)
    videos = summarize_videos(index)
    duplicates = exact_duplicate_report(index)
    redundancy = temporal_redundancy(index)

    outputs = {
        "index": index,
        "summary": summary,
        "videos": videos,
        "duplicates": duplicates,
        "redundancy": redundancy,
    }
    filenames = {
        "index": "00_dino_frame_index.csv",
        "summary": "00_dino_dataset_audit.csv",
        "videos": "00_dino_video_groups.csv",
        "duplicates": "00_dino_exact_duplicates.csv",
        "redundancy": "00_dino_temporal_redundancy.csv",
    }
    for name, frame in outputs.items():
        frame.to_csv(REPORTS_ROOT / filenames[name], index=False, encoding="utf-8-sig")

    development = index[index["role"] == "development"]
    external = index[index["role"] == "external_test"]
    cross_patient_duplicates = int(duplicates["cross_patient"].sum()) if not duplicates.empty else 0
    missing_groups = int((summary["group_exists"] == 0).sum())
    inconsistent_groups = int((summary["count_consistent_with_101_per_video"] == 0).sum())
    known_p005_mismatches = int(
        index[(index["patient"] == "P005") & (index["patient_name_mismatch"] == 1)].shape[0]
    )
    report = [
        "# Auditoria del dataset DINOv2", "",
        f"- Frames totales: {len(index)}",
        f"- Frames de desarrollo P001-P003: {len(development)}",
        f"- Frames externos P005: {len(external)}",
        f"- Videos fuente detectados: {videos['video_id'].nunique()}",
        f"- Grupos paciente/vista/calidad ausentes: {missing_groups}",
        f"- Grupos inconsistentes con 101 frames por video: {inconsistent_groups}",
        f"- Imágenes ilegibles: {int((index['readable'] == 0).sum())}",
        f"- Grupos de duplicados exactos: {len(duplicates)}",
        f"- Duplicados que cruzan pacientes: {cross_patient_duplicates}",
        f"- Nombres P005 con identificador interno distinto: {known_p005_mismatches}", "",
        "## Conteo por paciente, vista y calidad", "", table_markdown(summary), "",
        "## Videos fuente", "", table_markdown(videos), "",
        "## Riesgo metodologico", "",
        "Los frames de un mismo video son observaciones temporalmente correlacionadas. "
        "Nunca deben repartirse aleatoriamente entre entrenamiento y validación.", "",
        "P005 permanece completamente separado como prueba externa. Para validación interna se "
        "recomienda leave-one-patient-out con P001, P002 y P003, conservando videos completos.", "",
        "Los grupos con 202 frames corresponden a dos videos fuente de 101 frames, no a un único "
        "video duplicado. Deben conservarse como grupos separados y controlarse mediante video_id.", "",
        "El error de nombre interno de P005 se normaliza usando la carpeta P005; no se infiere el "
        "paciente desde el texto PACIENTE 003 del archivo.", "",
    ]
    (REPORTS_ROOT / "00_dino_dataset_audit.md").write_text(
        "\n".join(report), encoding="utf-8"
    )

    split_report = [
        "# Protocolo de separación DINOv2", "",
        "- Desarrollo: P001, P002 y P003.",
        "- Prueba externa final: P005.",
        "- Selección interna: leave-one-patient-out sobre los tres pacientes de desarrollo.",
        "- Unidad de agrupación mínima: video_id; ningún video puede cruzar particiones.",
        "- Las etiquetas iniciales son clear, medium y blurry.",
        "- Las métricas de P005 se calculan una sola vez después de seleccionar el clasificador.", "",
        "No se afirmará generalización clínica fuerte debido al número reducido de pacientes.",
    ]
    (REPORTS_ROOT / "00_dino_split_protocol.md").write_text(
        "\n".join(split_report), encoding="utf-8"
    )
    return outputs
