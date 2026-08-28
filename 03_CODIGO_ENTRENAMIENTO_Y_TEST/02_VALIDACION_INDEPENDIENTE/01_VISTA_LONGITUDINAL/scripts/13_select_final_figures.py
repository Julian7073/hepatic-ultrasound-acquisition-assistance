"""Selecciona y genera las figuras finales para la tesis."""

from __future__ import annotations

import argparse
import csv
import shutil
from datetime import datetime
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch
from PIL import Image


DEFAULT_ROOT = Path(os.environ.get("THESIS_PROJECT_ROOT", Path(__file__).resolve().parents[2]))
COLORS = {
    "blue": "#2563EB",
    "green": "#16A34A",
    "amber": "#D97706",
    "red": "#DC2626",
    "gray": "#64748B",
    "light": "#F1F5F9",
    "ink": "#0F172A",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def number(value: str | None, default: float = np.nan) -> float:
    try:
        return float(value) if value not in (None, "") else default
    except (TypeError, ValueError):
        return default


def save_figure(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def flow_figure(title: str, nodes: list[tuple[str, str]], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(13, 4.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    xs = np.linspace(0.08, 0.92, len(nodes))
    for index, ((label, color), x) in enumerate(zip(nodes, xs)):
        box = FancyBboxPatch(
            (x - 0.068, 0.38), 0.136, 0.25,
            boxstyle="round,pad=0.015,rounding_size=0.018",
            linewidth=1.5, edgecolor=color, facecolor="white",
        )
        ax.add_patch(box)
        ax.text(x, 0.505, label, ha="center", va="center", fontsize=10, color=COLORS["ink"], wrap=True)
        if index < len(nodes) - 1:
            ax.annotate(
                "", xy=(xs[index + 1] - 0.078, 0.505), xytext=(x + 0.078, 0.505),
                arrowprops=dict(arrowstyle="->", color=COLORS["gray"], lw=1.7),
            )
    ax.set_title(title, fontsize=16, fontweight="bold", color=COLORS["ink"], pad=18)
    save_figure(fig, path)


def figure_system(path: Path) -> None:
    nodes = [
        ("Video ecografico\npreviamente grabado", COLORS["blue"]),
        ("Seleccion\nde vista", COLORS["gray"]),
        ("Segmentacion\no DINOv2", COLORS["blue"]),
        ("Regla y\nestabilidad temporal", COLORS["amber"]),
        ("GUI y mensaje\nde asistencia", COLORS["green"]),
        ("CSV, captura o\nmejor candidato", COLORS["green"]),
    ]
    flow_figure("Flujo general del prototipo de asistencia", nodes, path)


def figure_longitudinal(path: Path) -> None:
    nodes = [
        ("Frame\nlongitudinal", COLORS["blue"]),
        ("ROI\nDeepLabV3+", COLORS["green"]),
        ("Higado\nDeepLabV3+", COLORS["green"]),
        ("LA\nU-Net", COLORS["green"]),
        ("Area, GLCM\ny borde", COLORS["amber"]),
        ("Decision\ntemporal", COLORS["red"]),
    ]
    flow_figure("Pipeline longitudinal final", nodes, path)


def figure_dino(path: Path) -> None:
    nodes = [
        ("Frame de video", COLORS["blue"]),
        ("Frame completo\no fan crop", COLORS["gray"]),
        ("DINOv2-Small\ncongelado", COLORS["blue"]),
        ("Ventana de\n5 embeddings", COLORS["amber"]),
        ("Clasificador\nbinario", COLORS["green"]),
        ("Adjust / Doubtful\n/ Capture", COLORS["red"]),
    ]
    flow_figure("Pipeline DINOv2 para otras vistas", nodes, path)


def figure_architectures(root: Path, path: Path) -> None:
    rows = read_csv(root / "outputs" / "experimental_segmentation_pipeline" / "reports" / "03_architecture_comparison.csv")
    architectures = ["U-Net", "DeepLabV3+", "SegFormer"]
    classes = ["ROI", "Higado"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6))
    for axis, class_name in zip(axes[:2], classes):
        values = []
        for architecture in architectures:
            match = next((row for row in rows if row.get("class_name") == class_name and row.get("architecture_label") == architecture), None)
            values.append(number(match.get("test_dice")) if match else np.nan)
        axis.bar(architectures, values, color=[COLORS["blue"], COLORS["green"], COLORS["amber"]])
        axis.set_ylim(0.75, 1.01)
        axis.set_title(f"{class_name}: Dice test")
        axis.tick_params(axis="x", rotation=20)
        axis.grid(axis="y", alpha=0.2)
        for i, value in enumerate(values):
            if np.isfinite(value):
                axis.text(i, value + 0.008, f"{value:.3f}", ha="center", fontsize=9)

    axis = axes[2]
    la_rows = [row for row in rows if row.get("class_name") == "LA"]
    positive = []
    fp = []
    for architecture in architectures:
        match = next((row for row in la_rows if row.get("architecture_label") == architecture), None)
        positive.append(number(match.get("test_positive_dice")) if match else np.nan)
        fp.append(number(match.get("test_empty_gt_false_positive_rate")) if match else np.nan)
    x = np.arange(len(architectures))
    axis.bar(x - 0.18, positive, 0.36, label="Dice positivo", color=COLORS["green"])
    axis.bar(x + 0.18, fp, 0.36, label="FP en vacias", color=COLORS["red"])
    axis.set_xticks(x, architectures, rotation=20)
    axis.set_ylim(0, 1)
    axis.set_title("LA: desempeño relevante")
    axis.legend(frameon=False, fontsize=8)
    axis.grid(axis="y", alpha=0.2)
    fig.suptitle("Comparacion controlada de arquitecturas", fontsize=16, fontweight="bold")
    fig.tight_layout()
    save_figure(fig, path)


def figure_resize(root: Path, path: Path) -> None:
    rows = read_csv(root / "outputs" / "experimental_segmentation_pipeline" / "reports" / "05_resize_resolution_comparison.csv")
    labels = []
    dice, fp, fps = [], [], []
    label_map = {
        "full_resize": "Full 512",
        "roi_crop_resize": "ROI crop 128",
        "original_or_padding": "Padding 512",
    }
    for row in rows:
        labels.append(label_map.get(row.get("resize_mode", ""), row.get("resize_mode", "")))
        dice.append(number(row.get("test_positive_dice")))
        fp.append(number(row.get("test_empty_gt_false_positive_rate")))
        fps.append(number(row.get("fps")))
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.3))
    for axis, values, title, color, limit in zip(
        axes, (dice, fp, fps), ("Dice positivo", "FP en mascaras vacias", "Velocidad (FPS)"),
        (COLORS["green"], COLORS["red"], COLORS["blue"]), ((0, 1), (0, 1), (0, max(fps or [1]) * 1.2)),
    ):
        axis.bar(labels, values, color=color)
        axis.set_title(title)
        axis.set_ylim(*limit)
        axis.tick_params(axis="x", rotation=20)
        axis.grid(axis="y", alpha=0.2)
    fig.suptitle("Ablation de resolucion para LA", fontsize=16, fontweight="bold")
    fig.tight_layout()
    save_figure(fig, path)


def figure_transfer(root: Path, path: Path) -> None:
    rows = read_csv(root / "outputs" / "experimental_segmentation_pipeline" / "reports" / "06_transfer_learning_comparison.csv")
    groups = [
        ("ROI DeepLabV3+", "ROI", "deeplabv3"),
        ("Higado DeepLabV3+", "Higado", "deeplabv3"),
        ("LA U-Net", "LA", "unet"),
        ("ROI SegFormer", "ROI", "segformer"),
    ]
    without, with_pretraining = [], []
    for _, class_name, architecture in groups:
        subset = [row for row in rows if row.get("class_name") == class_name and row.get("architecture") == architecture]
        metric = "test_positive_dice" if class_name == "LA" else "test_dice"
        no = next((row for row in subset if row.get("pretrained") == "False"), None)
        yes = next((row for row in subset if row.get("pretrained") == "True"), None)
        without.append(number(no.get(metric)) if no else np.nan)
        with_pretraining.append(number(yes.get(metric)) if yes else np.nan)
    x = np.arange(len(groups))
    fig, ax = plt.subplots(figsize=(12, 4.8))
    ax.bar(x - 0.19, without, 0.38, label="Sin pretraining", color=COLORS["gray"])
    ax.bar(x + 0.19, with_pretraining, 0.38, label="Con pretraining", color=COLORS["blue"])
    ax.set_xticks(x, [group[0] for group in groups], rotation=12)
    ax.set_ylim(0.75, 1.01)
    ax.set_ylabel("Dice de test")
    ax.set_title("Transferencia de aprendizaje")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.2)
    save_figure(fig, path)


def figure_benchmark(root: Path, path: Path) -> None:
    rows = read_csv(root / "outputs" / "experimental_segmentation_pipeline" / "reports" / "07_inference_benchmark.csv")
    wanted = ["ROI", "Higado", "LA", "pipeline_3_models_total"]
    selected = [next((row for row in rows if row.get("component") == name), {}) for name in wanted]
    values = [number(row.get("fps"), 0) for row in selected]
    labels = ["ROI", "Higado", "LA", "Pipeline total"]
    fig, ax = plt.subplots(figsize=(10, 4.8))
    bars = ax.bar(labels, values, color=[COLORS["green"], COLORS["green"], COLORS["green"], COLORS["red"]])
    ax.axhline(30, color=COLORS["amber"], linestyle="--", linewidth=2, label="Objetivo 30 FPS")
    ax.set_ylabel("FPS")
    ax.set_title("Benchmark de inferencia longitudinal")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.2)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 2, f"{value:.1f}", ha="center")
    save_figure(fig, path)


def copy_or_placeholder(source: Path, destination: Path, title: str) -> None:
    if source.exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.axis("off")
    ax.text(0.5, 0.55, title, ha="center", va="center", fontsize=18, fontweight="bold")
    ax.text(0.5, 0.43, "Evidencia no disponible en el cierre", ha="center", va="center", color=COLORS["gray"])
    save_figure(fig, destination)


def figure_dino_views(root: Path, path: Path) -> None:
    rows = read_csv(root / "outputs" / "dino_experimental" / "binary_improvement" / "reports" / "09_binary_p005_video_predictions.csv")
    views = ["transversal", "oblicua", "hepatorrenal"]
    qualities = ["clear", "medium", "blurry"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8), sharey=True)
    action_colors = {"capture": COLORS["green"], "doubtful": COLORS["amber"], "adjust": COLORS["red"]}
    for axis, view in zip(axes, views):
        subset = {row.get("true_quality"): row for row in rows if row.get("view") == view}
        values = [number(subset.get(quality, {}).get("probability_clear"), 0) for quality in qualities]
        actions = [subset.get(quality, {}).get("action", "missing") for quality in qualities]
        bars = axis.bar(qualities, values, color=[action_colors.get(action, COLORS["gray"]) for action in actions])
        axis.axhline(0.65, color=COLORS["green"], linestyle="--", linewidth=1)
        axis.axhline(0.35, color=COLORS["red"], linestyle=":", linewidth=1)
        axis.set_title(view.capitalize())
        axis.set_ylim(0, 1)
        axis.grid(axis="y", alpha=0.2)
        for bar, action in zip(bars, actions):
            axis.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.03, action, ha="center", fontsize=8, rotation=15)
    axes[0].set_ylabel("Probabilidad clear")
    fig.suptitle("Acciones DINOv2 en P005 por vista", fontsize=16, fontweight="bold")
    fig.tight_layout()
    save_figure(fig, path)


def latest_gui_session(root: Path, view: str) -> Path | None:
    sessions = []
    gui_root = root / "outputs" / "unified_gui_sessions"
    for session in gui_root.glob(f"*_{view}"):
        summary = session / "session_summary.csv"
        if not summary.exists() or not (session / "frame_results.csv").exists():
            continue
        rows = read_csv(summary)
        if rows and "paciente 005" in rows[0].get("video_path", "").lower() and "clear" in rows[0].get("video_path", "").lower():
            sessions.append(session)
    return max(sessions, key=lambda p: p.stat().st_mtime) if sessions else None


def session_image(session: Path | None) -> Path | None:
    if session is None:
        return None
    for name in ("best_informative_frame.png", "best_candidate_frame.png"):
        path = session / name
        if path.exists():
            return path
    captures = sorted((session / "captured_frames").glob("*.png")) if (session / "captured_frames").exists() else []
    return captures[0] if captures else None


def figure_gui(root: Path, path: Path) -> None:
    views = ["longitudinal", "transversal", "oblicua", "hepatorrenal"]
    images: list[Path | None] = []
    for view in views:
        if view == "longitudinal":
            candidate = root / "outputs" / "experimental_segmentation_pipeline" / "p005_longitudinal_final" / "overlays" / "medium_frame_000000.png"
            images.append(candidate if candidate.exists() else None)
        else:
            images.append(session_image(latest_gui_session(root, view)))
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5))
    for axis, view, image_path in zip(axes.ravel(), views, images):
        axis.axis("off")
        if image_path and image_path.exists():
            image = Image.open(image_path).convert("RGB")
            axis.imshow(image)
        else:
            axis.set_facecolor(COLORS["light"])
            axis.text(0.5, 0.5, "Sin imagen representativa", ha="center", va="center", transform=axis.transAxes)
        axis.set_title(view.capitalize(), fontsize=12, fontweight="bold")
    fig.suptitle("GUI experimental: procesamiento de videos previamente grabados", fontsize=16, fontweight="bold")
    fig.text(0.5, 0.02, "La interfaz genera mensajes, CSV, capturas y mejor frame/candidato; no esta conectada directamente al ecografo.", ha="center", fontsize=10)
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    save_figure(fig, path)


def write_index(folder: Path, rows: list[dict[str, str]]) -> None:
    csv_path = folder / "figures_index.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["nombre", "ruta", "que_muestra", "seccion_sugerida", "comentario"])
        writer.writeheader()
        writer.writerows(rows)
    lines = ["# Indice de figuras", "", "| Figura | Que muestra | Seccion sugerida | Interpretacion |", "|---|---|---|---|"]
    for row in rows:
        lines.append(f"| {row['nombre']} | {row['que_muestra']} | {row['seccion_sugerida']} | {row['comentario']} |")
    (folder / "figures_index.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--timestamp", default=datetime.now().strftime("%Y%m%d_%H%M"))
    args = parser.parse_args()
    root = args.project_root.resolve()
    evidence = root / "outputs" / f"TESIS_EVIDENCIA_FINAL_{args.timestamp}"
    folder = evidence / "07_figuras_para_tesis"
    folder.mkdir(parents=True, exist_ok=True)

    outputs = {
        "fig_01_sistema_completo.png": (figure_system, "Flujo general del sistema", "Metodologia", "Integra entrada, analisis, decision y evidencia."),
        "fig_02_pipeline_longitudinal.png": (figure_longitudinal, "Pipeline longitudinal", "Segmentacion longitudinal", "Resume segmentadores y regla final congelada."),
        "fig_03_pipeline_dino.png": (figure_dino, "Pipeline DINOv2", "DINOv2", "Muestra ventana temporal y abstencion."),
        "fig_04_comparacion_segmentacion.png": (lambda p: figure_architectures(root, p), "Comparacion de arquitecturas", "Resultados de segmentacion", "LA requiere metricas positivas y FP vacias."),
        "fig_05_ablation_resolucion.png": (lambda p: figure_resize(root, p), "Ablation de resolucion", "Ablations", "El crop 128 acelera pero eleva falsos positivos."),
        "fig_06_transfer_learning.png": (lambda p: figure_transfer(root, p), "Transfer learning", "Ablations", "El efecto depende de estructura y arquitectura."),
        "fig_07_benchmark_fps.png": (lambda p: figure_benchmark(root, p), "Benchmark longitudinal", "Benchmark", "Las redes individuales superan 30 FPS; el pipeline total no."),
        "fig_08_p005_longitudinal.png": (lambda p: copy_or_placeholder(root / "outputs" / "experimental_segmentation_pipeline" / "figures" / "p005_longitudinal_final_examples.png", p, "P005 longitudinal"), "P005 longitudinal", "Evaluacion operacional", "No hubo capturas confirmadas; el area LA fue limitante."),
        "fig_09_dino_vistas.png": (lambda p: figure_dino_views(root, p), "Resultados DINO por vista", "Resultados DINO", "P005 muestra comportamiento diferente entre vistas."),
        "fig_10_gui_final.png": (lambda p: figure_gui(root, p), "GUI final", "Interfaz", "Evidencia visual del procesamiento de videos grabados."),
    }
    index_rows = []
    for filename, (builder, description, section, comment) in outputs.items():
        destination = folder / filename
        builder(destination)
        index_rows.append({
            "nombre": filename,
            "ruta": str(destination),
            "que_muestra": description,
            "seccion_sugerida": section,
            "comentario": comment,
        })
    write_index(folder, index_rows)

    backup_figures = root / f"TESIS_VERSION_FINAL_{args.timestamp}" / "05_figuras"
    if backup_figures.parent.exists():
        backup_figures.mkdir(parents=True, exist_ok=True)
        for path in folder.glob("fig_*.png"):
            shutil.copy2(path, backup_figures / path.name)
        shutil.copy2(folder / "figures_index.csv", backup_figures / "figures_index.csv")
        shutil.copy2(folder / "figures_index.md", backup_figures / "figures_index.md")
    print(f"Figuras generadas: {folder}")


if __name__ == "__main__":
    main()
