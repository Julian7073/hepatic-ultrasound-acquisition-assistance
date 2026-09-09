"""Regenerate thesis result figures with percentage-formatted metrics.

This script changes presentation only. It uses the frozen CSV values and the
existing ultrasound panels; no model output or decision is recomputed.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[3]
REPO = Path(__file__).resolve().parents[2]
SOURCE_FIGURES = ROOT / "Overleaf_figures_optimized"
OUTPUT = ROOT / "Overleaf_figures_percentage_final"
REPO_FIGURES = REPO / "results" / "figures"
REPORTS = ROOT / "outputs" / "experimental_segmentation_pipeline" / "reports"

NAVY = "#0c2340"
GREEN = "#00873e"
RED = "#d7191c"


def save_both(fig: plt.Figure, name: str) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    REPO_FIGURES.mkdir(parents=True, exist_ok=True)
    for folder in (OUTPUT, REPO_FIGURES):
        fig.savefig(folder / name, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def make_resolution_figure() -> None:
    df = pd.read_csv(REPORTS / "05_resize_resolution_comparison.csv")
    labels = ["Full resize\n512", "Padding\n512", "ROI crop\n128"]
    dice = 100 * df["test_positive_dice"].to_numpy()
    fp = 100 * df["test_empty_gt_false_positive_rate"].to_numpy()
    fps = df["fps"].to_numpy()
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.2))
    fig.suptitle("LA input-resolution ablation", fontsize=17, fontweight="bold", color=NAVY)
    series = [(dice, "Positive-mask Dice (%)", GREEN, (0, 100)),
              (fp, "False positives on empty masks (%)", RED, (0, 100)),
              (fps, "Processing rate (FPS)", "#2463db", (0, max(fps) * 1.18))]
    for ax, (values, title, color, ylim) in zip(axes, series):
        bars = ax.bar(labels, values, color=color, width=0.66)
        ax.set_title(title, fontsize=12)
        ax.set_ylim(*ylim)
        ax.grid(axis="y", alpha=0.22)
        ax.tick_params(axis="x", labelrotation=0)
        for bar, value in zip(bars, values):
            label = f"{value:.1f}%" if "%" in title else f"{value:.1f}"
            ax.text(bar.get_x() + bar.get_width() / 2, value + (ylim[1] * 0.025),
                    label, ha="center", va="bottom", fontsize=10, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    save_both(fig, "fig_05_ablation_resolucion.png")


def make_transfer_figure() -> None:
    df = pd.read_csv(REPORTS / "06_transfer_learning_comparison.csv")
    rows = []
    for target in ("ROI", "Higado", "LA"):
        subset = df[df["class_name"] == target]
        for architecture in ("unet", "deeplabv3", "segformer"):
            pair = subset[subset["architecture"] == architecture]
            scratch = pair[pair["pretrained"] == False]["test_dice"].iloc[0] * 100  # noqa: E712
            pretrained = pair[pair["pretrained"] == True]["test_dice"].iloc[0] * 100  # noqa: E712
            rows.append((target, architecture, scratch, pretrained))
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 4.5), sharey=True)
    fig.suptitle("Transfer-initialization comparison", fontsize=17, fontweight="bold", color=NAVY)
    arch_labels = ["U-Net", "DeepLabV3+", "SegFormer"]
    for ax, target in zip(axes, ("ROI", "Higado", "LA")):
        part = [r for r in rows if r[0] == target]
        x = np.arange(3)
        scratch = [r[2] for r in part]
        pretrained = [r[3] for r in part]
        w = 0.36
        b1 = ax.bar(x - w / 2, scratch, w, label="From scratch", color="#9bb7d4")
        b2 = ax.bar(x + w / 2, pretrained, w, label="Pretrained", color="#174a7e")
        ax.set_title("Liver" if target == "Higado" else target, fontsize=13)
        ax.set_xticks(x, arch_labels, rotation=18, ha="right")
        ax.set_ylim(45, 101)
        ax.grid(axis="y", alpha=0.22)
        for bars in (b1, b2):
            for bar in bars:
                value = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2, value + 0.8, f"{value:.1f}%",
                        ha="center", va="bottom", fontsize=8, rotation=90)
    axes[0].set_ylabel("Test Dice (%)")
    axes[-1].legend(loc="lower right", frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    save_both(fig, "fig_06_transfer_learning.png")


def font(size: int, bold: bool = True) -> ImageFont.FreeTypeFont:
    path = Path(r"C:\Windows\Fonts\arialbd.ttf" if bold else r"C:\Windows\Fonts\arial.ttf")
    return ImageFont.truetype(str(path), size=size)


def recaption(name: str, top: int, captions: list[tuple[str, str, str]]) -> None:
    image = Image.open(SOURCE_FIGURES / name).convert("RGB")
    draw = ImageDraw.Draw(image)
    width, height = image.size
    draw.rectangle((0, top, width, height), fill="white")
    panel = width / len(captions)
    label_font = font(13)
    prob_font = font(13)
    for i, (label, probability, color) in enumerate(captions):
        center = (i + 0.5) * panel
        for y, text_value, text_font in ((top + 9, label, label_font), (top + 26, probability, prob_font)):
            box = draw.textbbox((0, 0), text_value, font=text_font)
            draw.text((center - (box[2] - box[0]) / 2, y), text_value, font=text_font, fill=color)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    REPO_FIGURES.mkdir(parents=True, exist_ok=True)
    image.save(OUTPUT / name, optimize=True)
    image.save(REPO_FIGURES / name, optimize=True)


def make_probability_figures() -> None:
    recaption(
        "fig_dino_transversal_internal_examples.png", 267,
        [
            ("(a) correct: GT=blurry, pred=blurry", "P(clear)=29.1%", GREEN),
            ("(b) correct: GT=blurry, pred=blurry", "P(clear)=19.4%", GREEN),
            ("(c) correct: GT=blurry, pred=blurry", "P(clear)=32.5%", GREEN),
            ("(d) error: GT=blurry, pred=clear", "P(clear)=61.2%", RED),
        ],
    )
    recaption(
        "fig_dino_oblicua_internal_examples.png", 267,
        [
            ("(a) correct: GT=blurry, pred=blurry", "P(clear)=35.1%", GREEN),
            ("(b) correct: GT=blurry, pred=blurry", "P(clear)=21.2%", GREEN),
            ("(c) correct: GT=blurry, pred=blurry", "P(clear)=36.4%", GREEN),
            ("(d) error: GT=blurry, pred=clear", "P(clear)=78.6%", RED),
        ],
    )
    recaption(
        "fig_dino_hepatorrenal_internal_examples.png", 217,
        [
            ("(a) correct: GT=blurry, pred=blurry", "P(clear)=41.9%", GREEN),
            ("(b) correct: GT=blurry, pred=blurry", "P(clear)=27.6%", GREEN),
            ("(c) correct: GT=blurry, pred=blurry", "P(clear)=28.2%", GREEN),
            ("(d) error: GT=blurry, pred=clear", "P(clear)=57.8%", RED),
            ("(e) error: GT=blurry, pred=clear", "P(clear)=55.7%", RED),
        ],
    )
    recaption(
        "fig_external_correct_incorrect_examples.png", 205,
        [
            ("(a) Longitudinal: error", "GT=clear; output=non-informative", RED),
            ("(b) Transverse: correct", "GT=clear; P(clear)=81.9%", GREEN),
            ("(c) Transverse: error", "GT=blurry; P(clear)=54.0%", RED),
            ("(d) Oblique: correct", "GT=clear; P(clear)=97.4%", GREEN),
            ("(e) Hepatorenal: correct", "GT=clear; P(clear)=61.2%", GREEN),
        ],
    )


if __name__ == "__main__":
    make_resolution_figure()
    make_transfer_figure()
    make_probability_figures()
    print(f"Wrote percentage-formatted figures to {OUTPUT} and {REPO_FIGURES}")
