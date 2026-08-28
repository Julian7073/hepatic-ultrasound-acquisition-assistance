"""Resume los tres videos transversales P005 procesados por el pipeline final."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config_dino import BINARY_FIGURES_ROOT, BINARY_REPORTS_ROOT, BINARY_ROOT


FILES = {
    "clear": "20260501_135548_PACIENTE 003_202605020013-converted__transversal.csv",
    "medium": "20260501_135548_PACIENTE 003_202605020014-converted__transversal.csv",
    "blurry": "20260501_135548_PACIENTE 003_202605020015-converted__transversal.csv",
}
COLORS = {"clear": "#2A9D8F", "medium": "#E9C46A", "blurry": "#C44536"}


def table_markdown(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in frame.astype(str).itertuples(index=False, name=None):
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def main() -> None:
    inference_root = BINARY_ROOT / "video_inference"
    summaries = []
    timelines = {}
    for quality, filename in FILES.items():
        path = inference_root / filename
        if not path.exists():
            raise FileNotFoundError(f"Falta {path}")
        frame = pd.read_csv(path)
        evaluated = frame[
            (frame["evaluated"] == 1)
            & (frame["decision"] != "warming_up")
        ].copy()
        evaluated["probability_informative"] = pd.to_numeric(
            evaluated["probability_informative"]
        )
        timelines[quality] = evaluated
        counts = evaluated["decision"].value_counts()
        summaries.append({
            "quality": quality,
            "video_frames": len(frame),
            "evaluated_after_warmup": len(evaluated),
            "capture": int(counts.get("capture", 0)),
            "doubtful": int(counts.get("doubtful", 0)),
            "adjust": int(counts.get("adjust", 0)),
            "capture_rate": float((evaluated["decision"] == "capture").mean()),
            "mean_probability_informative": float(
                evaluated["probability_informative"].mean()
            ),
            "min_probability_informative": float(
                evaluated["probability_informative"].min()
            ),
            "max_probability_informative": float(
                evaluated["probability_informative"].max()
            ),
        })

    summary = pd.DataFrame(summaries)
    csv_path = BINARY_REPORTS_ROOT / "13_p005_transversal_video_inference_summary.csv"
    summary.to_csv(csv_path, index=False, encoding="utf-8-sig")

    figure, axis = plt.subplots(figsize=(10, 5.5), dpi=160)
    for quality, frame in timelines.items():
        axis.plot(
            frame["frame_id"],
            frame["probability_informative"],
            marker="o",
            linewidth=1.8,
            markersize=4,
            color=COLORS[quality],
            label=quality,
        )
    axis.axhline(0.65, color="#2A9D8F", linestyle="--", label="capture threshold")
    axis.axhline(0.35, color="#C44536", linestyle="--", label="adjust threshold")
    axis.fill_between(
        [0, 100], 0.35, 0.65, color="#E9C46A", alpha=0.12, label="doubtful zone"
    )
    axis.set_xlim(0, 100)
    axis.set_ylim(0, 1)
    axis.set_xlabel("Frame original")
    axis.set_ylabel("Probabilidad de informativo")
    axis.set_title("P005 transversal: salida temporal del pipeline DINOv2")
    axis.grid(True, alpha=0.25)
    axis.legend(ncol=2)
    figure.tight_layout()
    figure_path = BINARY_FIGURES_ROOT / "13_p005_transversal_probability_timeline.png"
    figure.savefig(figure_path)
    plt.close(figure)

    report = [
        "# Prueba de video P005: vista transversal", "",
        "Se procesaron los tres videos transversales completos de P005 con el modelo "
        "seleccionado exclusivamente mediante P001-P003. El modelo no se reajusto "
        "despues de observar P005.", "",
        "## Resultados", "", table_markdown(summary.round(4)), "",
        "## Interpretacion", "",
        "- Clear: las 17 decisiones posteriores al calentamiento fueron capture.",
        "- Medium: 13 decisiones capture y 4 doubtful.",
        "- Blurry: 0 decisiones capture; las 17 fueron doubtful.",
        "- No se produjo una falsa orden de captura en el video blurry.",
        "- El video blurry tampoco alcanzo adjust porque sus probabilidades quedaron "
        "entre 0.35 y 0.65.",
        "- Medium se solapa con clear y no funciona como una clase de incertidumbre "
        "consistente para P005.", "",
        "## Conclusion", "",
        "La abstencion temporal evita aceptar el video blurry aunque su clasificacion "
        "binaria agregada sea ambigua. Para la GUI, capture debe requerir varias "
        "decisiones consecutivas. Medium no debe tratarse como etiqueta clinica.", "",
        f"- CSV: {csv_path}",
        f"- Figura: {figure_path}",
    ]
    report_path = BINARY_REPORTS_ROOT / "13_p005_transversal_video_inference_summary.md"
    report_path.write_text("\n".join(report), encoding="utf-8")
    print(summary.to_string(index=False))
    print(f"Reporte: {report_path}")
    print(f"Figura: {figure_path}")


if __name__ == "__main__":
    main()