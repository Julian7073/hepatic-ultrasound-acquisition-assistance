"""Exporta aliases trazables de los modelos seleccionados por validacion."""

from __future__ import annotations

import hashlib
import shutil
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config_experimental import FINAL_MODELS_ROOT, REPORTS_ROOT, ensure_directories


def sha256(path: Path) -> str:
    """Calcula la huella del checkpoint exportado."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    ensure_directories()
    selection_path = REPORTS_ROOT / "best_models_by_class.csv"
    if not selection_path.exists():
        raise FileNotFoundError(
            "Falta best_models_by_class.csv. Ejecute scripts/08_update_reports.py primero."
        )

    selection = pd.read_csv(selection_path)
    aliases = {
        "ROI": "best_roi_model.pth",
        "Higado": "best_higado_model.pth",
        "LA": "best_la_model.pth",
    }
    rows = []
    for class_name, alias in aliases.items():
        selected = selection[selection["class_name"] == class_name]
        if selected.empty:
            raise RuntimeError(f"No existe modelo seleccionado para {class_name}.")
        row = selected.iloc[0]
        source = Path(str(row["checkpoint_path"]))
        if not source.exists():
            raise FileNotFoundError(f"No existe el checkpoint seleccionado: {source}")

        versioned = (
            FINAL_MODELS_ROOT
            / f"{class_name.lower()}__{row['experiment_name']}__best_model.pth"
        )
        destination = FINAL_MODELS_ROOT / alias
        shutil.copy2(source, versioned)
        shutil.copy2(source, destination)
        rows.append({
            "class_name": class_name,
            "architecture": row["architecture"],
            "experiment_name": row["experiment_name"],
            "selection_basis": row["selection_basis"],
            "source_checkpoint": str(source),
            "versioned_checkpoint": str(versioned),
            "alias_checkpoint": str(destination),
            "sha256": sha256(destination),
            "selection_is_final": False,
            "note": "Seleccion provisional por validacion; requiere evaluacion externa P005.",
        })

    manifest = pd.DataFrame(rows)
    manifest_path = FINAL_MODELS_ROOT / "selected_models_manifest.csv"
    manifest.to_csv(manifest_path, index=False, encoding="utf-8-sig")
    print(
        manifest[
            ["class_name", "architecture", "experiment_name", "alias_checkpoint"]
        ].to_string(index=False)
    )
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
