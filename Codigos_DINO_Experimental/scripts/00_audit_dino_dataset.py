"""Ejecuta la auditoria previa obligatoria para DINOv2."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dataset_audit import run_audit


if __name__ == "__main__":
    results = run_audit()
    summary = results["summary"]
    videos = results["videos"]
    duplicates = results["duplicates"]
    print("\nAuditoria DINOv2 finalizada.")
    print(f"Frames: {len(results['index'])}")
    print(f"Videos: {len(videos)}")
    print(f"Grupos con duplicados exactos: {len(duplicates)}")
    print(summary.to_string(index=False))
