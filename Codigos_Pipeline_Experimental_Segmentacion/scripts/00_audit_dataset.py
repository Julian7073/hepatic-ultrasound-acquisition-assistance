"""Ejecuta solamente la auditoria COCO y de leakage."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from config_experimental import ensure_directories
from src.split_audit import run_audit
if __name__ == "__main__":
    ensure_directories()
    audit, leakage = run_audit()
    print("\nAuditoria COCO:")
    print(audit.to_string(index=False))
    print(f"\nGrupos con posible leakage: {int(leakage['possible_leakage'].sum())}")
    print("Reportes guardados en outputs/experimental_segmentation_pipeline/reports")
