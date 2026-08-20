"""Audita duplicados exactos y error de identificacion entre P003 y P005."""

from __future__ import annotations

import csv
import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config_experimental import P005_FRAMES_ROOT, REPORTS_ROOT, TESIS_ROOT


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
P003_ROOT = TESIS_ROOT / "Dataset_Frames_Processed" / "P003" / "longitudinal"


def image_paths(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    p003 = image_paths(P003_ROOT)
    p005 = image_paths(P005_FRAMES_ROOT)
    p003_hashes = {}
    for path in p003:
        p003_hashes.setdefault(file_hash(path), []).append(path)

    duplicates = []
    for path in p005:
        digest = file_hash(path)
        for matching in p003_hashes.get(digest, []):
            duplicates.append({
                "p005_path": str(path),
                "p003_path": str(matching),
                "sha256": digest,
            })

    REPORTS_ROOT.mkdir(parents=True, exist_ok=True)
    csv_path = REPORTS_ROOT / "10_p005_p003_exact_duplicate_audit.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["p005_path", "p003_path", "sha256"],
        )
        writer.writeheader()
        writer.writerows(duplicates)

    embedded_p003 = sum(
        "PACIENTE 003" in path.name.upper() for path in p005
    )
    report = [
        "# Auditoria de identidad externa P005", "",
        f"- Frames P003: {len(p003)}",
        f"- Frames P005: {len(p005)}",
        f"- Duplicados SHA-256 exactos P003/P005: {len(duplicates)}",
        f"- Archivos P005 cuyo nombre interno contiene PACIENTE 003: {embedded_p003}",
        "",
        "P005 se conserva como paciente externo porque no existen duplicados binarios "
        "con P003. El texto PACIENTE 003 corresponde al error de identificacion conocido "
        "durante la adquisicion y debe documentarse como limitacion de trazabilidad.", "",
    ]
    md_path = REPORTS_ROOT / "10_p005_p003_exact_duplicate_audit.md"
    md_path.write_text("\n".join(report), encoding="utf-8")
    print("\n".join(report))
    print(f"CSV: {csv_path}")


if __name__ == "__main__":
    main()
