"""Congela el proyecto final y construye el paquete de evidencia de tesis."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


DEFAULT_ROOT = Path(os.environ.get("THESIS_PROJECT_ROOT", Path(__file__).resolve().parents[2]))
CODE_DIRS = (
    "Codigos_Pipeline_Experimental_Segmentacion",
    "Codigos_DINO_Experimental",
    "Codigos_Segmentacion_Longitudinal",
)
CODE_SUFFIXES = {
    ".py", ".md", ".txt", ".json", ".csv", ".bat", ".ps1",
    ".yaml", ".yml", ".toml", ".ini", ".cfg",
}
EXCLUDED_DIRS = {
    "__pycache__", ".venv", "venv", ".git", ".idea", ".vscode",
    "cache", "caches", ".cache", "checkpoints", "experiments",
    "outputs", "node_modules",
}
EXCLUDED_FILE_PREFIXES = ("~$", "~WRL")


@dataclass
class PlannedCopy:
    source: Path
    destination: Path
    category: str
    required: bool = False


class FreezeBuilder:
    def __init__(self, root: Path, timestamp: str, dry_run: bool) -> None:
        self.root = root.resolve()
        self.timestamp = timestamp
        self.dry_run = dry_run
        self.backup = self.root / f"TESIS_VERSION_FINAL_{timestamp}"
        self.evidence = self.root / "outputs" / f"TESIS_EVIDENCIA_FINAL_{timestamp}"
        self.planned: list[PlannedCopy] = []
        self.missing: list[tuple[str, str, str]] = []
        self.selected_sessions: dict[str, Path] = {}

    def add(
        self,
        source: Path,
        destination: Path,
        category: str,
        required: bool = False,
    ) -> None:
        source = source if source.is_absolute() else self.root / source
        if source.is_file():
            self.planned.append(PlannedCopy(source, destination, category, required))
        else:
            status = "missing critical" if required else "missing optional"
            self.missing.append((status, str(source), category))

    def add_to_both(
        self,
        source: Path,
        backup_relative: Path,
        evidence_relative: Path,
        category: str,
        required: bool = False,
    ) -> None:
        self.add(source, self.backup / backup_relative, category, required)
        self.add(source, self.evidence / evidence_relative, category, required)

    def add_code_tree(self, directory_name: str) -> None:
        source_root = self.root / directory_name
        if not source_root.exists():
            self.missing.append(("missing critical", str(source_root), "codigo"))
            return
        for source in source_root.rglob("*"):
            if not source.is_file():
                continue
            relative = source.relative_to(source_root)
            if any(part.lower() in EXCLUDED_DIRS for part in relative.parts[:-1]):
                continue
            if source.name.startswith(EXCLUDED_FILE_PREFIXES):
                continue
            if source.suffix.lower() not in CODE_SUFFIXES:
                continue
            destination = self.backup / "01_codigos_finales" / directory_name / relative
            self.planned.append(PlannedCopy(source, destination, "codigo_final", True))

    def plan_models(self) -> None:
        final_models = self.root / "outputs" / "experimental_segmentation_pipeline" / "final_models"
        for filename in (
            "best_roi_model.pth",
            "best_higado_model.pth",
            "best_la_model.pth",
            "selected_models_manifest.csv",
        ):
            self.add_to_both(
                final_models / filename,
                Path("02_modelos_finales") / filename,
                Path("02_modelos_y_hashes") / filename,
                "modelo_longitudinal",
                required=True,
            )

        dino_models = self.root / "outputs" / "dino_experimental" / "binary_improvement" / "models"
        for view in ("transversal", "oblicua", "hepatorrenal"):
            filename = f"{view}__binary_dinov2.joblib"
            self.add_to_both(
                dino_models / filename,
                Path("02_modelos_finales") / filename,
                Path("02_modelos_y_hashes") / filename,
                "modelo_dino",
                required=True,
            )

    def plan_reports(self) -> None:
        exp = self.root / "outputs" / "experimental_segmentation_pipeline"
        reports = exp / "reports"
        master = self.root / "outputs" / "reports" / "RECOPILACION_TECNICA_COMPLETA_TESIS.md"
        selected = [
            master,
            reports / "03_architecture_comparison_report.md",
            reports / "04_augmentation_comparison.md",
            reports / "05_resize_resolution_comparison.md",
            reports / "06_transfer_learning_comparison.md",
            reports / "07_inference_benchmark.md",
            reports / "08_final_experimental_summary.md",
            reports / "09_thesis_methodology_tables.md",
            reports / "10_longitudinal_rule_consistency_audit.md",
            reports / "11_final_longitudinal_methodology_update.md",
            reports / "RECOPILACION_TECNICA_ADDENDUM_REGLA_LONGITUDINAL_FINAL.md",
            exp / "p005_longitudinal_final" / "p005_longitudinal_final_report.md",
        ]
        dino_reports = self.root / "outputs" / "dino_experimental" / "binary_improvement" / "reports"
        selected += [
            dino_reports / "10_binary_dino_technical_report.md",
            dino_reports / "11_dino_binary_improvement_summary.md",
            dino_reports / "12_binary_dino_inference_benchmark.md",
            dino_reports / "13_p005_transversal_video_inference_summary.md",
        ]
        for source in selected:
            self.add(source, self.backup / "03_reportes_tecnicos" / source.name, "reporte_tecnico")

        methodology = [
            reports / "11_final_longitudinal_methodology_update.md",
            reports / "10_longitudinal_rule_consistency_audit.md",
            master,
        ]
        for source in methodology:
            self.add(source, self.evidence / "01_metodologia" / source.name, "metodologia")
        for source in selected:
            if "dino" in source.name.lower() or "p005_transversal" in source.name.lower():
                self.add(source, self.evidence / "04_resultados_dino_otras_vistas" / source.name, "reporte_dino")
        self.add(
            exp / "p005_longitudinal_final" / "p005_longitudinal_final_report.md",
            self.evidence / "06_resultados_p005" / "longitudinal" / "p005_longitudinal_final_report.md",
            "p005_longitudinal",
            True,
        )

    def plan_csv(self) -> None:
        exp = self.root / "outputs" / "experimental_segmentation_pipeline"
        reports = exp / "reports"
        files = [
            reports / "all_metrics.csv",
            reports / "all_benchmarks.csv",
            reports / "best_models_by_class.csv",
            reports / "03_architecture_comparison.csv",
            reports / "04_augmentation_comparison.csv",
            reports / "05_resize_resolution_comparison.csv",
            reports / "06_transfer_learning_comparison.csv",
            reports / "07_inference_benchmark.csv",
            exp / "final_models" / "selected_models_manifest.csv",
        ]
        p005 = exp / "p005_longitudinal_final"
        p005_files = [
            p005 / "frame_results.csv",
            p005 / "summary_by_quality.csv",
            p005 / "decision_summary.csv",
            p005 / "model_hash_verification.csv",
        ]
        dino = self.root / "outputs" / "dino_experimental" / "binary_improvement" / "reports"
        dino_files = [
            dino / "08_binary_winners_by_view.csv",
            dino / "08_binary_lopo_metrics_by_fold.csv",
            dino / "08_binary_lopo_video_predictions.csv",
            dino / "09_binary_abstention_thresholds.csv",
            dino / "09_binary_internal_action_metrics.csv",
            dino / "09_binary_model_manifest.csv",
            dino / "09_binary_p005_metrics.csv",
            dino / "09_binary_p005_video_predictions.csv",
            dino / "11_baseline_vs_binary_summary.csv",
            dino / "12_binary_dino_inference_benchmark.csv",
            dino / "13_p005_transversal_video_inference_summary.csv",
        ]
        for source in files + p005_files + dino_files:
            self.add(source, self.backup / "04_resultados_csv" / source.name, "resultado_csv")
        for source in files:
            self.add(source, self.evidence / "03_resultados_segmentacion_longitudinal" / source.name, "csv_segmentacion")
        for source in p005_files:
            self.add(source, self.evidence / "06_resultados_p005" / "longitudinal" / source.name, "csv_p005")
        for source in dino_files:
            self.add(source, self.evidence / "04_resultados_dino_otras_vistas" / source.name, "csv_dino")

    def plan_existing_figures(self) -> None:
        exp_figures = self.root / "outputs" / "experimental_segmentation_pipeline" / "figures"
        names = [
            "03_test_dice_by_architecture_class.png",
            "03_la_positive_dice_and_fp_rate.png",
            "04_la_augmentation_comparison.png",
            "05_la_resize_resolution_comparison.png",
            "06_transfer_learning_comparison.png",
            "07_inference_fps.png",
            "p005_longitudinal_final_examples.png",
        ]
        dino_figures = self.root / "outputs" / "dino_experimental" / "binary_improvement" / "figures"
        dino_names = [
            "06_fan_crop_qc.png",
            "08_binary_configuration_comparison.png",
            "09_binary_selected_confusion_matrices.png",
            "09_binary_abstention_thresholds.png",
            "09_binary_action_distribution.png",
            "12_binary_dino_inference_benchmark.png",
            "13_p005_transversal_probability_timeline.png",
        ]
        for name in names:
            self.add(exp_figures / name, self.backup / "05_figuras" / name, "figura_existente")
        for name in dino_names:
            self.add(dino_figures / name, self.backup / "05_figuras" / name, "figura_dino")

    def plan_overlays(self) -> None:
        exp_root = self.root / "outputs" / "experimental_segmentation_pipeline"
        selected = {
            "ROI": "deeplabv3_roi_50ep_group_video_earlystop",
            "Higado": "deeplabv3_higado_pretrained_imagenet_50ep",
            "LA": "unet_la_pretrained_imagenet_50ep_balanced",
        }
        for class_name, experiment in selected.items():
            overlay_root = exp_root / "experiments" / experiment / "overlays"
            positives = sorted(overlay_root.glob("positive_*.png"))[:5]
            failures = sorted(overlay_root.glob("worst_*.png"))[:3]
            for source in positives + failures:
                group = "correctos" if source.name.startswith("positive_") else "fallos"
                rel = Path("06_overlays_evidencia") / class_name / group / source.name
                self.add(source, self.backup / rel, f"overlay_{class_name.lower()}")
                ev = Path("08_overlays_y_casos") / f"longitudinal_{group}" / f"{class_name}_{source.name}"
                self.add(source, self.evidence / ev, f"overlay_{class_name.lower()}")

        p005_root = exp_root / "p005_longitudinal_final" / "overlays"
        p005_candidates = []
        for frame in (0, 3, 72):
            candidate = p005_root / f"medium_frame_{frame:06d}.png"
            if candidate.exists():
                p005_candidates.append(candidate)
        for source in p005_candidates:
            self.add(source, self.backup / "06_overlays_evidencia" / "P005" / source.name, "overlay_p005")
            self.add(source, self.evidence / "08_overlays_y_casos" / "longitudinal_fallos" / f"P005_{source.name}", "overlay_p005")

    def select_gui_sessions(self) -> None:
        root = self.root / "outputs" / "unified_gui_sessions"
        candidates: dict[str, list[Path]] = {view: [] for view in ("longitudinal", "transversal", "oblicua", "hepatorrenal")}
        if not root.exists():
            self.missing.append(("missing optional", str(root), "sesiones_gui"))
            return
        for session in root.iterdir():
            summary = session / "session_summary.csv"
            results = session / "frame_results.csv"
            if not session.is_dir() or not summary.exists() or not results.exists():
                continue
            try:
                with summary.open("r", encoding="utf-8-sig", newline="") as stream:
                    row = next(csv.DictReader(stream))
            except Exception:
                continue
            view = row.get("view", "")
            video = row.get("video_path", "").lower()
            if view in candidates and "paciente 005" in video and "clear" in video:
                candidates[view].append(session)
        for view, sessions in candidates.items():
            if sessions:
                self.selected_sessions[view] = max(sessions, key=lambda path: path.stat().st_mtime)
            else:
                self.missing.append(("missing optional", f"GUI P005 clear: {view}", "sesiones_gui"))

    def plan_gui_sessions(self) -> None:
        self.select_gui_sessions()
        allowed_names = {
            "frame_results.csv", "session_summary.csv", "session_summary.md",
            "best_frame_summary.csv", "best_informative_frame.png",
            "best_candidate_frame.png", "input_video.mp4", "input_video.avi",
            "input_video.mov", "input_video.mkv",
        }
        for view, session in self.selected_sessions.items():
            chosen: list[Path] = []
            for source in session.iterdir():
                if source.is_file() and source.name in allowed_names:
                    chosen.append(source)
            capture_dir = session / "captured_frames"
            if capture_dir.exists():
                chosen += sorted(path for path in capture_dir.iterdir() if path.is_file())[:3]
            for source in chosen:
                subdir = "captured_frames" if source.parent.name == "captured_frames" else ""
                destination_relative = Path(view) / session.name / subdir / source.name
                self.add(source, self.backup / "07_gui_sesiones" / destination_relative, "sesion_gui")
                self.add(source, self.evidence / "05_resultados_gui" / destination_relative, "sesion_gui")

    def plan_documents(self) -> None:
        requests = [
            (self.root / "Reuniones" / "RESEARCH PROJECT TOPIC FINAL.pdf", True),
            (self.root / "Reuniones" / "RESEARCH PROJECT PLAN FINAL.docx", True),
            (self.root / "outputs" / "segmentation_training" / "reports" / "documento_tecnico_segmentacion_longitudinal.docx", False),
            (self.root / "3.docx", False),
            (self.root / "Guide to write justification.pdf", False),
            (self.root / "outputs" / "reports" / "RECOPILACION_TECNICA_COMPLETA_TESIS.md", True),
            (self.root / "outputs" / "experimental_segmentation_pipeline" / "reports" / "11_final_longitudinal_methodology_update.md", True),
        ]
        for source, required in requests:
            self.add(source, self.backup / "08_documentos_base" / source.name, "documento_base", required)

    def plan_all(self) -> None:
        for directory in CODE_DIRS:
            self.add_code_tree(directory)
        adquisicion = self.root / "Códigos" / "AdquisicionFrames.py"
        self.add(adquisicion, self.backup / "01_codigos_finales" / "AdquisicionFrames.py", "codigo_final")
        self.plan_models()
        self.plan_reports()
        self.plan_csv()
        self.plan_existing_figures()
        self.plan_overlays()
        self.plan_gui_sessions()
        self.plan_documents()

    def ensure_structure(self) -> None:
        backup_dirs = [
            "00_manifest", "01_codigos_finales", "02_modelos_finales",
            "03_reportes_tecnicos", "04_resultados_csv", "05_figuras",
            "06_overlays_evidencia", "07_gui_sesiones", "08_documentos_base",
            "09_logs_reproducibilidad",
        ]
        evidence_dirs = [
            "01_metodologia", "02_modelos_y_hashes",
            "03_resultados_segmentacion_longitudinal",
            "04_resultados_dino_otras_vistas", "05_resultados_gui",
            "06_resultados_p005", "07_figuras_para_tesis",
            "08_overlays_y_casos", "09_anexos_reproducibilidad",
            "10_documento_word_final",
        ]
        for name in backup_dirs:
            (self.backup / name).mkdir(parents=True, exist_ok=True)
        for name in evidence_dirs:
            (self.evidence / name).mkdir(parents=True, exist_ok=True)

    def copy_planned(self) -> None:
        seen: set[Path] = set()
        for item in self.planned:
            destination = item.destination.resolve()
            if destination in seen:
                continue
            seen.add(destination)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item.source, destination)

    def generated_methodology(self) -> None:
        folder = self.evidence / "01_metodologia"
        files = {
            "resumen_metodologico_final.md": """# Resumen metodologico final\n\nEl prototipo procesa videos ecograficos previamente grabados. La vista longitudinal usa segmentacion binaria de ROI, Higado y LA seguida de una regla interpretable congelada. Las vistas transversal, oblicua y hepatorrenal usan DINOv2-Small, una ventana temporal de cinco embeddings y un clasificador binario con abstencion. La GUI Streamlit organiza cada ejecucion en una sesion reproducible. El sistema no captura directamente desde un ecografo, no diagnostica y no ha sido validado clinicamente.\n""",
            "tabla_modelos_finales.md": """# Modelos finales\n\n| Vista/estructura | Modelo final |\n|---|---|\n| ROI longitudinal | DeepLabV3+ ResNet-34 sin pretraining |\n| Higado longitudinal | DeepLabV3+ ResNet-34 ImageNet |\n| LA longitudinal | U-Net ResNet-34 ImageNet + balanced_la |\n| Transversal | DINOv2-Small + Random Forest + window5 |\n| Oblicua | DINOv2-Small + Regresion logistica + window5 |\n| Hepatorrenal | DINOv2-Small + k-NN + window5 |\n""",
            "tabla_hiperparametros.md": """# Hiperparametros principales\n\n| Parametro | Segmentacion | DINO |\n|---|---|---|\n| Split | group_video | LOPO P001-P003 |\n| Resolucion | 512x512 | preprocesamiento por vista |\n| Batch | 2 | no aplica al clasificador desplegado |\n| Epocas maximas | 50 | modelos clasicos sobre embeddings |\n| Learning rate | 0.001 | segun estimador seleccionado |\n| Ventana temporal | 5 decisiones GUI | 5 embeddings |\n""",
            "tabla_metricas.md": """# Metricas\n\n| Metrica | Uso |\n|---|---|\n| Dice e IoU | solapamiento de segmentacion |\n| Precision y Recall | errores de inclusion y omision |\n| combined_la_score | Dice positivo menos tasa FP en mascaras vacias |\n| Macro F1 / balanced accuracy | clasificacion binaria DINO |\n| FPS y ms/frame | capacidad operacional |\n""",
            "tabla_limitaciones.md": """# Limitaciones\n\n- Pocos pacientes y alta correlacion temporal.\n- P005 longitudinal no tiene ground truth.\n- Las calidades clear/medium/blurry son nominales.\n- No existe estudio con usuarios ni validacion clinica.\n- El pipeline longitudinal completo no alcanza 30 FPS sin stride.\n- La probabilidad DINO no fue calibrada clinicamente.\n""",
            "descripcion_gui_final.md": """# Descripcion final de la GUI\n\nLa GUI fue desarrollada con Streamlit y procesa videos ecograficos previamente grabados; no captura directamente desde una camara o ecografo en tiempo real. Permite subir MP4, AVI, MOV o MKV o indicar una ruta local. Cada ejecucion crea una carpeta de sesion. Longitudinal usa segmentacion ROI/Higado/LA; transversal, oblicua y hepatorrenal usan DINOv2. La interfaz muestra mensajes, estados por color, porcentajes informativos y graficas, y guarda CSV, resumen, capturas y el mejor frame o candidato. Es experimental, no diagnostica.\n""",
        }
        for name, content in files.items():
            (folder / name).write_text(content, encoding="utf-8")

        model_readme = self.evidence / "02_modelos_y_hashes" / "modelos_finales_readme.md"
        model_readme.write_text(
            "# Modelos finales\n\nLos archivos de esta carpeta son copias de despliegue. "
            "Los pesos no fueron modificados durante el cierre. Ver `sha256_manifest.csv`.\n",
            encoding="utf-8",
        )

    def write_logs(self) -> None:
        backup_logs = self.backup / "09_logs_reproducibilidad"
        evidence_logs = self.evidence / "09_anexos_reproducibilidad"
        now = datetime.now().isoformat(timespec="seconds")
        env_lines = [
            f"freeze_timestamp={self.timestamp}",
            f"generated_at={now}",
            f"python={sys.version}",
            f"platform={platform.platform()}",
            f"processor={platform.processor()}",
        ]
        for package in ("torch", "opencv-python", "numpy", "pandas", "matplotlib", "python-docx", "streamlit", "scikit-learn"):
            try:
                env_lines.append(f"{package}={importlib.metadata.version(package)}")
            except importlib.metadata.PackageNotFoundError:
                env_lines.append(f"{package}=not installed in freeze runtime")
        environment = "\n".join(env_lines) + "\n"
        distributions = sorted(
            f"{dist.metadata['Name']}=={dist.version}"
            for dist in importlib.metadata.distributions()
            if dist.metadata.get("Name")
        )
        pip_freeze = "\n".join(distributions) + "\n"
        hardware = (
            "Hardware registrado en los experimentos finales:\n"
            "GPU: NVIDIA GeForce RTX 4060 Laptop GPU\n"
            "CPU: Intel64 Family 6 Model 186 Stepping 2, GenuineIntel\n"
            "CUDA: 12.6\n"
            "Nota: datos tomados de config.json y benchmarks finales.\n"
        )
        commands = f"""# Comandos de reproduccion\n\n## GUI\n\n```powershell\ncd \"{self.root}\"\npython -m streamlit run .\\Codigos_Pipeline_Experimental_Segmentacion\\gui_adquisicion_hepatica.py\n```\n\n## P005 longitudinal\n\n```powershell\npython .\\Codigos_Pipeline_Experimental_Segmentacion\\evaluate_p005_longitudinal_final.py --frame_stride 3 --save_overlays --save_csv --decision_config .\\Codigos_Pipeline_Experimental_Segmentacion\\configs\\longitudinal_decision_config.json\n```\n\n## Freeze\n\n```powershell\npython .\\Codigos_Pipeline_Experimental_Segmentacion\\scripts\\12_freeze_final_project.py --timestamp {self.timestamp}\n```\n"""
        warnings = """# Advertencias metodologicas\n\n- Prototipo funcional, no sistema diagnostico.\n- Procesa videos previamente grabados; no esta conectado al ecografo.\n- P005 longitudinal no tiene ground truth.\n- No afirmar generalizacion clinica.\n- No reajustar los umbrales con P005.\n- Los nombres internos de videos P005 contienen `PACIENTE 003`; la asignacion se basa en la ruta.\n"""
        paths = f"""# Rutas principales\n\n- Proyecto: `{self.root}`\n- Backup: `{self.backup}`\n- Evidencia: `{self.evidence}`\n- Modelos longitudinales: `outputs/experimental_segmentation_pipeline/final_models`\n- Bundles DINO: `outputs/dino_experimental/binary_improvement/models`\n"""
        payloads = {
            "environment_versions.txt": environment,
            "pip_freeze.txt": pip_freeze,
            "hardware_summary.txt": hardware,
            "commands_reproduction.md": commands,
            "rutas_principales.md": paths,
            "advertencias_metodologicas.md": warnings,
        }
        for name, content in payloads.items():
            (backup_logs / name).write_text(content, encoding="utf-8")
            (evidence_logs / name).write_text(content, encoding="utf-8")

    @staticmethod
    def sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    def build_inventory(self) -> list[dict[str, str | int]]:
        original_map = {str(item.destination.resolve()): str(item.source.resolve()) for item in self.planned}
        rows: list[dict[str, str | int]] = []
        for package_name, package_root in (("backup", self.backup), ("evidence", self.evidence)):
            if not package_root.exists():
                continue
            for path in sorted(item for item in package_root.rglob("*") if item.is_file()):
                if path.name in {"file_inventory.csv", "sha256_manifest.csv"}:
                    continue
                stat = path.stat()
                rows.append({
                    "package": package_name,
                    "relative_path": path.relative_to(package_root).as_posix(),
                    "original_path": original_map.get(str(path.resolve()), "generated"),
                    "size_bytes": stat.st_size,
                    "modified_time": datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds"),
                    "sha256": self.sha256(path),
                    "category": path.relative_to(package_root).parts[0],
                })
        return rows

    def write_manifests(self) -> None:
        rows = self.build_inventory()
        fields = ["package", "relative_path", "original_path", "size_bytes", "modified_time", "sha256", "category"]
        manifest_dir = self.backup / "00_manifest"
        inventory_path = manifest_dir / "file_inventory.csv"
        with inventory_path.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        hash_fields = ["package", "relative_path", "size_bytes", "sha256"]
        hash_path = manifest_dir / "sha256_manifest.csv"
        with hash_path.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=hash_fields)
            writer.writeheader()
            writer.writerows({field: row[field] for field in hash_fields} for row in rows)
        for destination in (
            self.evidence / "02_modelos_y_hashes" / "sha256_manifest.csv",
            self.evidence / "09_anexos_reproducibilidad" / "sha256_manifest.csv",
        ):
            shutil.copy2(hash_path, destination)
        shutil.copy2(inventory_path, self.evidence / "09_anexos_reproducibilidad" / "file_inventory.csv")

        missing_lines = ["# Reporte de archivos faltantes", ""]
        if self.missing:
            missing_lines += ["| Estado | Ruta | Categoria |", "|---|---|---|"]
            missing_lines += [f"| {status} | `{path}` | {category} |" for status, path, category in self.missing]
        else:
            missing_lines.append("No se detectaron archivos faltantes.")
        (manifest_dir / "missing_files_report.md").write_text("\n".join(missing_lines) + "\n", encoding="utf-8")

        backup_size = sum(path.stat().st_size for path in self.backup.rglob("*") if path.is_file())
        evidence_size = sum(path.stat().st_size for path in self.evidence.rglob("*") if path.is_file())
        critical_missing = [item for item in self.missing if item[0] == "missing critical"]
        summary = f"""# Resumen de congelamiento\n\n- Timestamp: `{self.timestamp}`\n- Archivos inventariados: {len(rows)}\n- Backup: `{self.backup}`\n- Evidencia: `{self.evidence}`\n- Tamano backup: {backup_size / (1024**2):.2f} MiB\n- Tamano evidencia: {evidence_size / (1024**2):.2f} MiB\n- Faltantes criticos: {len(critical_missing)}\n- Faltantes opcionales: {len(self.missing) - len(critical_missing)}\n\nNo se entrenaron modelos ni se modificaron pesos o umbrales.\n"""
        (manifest_dir / "freeze_summary.md").write_text(summary, encoding="utf-8")
        readme = f"""# Version final congelada de tesis\n\nEsta carpeta conserva el codigo, modelos, resultados y evidencia seleccionada al {self.timestamp}. Los datasets completos y experimentos intermedios no se duplican. Ver `file_inventory.csv` y `sha256_manifest.csv` para trazabilidad.\n\nEl sistema es un prototipo experimental que procesa videos previamente grabados. No realiza diagnostico ni ha sido validado clinicamente.\n"""
        (manifest_dir / "VERSION_FINAL_README.md").write_text(readme, encoding="utf-8")

        validation = f"""# Validacion del paquete final\n\n- Backup creado: {'si' if self.backup.exists() else 'no'}\n- Paquete de evidencia creado: {'si' if self.evidence.exists() else 'no'}\n- Archivos inventariados: {len(rows)}\n- Modelos longitudinales presentes: {sum((self.evidence / '02_modelos_y_hashes' / name).exists() for name in ('best_roi_model.pth','best_higado_model.pth','best_la_model.pth'))}/3\n- Bundles DINO presentes: {sum((self.evidence / '02_modelos_y_hashes' / f'{view}__binary_dinov2.joblib').exists() for view in ('transversal','oblicua','hepatorrenal'))}/3\n- selected_models_manifest.csv presente: {'si' if (self.evidence / '02_modelos_y_hashes' / 'selected_models_manifest.csv').exists() else 'no'}\n- Word final: `{self.evidence / '10_documento_word_final' / 'RESULTADOS_TECNICOS_TESIS_FINAL.docx'}`\n- Faltantes criticos: {len(critical_missing)}\n- Faltantes opcionales: {len(self.missing) - len(critical_missing)}\n- Tamano backup: {backup_size / (1024**2):.2f} MiB\n- Tamano evidencia: {evidence_size / (1024**2):.2f} MiB\n\nRecomendacion: entregar el Word, el paquete de evidencia y el manifiesto de hashes; conservar el backup completo como archivo maestro.\n"""
        (self.evidence / "FINAL_PACKAGE_VALIDATION.md").write_text(validation, encoding="utf-8")

    def dry_run_report(self) -> Path:
        unique_destinations = {item.destination.resolve(): item for item in self.planned}
        size = sum(item.source.stat().st_size for item in unique_destinations.values())
        critical = [item for item in self.missing if item[0] == "missing critical"]
        report = self.root / "outputs" / f"freeze_dry_run_{self.timestamp}.md"
        lines = [
            "# Dry-run de congelamiento final", "",
            f"- Timestamp: `{self.timestamp}`",
            f"- Backup planificado: `{self.backup}`",
            f"- Evidencia planificada: `{self.evidence}`",
            f"- Copias planificadas: {len(unique_destinations)}",
            f"- Tamano estimado: {size / (1024**2):.2f} MiB",
            f"- Faltantes criticos: {len(critical)}",
            f"- Faltantes opcionales: {len(self.missing) - len(critical)}",
            "", "## Sesiones GUI seleccionadas", "",
        ]
        lines += [f"- {view}: `{path}`" for view, path in sorted(self.selected_sessions.items())]
        lines += ["", "## Faltantes", ""]
        lines += [f"- {status}: `{path}` ({category})" for status, path, category in self.missing] or ["- Ninguno"]
        report.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return report

    def run(self, refresh_only: bool = False) -> None:
        self.plan_all()
        if self.dry_run:
            report = self.dry_run_report()
            print(f"Dry-run guardado: {report}")
            print(f"Copias planificadas: {len({item.destination for item in self.planned})}")
            print(f"Faltantes: {len(self.missing)}")
            return
        if not refresh_only and (self.backup.exists() or self.evidence.exists()):
            raise FileExistsError(
                "Las carpetas de cierre ya existen. Use --refresh-only para actualizar manifiestos."
            )
        self.ensure_structure()
        if not refresh_only:
            self.copy_planned()
            self.generated_methodology()
            self.write_logs()
        self.write_manifests()
        print(f"Backup: {self.backup}")
        print(f"Evidencia: {self.evidence}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--timestamp", default=datetime.now().strftime("%Y%m%d_%H%M"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--refresh-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.project_root.exists():
        raise FileNotFoundError(args.project_root)
    builder = FreezeBuilder(args.project_root, args.timestamp, args.dry_run)
    builder.run(refresh_only=args.refresh_only)


if __name__ == "__main__":
    main()
