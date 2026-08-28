"""Prueba funcional minima de la rama longitudinal en la GUI unificada."""

from pathlib import Path

from streamlit.testing.v1 import AppTest


ROOT = Path(__file__).resolve().parents[2]
APP = ROOT / "Codigos_Pipeline_Experimental_Segmentacion" / "gui_adquisicion_hepatica.py"
VIDEO = (
    ROOT
    / "Dataset"
    / "20260502_131713_PACIENTE 005"
    / "01052026"
    / "MP4"
    / "2. Longitudinal"
    / "1. Clear"
    / "20260501_135548_PACIENTE 003_202605020016-converted.mp4"
)
OUTPUT = ROOT / "outputs" / "unified_gui_sessions"


def main() -> None:
    before = {path.name for path in OUTPUT.glob("*")} if OUTPUT.exists() else set()
    app = AppTest.from_file(str(APP), default_timeout=420)
    app.run()
    app.radio[0].set_value("Ruta local")
    app.run()
    app.number_input[4].set_value(1)
    app.text_input[0].input(str(VIDEO))
    app.button[0].click()
    app.run(timeout=420)

    if len(app.exception):
        messages = [str(item.value) for item in app.exception]
        raise AssertionError(messages)
    after_paths = [path for path in OUTPUT.glob("*") if path.name not in before]
    if not after_paths:
        raise AssertionError("La GUI no creo una carpeta longitudinal.")
    latest = max(after_paths, key=lambda path: path.stat().st_mtime)
    if not (latest / "frame_results.csv").exists():
        raise AssertionError(f"Falta frame_results.csv en {latest}")
    if len(app.metric) != 0:
        raise AssertionError("La pestana Analisis no debe mostrar metricas tecnicas.")
    print("PRUEBA GUI LONGITUDINAL OK")
    print(f"Sesion: {latest}")
    print(f"Metricas renderizadas: {len(app.metric)}")


if __name__ == "__main__":
    main()
