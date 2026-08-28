"""Prueba funcional corta de la rama DINO en la GUI unificada."""

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
    / "1. Transversal"
    / "1. Clear"
    / "20260501_135548_PACIENTE 003_202605020013-converted.mp4"
)
OUTPUT = ROOT / "outputs" / "unified_gui_sessions"


def main() -> None:
    before = {path.name for path in OUTPUT.glob("*")} if OUTPUT.exists() else set()
    app = AppTest.from_file(str(APP), default_timeout=360)
    app.run()
    app.selectbox[0].select_index(1)
    app.run()
    app.radio[0].set_value("Ruta local")
    app.run()
    app.number_input[3].set_value(8)
    app.text_input[0].input(str(VIDEO))
    app.button[0].click()
    app.run(timeout=360)

    if len(app.exception):
        messages = [str(item.value) for item in app.exception]
        raise AssertionError(messages)
    after_paths = [path for path in OUTPUT.glob("*") if path.name not in before]
    if not after_paths:
        raise AssertionError("La GUI no creo una carpeta de sesion.")
    latest = max(after_paths, key=lambda path: path.stat().st_mtime)
    results = latest / "frame_results.csv"
    summary = latest / "session_summary.csv"
    best_summary = latest / "best_frame_summary.csv"
    best_image = latest / "best_informative_frame.png"
    if not results.exists() or not summary.exists():
        raise AssertionError(f"Faltan salidas en {latest}")
    if not best_summary.exists() or not best_image.exists():
        raise AssertionError("No se guardo el mejor frame confirmado.")
    captured = list((latest / "captured_frames").glob("*.png"))
    if not captured:
        raise AssertionError("No se guardaron imagenes informativas confirmadas.")
    if len(app.metric) != 0:
        raise AssertionError("La pestana Analisis no debe mostrar metricas tecnicas.")
    if len(app.warning) or len(app.info):
        raise AssertionError("La retroalimentacion visible debe ser binaria.")

    print("PRUEBA GUI DINO OK")
    print(f"Sesion: {latest}")
    print(f"Resultados: {results}")
    print(f"Mejor frame: {best_image}")
    print(f"Metricas renderizadas: {len(app.metric)}")
    print(f"Imagenes informativas guardadas: {len(captured)}")


if __name__ == "__main__":
    main()
