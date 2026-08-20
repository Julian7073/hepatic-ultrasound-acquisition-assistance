"""Prueba estructural de la GUI unificada sin ejecutar inferencias largas."""

from pathlib import Path

from streamlit.testing.v1 import AppTest


APP = (
    Path(__file__).resolve().parents[1]
    / "gui_adquisicion_hepatica.py"
)


def assert_clean(app: AppTest, stage: str) -> None:
    if len(app.exception):
        messages = [str(item.value) for item in app.exception]
        raise AssertionError(f"{stage}: {messages}")


def main() -> None:
    app = AppTest.from_file(str(APP), default_timeout=120)
    app.run()
    assert_clean(app, "carga inicial")

    expected = ["Longitudinal", "Transversal", "Oblicua", "Hepatorrenal"]
    options = list(app.selectbox[0].options)
    if options != expected:
        raise AssertionError(f"Vistas inesperadas: {options}")

    rows = []
    for index, view in enumerate(expected):
        app.selectbox[0].select_index(index)
        app.run()
        assert_clean(app, f"vista {view}")
        rows.append({
            "view": view,
            "buttons": len(app.button),
            "number_inputs": len(app.number_input),
            "sliders": len(app.slider),
            "file_uploaders": len(app.get("file_uploader")),
            "text_inputs": len(app.text_input),
            "info_messages": len(app.info),
        })

    print("GUI Streamlit OK")
    print(f"Vistas: {options}")
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()