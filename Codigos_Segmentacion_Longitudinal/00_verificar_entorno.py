r"""
Verifica si el entorno de Python tiene las librerias necesarias.

Ejecutar desde:

    <PROJECT_ROOT>\Codigos_Segmentacion_Longitudinal

Comando:

    python 00_verificar_entorno.py
"""

from __future__ import annotations

import importlib
import sys


REQUIRED_PACKAGES = {
    "numpy": "numpy",
    "pandas": "pandas",
    "PIL": "pillow",
}

RECOMMENDED_PACKAGES = {
    "cv2": "opencv-python",
    "skimage": "scikit-image",
    "matplotlib": "matplotlib",
    "tqdm": "tqdm",
    "openpyxl": "openpyxl",
}


def check_package(import_name: str) -> bool:
    """Devuelve True si el paquete puede importarse."""
    try:
        importlib.import_module(import_name)
        return True
    except Exception:
        return False


def main() -> None:
    """Imprime estado del entorno y comando de instalacion si falta algo."""
    print("Python usado:")
    print(sys.executable)
    print()

    missing_required = []
    missing_recommended = []

    print("Paquetes obligatorios:")
    for import_name, pip_name in REQUIRED_PACKAGES.items():
        ok = check_package(import_name)
        print(f"  {pip_name}: {'OK' if ok else 'FALTA'}")
        if not ok:
            missing_required.append(pip_name)

    print("\nPaquetes recomendados para el pipeline completo:")
    for import_name, pip_name in RECOMMENDED_PACKAGES.items():
        ok = check_package(import_name)
        print(f"  {pip_name}: {'OK' if ok else 'FALTA'}")
        if not ok:
            missing_recommended.append(pip_name)

    missing = missing_required + missing_recommended
    if missing:
        print("\nPara instalar lo faltante desde esta carpeta:")
        print("  python -m pip install -r requirements.txt")
        print("\nO, si quieres instalar solo lo faltante:")
        print("  python -m pip install " + " ".join(missing))
    else:
        print("\nEntorno listo para ejecutar el pipeline longitudinal completo.")


if __name__ == "__main__":
    main()

