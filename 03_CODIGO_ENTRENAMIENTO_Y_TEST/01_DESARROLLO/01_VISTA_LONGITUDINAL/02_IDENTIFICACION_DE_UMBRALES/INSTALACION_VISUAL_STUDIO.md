# Instalacion del entorno en Visual Studio

Esta guia prepara el entorno para ejecutar el pipeline longitudinal de la tesis.

## 1. Abrir la carpeta correcta

En Visual Studio o Visual Studio Code, abre:

```text
<PROJECT_ROOT>
```

La carpeta principal del pipeline nuevo es:

```text
Codigos_Segmentacion_Longitudinal
```

## 2. Crear un entorno virtual recomendado

Desde una terminal PowerShell:

```powershell
cd "<PROJECT_ROOT>\Codigos_Segmentacion_Longitudinal"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Si PowerShell bloquea la activacion del entorno virtual, ejecutar una sola vez:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Luego volver a activar:

```powershell
.\.venv\Scripts\Activate.ps1
```

## 3. Verificar librerias

```powershell
python 00_verificar_entorno.py
```

Debe aparecer `OK` en los paquetes principales.

## 4. Librerias incluidas

El archivo `requirements.txt` instala:

- `numpy`
- `pandas`
- `pillow`
- `opencv-python`
- `scikit-image`
- `matplotlib`
- `tqdm`
- `openpyxl`

## 5. Nota practica

El pipeline se escribira para funcionar con rutas configuradas en `config.py`.
No se deben mover ni modificar las imagenes originales.
