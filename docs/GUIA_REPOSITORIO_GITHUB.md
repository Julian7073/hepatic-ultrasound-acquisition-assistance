# Guía de publicación y mantenimiento en GitHub

## 1. Configuración recomendada

- **Cuenta:** `Julian7073`
- **Nombre:** `hepatic-ultrasound-acquisition-assistance`
- **Visibilidad:** pública, para que los anexos de la tesis tengan enlaces verificables sin iniciar sesión.
- **Descripción:** `Research prototype for standardized hepatic-ultrasound acquisition assistance using segmentation, DINOv2 and view-specific classifiers.`
- **README:** usar el incluido en la raíz.
- **Licencia:** no seleccionar una licencia abierta mientras no exista autorización institucional. Sin licencia, el código conserva el derecho de autor por defecto.
- **Rama principal:** `main`.
- **Topics:** `ultrasound`, `computer-vision`, `medical-imaging`, `image-segmentation`, `dinov2`, `streamlit`, `thesis`.

## 2. Contenido que sí debe publicarse

1. Los cuatro directorios `Codigos_*` del código auditado.
2. `requirements.txt`, `.gitignore`, `README.md` y `CITATION.cff`.
3. Los clasificadores `.joblib`, manifiestos y hashes de `models/`.
4. Las tablas CSV agregadas y figuras sin imágenes ecográficas individuales de `results/`.
5. El documento técnico en `docs/thesis.pdf`, si el autor acepta que el documento —incluidos sus datos de portada— sea público.
6. Los tres `.pth` como activos de la versión `v1.0.0-thesis`, no como archivos ordinarios del repositorio.

## 3. Contenido que no debe publicarse

- Videos ecográficos, *frames*, máscaras o anotaciones individuales.
- Copias completas de `Dataset/`, `Dataset_Frames_Processed/` o `Dataset_Roboflow_Longitudinal/`.
- Transcripciones de reuniones, comentarios privados de tutoría o versiones intermedias de la tesis.
- Entornos `.venv`, cachés, `__pycache__`, resultados temporales y sesiones locales.
- Rutas personales, credenciales, tokens o archivos que identifiquen directamente a participantes.

## 4. Primera publicación con Git

Desde la carpeta preparada:

```powershell
git init
git branch -M main
git add .
git commit -m "Publish audited thesis implementation"
git remote add origin https://github.com/Julian7073/hepatic-ultrasound-acquisition-assistance.git
git push -u origin main
```

La carga mediante la web de GitHub también es válida para los archivos ordinarios, pero los tres pesos `.pth` deben añadirse a una **Release** porque superan el límite de carga web ordinaria.

## 5. Publicación de modelos

Crear la versión `v1.0.0-thesis` con el título `Audited thesis models and implementation` y adjuntar:

- `best_roi_model.pth`
- `best_higado_model.pth`
- `best_la_model.pth`

Después de la descarga, verificar cada SHA-256 con `models/model_sha256.csv`. Los clasificadores DINOv2 pequeños permanecen versionados en `models/classifiers/`.

## 6. Enlaces para los anexos

Una vez publicado, usar enlaces permanentes de la etiqueta `v1.0.0-thesis` cuando sea posible:

- Implementación longitudinal: `https://github.com/Julian7073/hepatic-ultrasound-acquisition-assistance/tree/v1.0.0-thesis/Codigos_Pipeline_Experimental_Segmentacion`
- Derivación de umbrales: `https://github.com/Julian7073/hepatic-ultrasound-acquisition-assistance/tree/v1.0.0-thesis/Codigos_Segmentacion_Longitudinal`
- Entrenamiento de segmentación: `https://github.com/Julian7073/hepatic-ultrasound-acquisition-assistance/tree/v1.0.0-thesis/Codigos_Entrenamiento_Segmentacion`
- DINOv2 y clasificadores: `https://github.com/Julian7073/hepatic-ultrasound-acquisition-assistance/tree/v1.0.0-thesis/Codigos_DINO_Experimental`
- GUI: `https://github.com/Julian7073/hepatic-ultrasound-acquisition-assistance/blob/v1.0.0-thesis/Codigos_Pipeline_Experimental_Segmentacion/gui_adquisicion_hepatica.py`
- Resultados: `https://github.com/Julian7073/hepatic-ultrasound-acquisition-assistance/tree/v1.0.0-thesis/results`
- Modelos y hashes: `https://github.com/Julian7073/hepatic-ultrasound-acquisition-assistance/releases/tag/v1.0.0-thesis`
- Política de datos: `https://github.com/Julian7073/hepatic-ultrasound-acquisition-assistance/blob/v1.0.0-thesis/data/README.md`

No usar enlaces a `main` en la versión entregada de la tesis, porque pueden cambiar después. La etiqueta conserva una referencia estable a la implementación auditada.

## 7. Cambios posteriores

Para cada corrección posterior:

```powershell
git add .
git commit -m "Describe the correction"
git push
```

No sustituir la etiqueta `v1.0.0-thesis`. Si se publica una revisión material, crear `v1.0.1-thesis` y mantener la versión anterior para trazabilidad.
