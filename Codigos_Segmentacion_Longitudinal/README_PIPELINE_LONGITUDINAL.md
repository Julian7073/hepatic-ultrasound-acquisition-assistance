# Pipeline longitudinal COCO + GLCM

Este bloque implementa el flujo reproducible acordado para la vista longitudinal.

## Entrada principal

```text
<PROJECT_ROOT>\Dataset_Roboflow_Longitudinal\V2_COCO
```

Contiene:

- `train`
- `valid`
- `test`
- `_annotations.coco.json` en cada split

## Salidas principales

```text
<PROJECT_ROOT>\outputs
```

Subcarpetas usadas:

- `outputs/masks`
- `outputs/qc_masks`
- `outputs/metrics`
- `outputs/reports`
- `outputs/figures`

## Orden de ejecucion

Desde PowerShell o terminal de Visual Studio:

```powershell
cd "<PROJECT_ROOT>\Codigos_Segmentacion_Longitudinal"

python 00_verificar_entorno.py
python 01_auditar_dataset_roboflow.py
python 02_generar_mascaras_coco.py
python 03_qc_visual_mascaras.py --samples-per-split 15
python 04_pipeline_glcm_longitudinal.py
python 05_definir_umbral_aceptabilidad.py
```

Scripts posteriores preparados, pero no prioritarios ahora:

```powershell
python 06_entrenar_segmentacion_local.py
python 07_evaluar_paciente_externo_p005.py
```

## Regla metodologica

- Roboflow se usa como herramienta de anotacion y baseline preliminar.
- El modelo entrenado en Roboflow no es el modelo reproducible final porque no se descargaron pesos.
- El insumo reproducible es el dataset exportado en formato COCO Segmentation.
- P005 queda reservado para validacion externa y no se usa para ajuste de umbrales ni entrenamiento.

## Umbral inicial generado

Los umbrales iniciales se calculan desde imagenes `clear` de `train` y `valid`.

La regla inicial es:

```text
has_la == 1
AND la_area_px >= min_la_area_px
AND la_std_intensity <= max_la_std_intensity
AND glcm_entropy <= max_glcm_entropy
```

Estos umbrales deben interpretarse como punto de partida, no como resultado clinico final.
