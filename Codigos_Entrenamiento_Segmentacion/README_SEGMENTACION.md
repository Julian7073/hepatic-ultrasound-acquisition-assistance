# Entrenamiento local de segmentacion longitudinal

Esta carpeta contiene la fase de entrenamiento local para comparar tres arquitecturas:

- U-Net
- DeepLabV3+
- SegFormer

Cada arquitectura se entrenara por separado para:

- ROI
- Higado
- LA

Roboflow se usa solo como fuente de anotaciones COCO. El entrenamiento formal es local.

## Orden recomendado

Primero auditar datasets separados:

```powershell
cd "<PROJECT_ROOT>"
python ".\Codigos_Entrenamiento_Segmentacion\00_auditar_coco_separados.py"
```

Instalar dependencias de entrenamiento:

```powershell
cd "<PROJECT_ROOT>\Codigos_Entrenamiento_Segmentacion"
python -m pip install -r requirements_segmentation.txt
```

Piloto recomendado:

```powershell
cd "<PROJECT_ROOT>"
python ".\Codigos_Entrenamiento_Segmentacion\02_entrenar_unet.py" --class_name ROI --epochs 5 --batch_size 2
```

## Salidas

Los resultados se guardan en:

```text
outputs/segmentation_training/
```

Subcarpetas principales:

- checkpoints
- final_models
- metrics
- figures
- overlays
- reports
- logs

## Nota metodologica

Los datasets exportados desde Roboflow se mantienen sin resize y sin augmentations. El resize y las augmentations leves se aplican localmente dentro del codigo de entrenamiento para asegurar una comparacion justa entre arquitecturas.

## Verificar DataLoader

Antes de entrenar, conviene revisar que imagen y mascara salgan con el mismo tamano:

```powershell
cd "<PROJECT_ROOT>"
python ".\Codigos_Entrenamiento_Segmentacion\01_preparar_dataloaders.py" --class_name ROI --split train --batch_size 2
```

## Entrenamiento piloto U-Net ROI

Usar pocas epocas primero para confirmar que todo corre:

```powershell
cd "<PROJECT_ROOT>"
python ".\Codigos_Entrenamiento_Segmentacion\02_entrenar_unet.py" --class_name ROI --epochs 5 --batch_size 2 --image_size 512
```

Si tu GPU tiene poca memoria, baja a:

```powershell
python ".\Codigos_Entrenamiento_Segmentacion\02_entrenar_unet.py" --class_name ROI --epochs 5 --batch_size 1 --image_size 384
```

## Evaluar modelos entrenados

```powershell
cd "<PROJECT_ROOT>"
python ".\Codigos_Entrenamiento_Segmentacion\05_evaluar_modelos.py"
python ".\Codigos_Entrenamiento_Segmentacion\06_comparar_arquitecturas.py"
python ".\Codigos_Entrenamiento_Segmentacion\07_generar_reporte_segmentacion.py"
```

## Entrenar las 9 combinaciones

Solo ejecutar despues de que el piloto U-Net ROI funcione:

```powershell
cd "<PROJECT_ROOT>"
.\Codigos_Entrenamiento_Segmentacion\train_all.bat
```

## Base de inferencia para GUI

Cuando existan los modelos finales en `outputs/segmentation_training/final_models`, se podra procesar un video:

```powershell
cd "<PROJECT_ROOT>"
python ".\Codigos_Entrenamiento_Segmentacion\08_inferencia_video_longitudinal_base.py" --video_path "ruta\al\video.mp4"
```
## Seleccion de checkpoint para LA

La clase LA esta muy desbalanceada: muchas imagenes no tienen lumen anecoico anotado. Por eso el Dice global puede premiar modelos que predicen mascara vacia.

El entrenador ahora usa `--checkpoint_metric auto` por defecto. En ROI e Higado selecciona por Dice global; en LA selecciona por `combined_la_score`, definido como:

```text
combined_la_score = positive_dice - empty_false_positive_rate
```

Para reentrenar LA de forma defendible:

```powershell
cd "<PROJECT_ROOT>"
python ".\Codigos_Entrenamiento_Segmentacion\03_entrenar_deeplabv3.py" --class_name LA --epochs 5 --batch_size 2 --image_size 512 --checkpoint_metric combined_la_score
```

## Reporte tecnico final de segmentacion longitudinal

Generar el documento tecnico con auditoria, referencias, fallos/correcciones, tablas, graficas y resumen:

```powershell
cd "<PROJECT_ROOT>"
python ".\Codigos_Entrenamiento_Segmentacion\09_generar_documento_tecnico_final.py"
```

Salidas principales:

```text
outputs\segmentation_training\reports\documento_tecnico_segmentacion_longitudinal.docx
outputs\segmentation_training\reports\documento_tecnico_segmentacion_longitudinal.md
```

## Inferencia longitudinal sobre video

Cuando quieras probar el pipeline con un video longitudinal, usa:

```powershell
cd "<PROJECT_ROOT>"
python ".\Codigos_Entrenamiento_Segmentacion\08_inferencia_video_longitudinal_base.py" --video_path "ruta\al\video.mp4" --frame_stride 5 --save_overlays
```

El script usa los modelos finales en:

```text
outputs\segmentation_training\final_models\best_roi_model.pth
outputs\segmentation_training\final_models\best_higado_model.pth
outputs\segmentation_training\final_models\best_la_model.pth
```

Genera un CSV por frame con areas, proporcion higado/ROI, GLCM, desviacion estandar de LA, decision y mensaje para usuario. Si se usa `--save_overlays`, tambien guarda imagenes PNG con las mascaras predichas.
