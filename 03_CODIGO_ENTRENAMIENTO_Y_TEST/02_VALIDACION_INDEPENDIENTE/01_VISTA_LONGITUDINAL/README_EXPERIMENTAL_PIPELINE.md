# Pipeline experimental de segmentacion longitudinal

Fase aislada y reproducible para ROI, Higado y lumen anecoico (LA). No modifica el pipeline preliminar ni sus resultados.

## Flujo

1. Auditar COCO y posible leakage por paciente/video.
2. Ejecutar entrenamiento, validacion y test desde un unico script.
3. Seleccionar checkpoint por Dice en ROI/Higado y combined_la_score en LA.
4. Medir inferencia con warmup, batch 1 y sincronizacion CUDA.
5. Guardar config, pesos, CSV, curvas, overlays y Markdown por experimento.

## Auditoria

    cd "<PROJECT_ROOT>"
    python ".\Codigos_Pipeline_Experimental_Segmentacion\scripts\00_audit_dataset.py"

## Smoke test oficial

    python ".\Codigos_Pipeline_Experimental_Segmentacion\run_segmentation_experiment.py" --class_name ROI --architecture unet --epochs 2 --batch_size 2 --image_size 128 --resize_mode full_resize --augmentation none --pretrained false --experiment_name smoke_unet_roi_2ep --run_test true --run_benchmark true --save_overlays

Esta corrida es una prueba funcional, no concluyente.

## Modos de resize

- full_resize: redimensiona toda la imagen al cuadrado indicado.
- roi_crop_resize: recorta la caja de la mascara ROI emparejada por nombre y luego redimensiona.
- original_or_padding: conserva proporcion con letterbox/padding al cuadrado indicado.

## Splits

El valor por defecto es coco para reproducir los splits exportados. group_video y group_patient reasignan grupos completos. Con solo tres pacientes, group_patient tiene alta varianza y debe reportarse como limitacion.

## Transferencia

U-Net y DeepLabV3+ usan ResNet-34. pretrained=true solicita pesos ImageNet y ajusta toda la red. SegFormer usa MiT-B0; pretrained=true solicita nvidia/mit-b0. La primera ejecucion preentrenada puede requerir Internet.

## Entrenamientos largos

Los wrappers 02 a 06 solo muestran comandos por defecto. Requieren --execute para comenzar entrenamientos.


## Recomendacion tras la auditoria

Los entrenamientos comparativos deben usar --split_strategy group_video para que un mismo video no aparezca en varios splits. Con seed 42, LA conserva positivos en train, valid y test. Este split sigue siendo interno y no sustituye la validacion externa con P005.


## Muestreo para LA

- natural: conserva la proporcion original de positivos y vacios.
- balanced_la: usa WeightedRandomSampler para aproximar 50% positivos y 50% vacios en cada epoca de train.
- Valid y test nunca se balancean.
- El balanceo no copia ni modifica imagenes originales.


## Augmentation selectiva para LA

- positive_x4 conserva una copia de cada imagen vacia y crea cuatro variantes virtuales de cada imagen positiva.
- Solo se aplica en train; valid y test permanecen intactos.
- Debe combinarse con sampling_strategy natural para evitar doble rebalanceo.
- No se crean ni modifican archivos de imagen.

Verificacion sin entrenamiento:

    python ".\Codigos_Pipeline_Experimental_Segmentacion\scripts\01_check_positive_x4.py"

Entrenamiento propuesto:

    python ".\Codigos_Pipeline_Experimental_Segmentacion\run_segmentation_experiment.py" --class_name LA --architecture unet --epochs 50 --batch_size 2 --image_size 512 --resize_mode full_resize --augmentation positive_x4 --sampling_strategy natural --pretrained false --checkpoint_metric combined_la_score --split_strategy group_video --early_stopping_patience 15 --checkpoint_min_delta 0.0001 --experiment_name unet_la_positive_x4_50ep_group_video_natural --run_test true --run_benchmark true --save_overlays


## Evaluacion externa P005

P005 no tiene mascaras ground truth. La ejecucion genera predicciones, overlays, metricas de textura y candidatos de revision, pero no Dice/IoU externos.

    python ".\Codigos_Pipeline_Experimental_Segmentacion\scripts\10_evaluate_external_p005.py" --overlays_per_quality 5 --save_overlays
    python ".\Codigos_Pipeline_Experimental_Segmentacion\scripts\11_audit_p005_identity.py"
    python ".\Codigos_Pipeline_Experimental_Segmentacion\scripts\12_analyze_external_p005.py"

## Inferencia longitudinal de video

    python ".\Codigos_Pipeline_Experimental_Segmentacion\run_longitudinal_video_inference.py" --video_path "RUTA_AL_VIDEO.mp4" --frame_stride 1 --warmup 3 --save_overlay_every 30

El pipeline devuelve una fila CSV por frame con mascaras, areas, GLCM, decision y mensaje. Los mensajes son experimentales y no constituyen diagnostico medico.


## GUI longitudinal

La GUI procesa un video local, estabiliza los mensajes en una ventana temporal y guarda un frame solo despues de varias decisiones capture consecutivas.

    cd "<PROJECT_ROOT>"
    python -m streamlit run ".\Codigos_Pipeline_Experimental_Segmentacion\gui_longitudinal.py" --server.port 8501

Tambien puede iniciarse con:

    .\Codigos_Pipeline_Experimental_Segmentacion\run_gui_longitudinal.bat

La configuracion inicial recomendada es frame_stride=3, ventana temporal=5, tres frames aceptables consecutivos y cooldown=10. Cada sesion se guarda en outputs/experimental_segmentation_pipeline/gui_sessions.

## GUI unificada de adquisicion hepatica

La GUI nueva conserva la segmentacion longitudinal e integra los modelos DINOv2
para transversal, oblicua y hepatorrenal.

Inicio desde Visual Studio o PowerShell:

    cd "<PROJECT_ROOT>"
    .Codigos_Pipeline_Experimental_Segmentacionun_gui_hepatica.bat

Direccion local:

    http://127.0.0.1:8501

Funciones:

- Cargar un video o usar una ruta local real.
- Elegir Longitudinal, Transversal, Oblicua o Hepatorrenal.
- Mostrar el frame original y la region analizada.
- Mostrar el campo ecografico limpio, sin texto ni encabezado, en las vistas DINO.
- Permitir recorrer todos los frames DINO evaluados mediante un selector persistente.
- Aplicar consenso temporal antes de guardar una captura.
- Seleccionar al finalizar el frame confirmado con mayor probabilidad informativa.
- Guardar resultados por frame, resumen de sesion y capturas aceptadas.
- Descargar el CSV desde la propia interfaz.

Salidas:

    outputs/unified_gui_sessions/<fecha_vista>/

La GUI longitudinal anterior sigue disponible mediante run_gui_longitudinal.bat.
Los mensajes son asistencia experimental y no constituyen diagnostico medico.
### Seleccion del mejor frame DINO

El ultimo frame mostrado durante el recorrido no se toma automaticamente como
resultado. La GUI compara las probabilidades de los frames que completaron la
confirmacion temporal y guarda el mejor como `best_informative_frame.png`. Si
ninguno fue confirmado, conserva el mejor candidato como
`best_candidate_frame.png` y lo identifica como no confirmado.

La seleccion se registra en `best_frame_summary.csv` y en las columnas
`is_best_frame`, `best_frame_selection` y `best_frame_path` de
`frame_results.csv`.