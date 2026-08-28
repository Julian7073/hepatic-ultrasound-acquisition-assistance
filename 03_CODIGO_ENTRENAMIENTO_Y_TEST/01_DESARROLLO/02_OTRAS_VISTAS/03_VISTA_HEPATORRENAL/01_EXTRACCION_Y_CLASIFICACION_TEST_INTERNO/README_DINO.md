# Pipeline experimental DINOv2

Fase separada para las vistas transversal, oblicua y hepatorrenal. P001-P003 se usan para desarrollo y P005 permanece como prueba externa final.

## Auditoria obligatoria

```powershell
cd "<PROJECT_ROOT>"
python ".\Codigos_DINO_Experimental\scripts\00_audit_dino_dataset.py"
```

La auditoria genera un índice maestro, conteos, videos fuente, duplicados exactos y redundancia temporal. No descarga DINOv2 ni entrena modelos.

## Diseño previsto

1. Extraer embeddings congelados con `facebook/dinov2-small`.
2. Evaluar cada vista por separado.
3. Comparar Logistic Regression, SVM, Random Forest y k-NN.
4. Seleccionar hiperparámetros solo con P001-P003 mediante leave-one-patient-out.
5. Evaluar P005 una sola vez al final.

Los frames del mismo video nunca deben cruzar particiones.

## Smoke test DINOv2

```powershell
python ".\Codigos_DINO_Experimental\scripts\01_smoke_test_dinov2.py" --max_images 18 --batch_size 6
```

La primera ejecución descarga `facebook/dinov2-small`. El resultado esperado es una matriz `18 x 384`; es una prueba funcional, no concluyente.

## Índice para embeddings

```powershell
python ".\Codigos_DINO_Experimental\scripts\02_prepare_embedding_index.py" --stride 5
```

El stride se reinicia dentro de cada video. P005 permanece separado como `external_test`.
## Extracción definitiva y clasificación

```powershell
python ".\Codigos_DINO_Experimental\scripts\03_extract_dinov2_embeddings.py" --stride 5 --batch_size 16
python ".\Codigos_DINO_Experimental\scripts\04_train_evaluate_classifiers.py" --stride 5 --seed 42
python ".\Codigos_DINO_Experimental\scripts\05_generate_dino_technical_report.py"
```

La selección usa únicamente P001-P003 mediante leave-one-patient-out. P005 se evalúa después de fijar el clasificador ganador de cada vista.
## Mejora binaria, recorte y contexto temporal

Esta fase conserva intacto el baseline de tres clases. El objetivo operativo se
reformula como clear frente a blurry; medium no se usa para entrenar el
clasificador y se analiza como zona de incertidumbre. La seleccion sigue usando
solo P001-P003 mediante leave-one-patient-out. P005 se evalua despues de fijar
toda la configuracion.

    cd "<PROJECT_ROOT>"
    python ".\Codigos_DINO_Experimental\scripts\06_audit_fan_crop.py"
    python ".\Codigos_DINO_Experimental\scripts\07_extract_binary_embeddings.py" --backbone small --preprocessing fan_crop --stride 5 --batch_size 16
    python ".\Codigos_DINO_Experimental\scripts\07_extract_binary_embeddings.py" --backbone base --preprocessing fan_crop --stride 5 --batch_size 8
    python ".\Codigos_DINO_Experimental\scripts\08_run_binary_temporal_experiment.py" --stride 5 --seed 42 --minimum_action_precision 0.90
    python ".\Codigos_DINO_Experimental\scripts\09_generate_binary_improvement_report.py"
    python ".\Codigos_DINO_Experimental\scripts\11_benchmark_binary_dino.py

La comparacion incluye:

- DINOv2-Small con frame completo.
- DINOv2-Small con el campo ecografico recortado.
- DINOv2-Base con el mismo recorte.
- Prediccion por frame y por ventanas de cinco embeddings.
- Logistic Regression, SVM RBF, Random Forest y k-NN.
- Umbrales conservadores para capture, adjust y doubtful.

Los modelos finales quedan en:

    outputs/dino_experimental/binary_improvement/models/

### Inferencia sobre un video nuevo

    python ".\Codigos_DINO_Experimental\scripts\10_run_binary_video_inference.py" --video "<PROJECT_ROOT>\data\example_video.mp4" --view transversal

El CSV se guarda por defecto en:

    outputs/dino_experimental/binary_improvement/video_inference/

Clear y blurry siguen siendo etiquetas nominales de adquisicion. Una salida
capture significa potencialmente informativa bajo este experimento, no una
certificacion clinica independiente.
    python ".\Codigos_DINO_Experimental\scripts\12_summarize_p005_transversal_inference.py"

## Integracion en GUI

Los modelos DINOv2 seleccionados se integran en:

    Codigos_Pipeline_Experimental_Segmentacion/gui_adquisicion_hepatica.py

Inicio:

    .Codigos_Pipeline_Experimental_Segmentacionun_gui_hepatica.bat

La GUI usa stride 5, ventana de cinco embeddings y tres decisiones capture
consecutivas antes de guardar un frame.
