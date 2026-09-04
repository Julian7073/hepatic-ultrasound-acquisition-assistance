# Code organization by experimental stage

This directory is the thesis-facing map of executable code. It separates development from independent validation and then separates the required stages by hepatic view.

## `01_DESARROLLO`

- `01_VISTA_LONGITUDINAL/01_ENTRENAMIENTO_Y_VALIDACION_MODELOS_IA`
- `01_VISTA_LONGITUDINAL/02_IDENTIFICACION_DE_UMBRALES`
- `01_VISTA_LONGITUDINAL/03_PIPELINE_PROCESAMIENTO_TEST_INTERNO`
- `02_OTRAS_VISTAS/<view>/01_IDENTIFICACION_DE_UMBRALES`
- `02_OTRAS_VISTAS/<view>/02_ENTRENAMIENTO_Y_VALIDACION_CLASIFICADORES`
- `02_OTRAS_VISTAS/<view>/03_PIPELINE_PROCESAMIENTO_TEST_INTERNO`

The three other-view branches use the same DINOv2-Small feature extractor but separate view-specific inputs, classifiers, thresholds, and outputs. The original combined folder is retained for backward compatibility with the frozen `v1.1.0-thesis` release; the stage-specific folders are the current entry points.

## `02_VALIDACION_INDEPENDIENTE`

- `01_VISTA_LONGITUDINAL/01_PIPELINE_PROCESAMIENTO_VIDEO_TEST_INDEPENDIENTE`
- `02_OTRAS_VISTAS/<view>/01_PIPELINE_PROCESAMIENTO_VIDEO_TEST_INDEPENDIENTE`

Independent DINOv2-Small replay loads saved classifiers and never calls `fit`. Its output joins the five source-frame identifiers in each window to the Kaggle `frames_manifest.csv`, preserves label provenance, and reports window and video metrics separately. The sequential `run_video.py` entry points remain distinct because GUI sliding-buffer updates are not the non-overlapping windows used for archived offline metrics.

No medical images, annotations, private paths, models, or generated outputs are stored under this hierarchy. Obtain authorized data from the private Kaggle dataset and checkpoints from the frozen GitHub release, then pass local paths explicitly. New reproduction outputs default to `outputs/stage_reproduction` and do not overwrite archived thesis evidence.
