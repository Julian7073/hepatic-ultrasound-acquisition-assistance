# 03_PIPELINE_PROCESAMIENTO_TEST_INTERNO: transversal

This stage reads development-only out-of-fold records. It does not fit on or evaluate P005.

Run from the repository root:

```powershell
python "03_CODIGO_ENTRENAMIENTO_Y_TEST/01_DESARROLLO/02_OTRAS_VISTAS/01_VISTA_TRANSVERSAL/03_PIPELINE_PROCESAMIENTO_TEST_INTERNO/reproduce.py" --artifact-root "PATH_TO_PRIVATE_THESIS_ARTIFACTS"
```

Recalculates held-out-patient internal video metrics; medium is excluded from binary metrics. New results are written to outputs/stage_reproduction, never to the frozen result tables.
