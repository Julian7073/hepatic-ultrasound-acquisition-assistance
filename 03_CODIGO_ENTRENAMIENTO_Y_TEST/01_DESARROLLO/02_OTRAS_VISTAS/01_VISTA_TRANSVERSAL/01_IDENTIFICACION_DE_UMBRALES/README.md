# 01_IDENTIFICACION_DE_UMBRALES: transversal

This stage reads development-only out-of-fold records. It does not fit on or evaluate P005.

Run from the repository root:

```powershell
python "03_CODIGO_ENTRENAMIENTO_Y_TEST/01_DESARROLLO/02_OTRAS_VISTAS/01_VISTA_TRANSVERSAL/01_IDENTIFICACION_DE_UMBRALES/reproduce.py" --artifact-root "PATH_TO_PRIVATE_THESIS_ARTIFACTS"
```

Recomputes the action-limit procedure and separately evaluates the deployed fallback limits. New results are written to outputs/stage_reproduction, never to the frozen result tables.
