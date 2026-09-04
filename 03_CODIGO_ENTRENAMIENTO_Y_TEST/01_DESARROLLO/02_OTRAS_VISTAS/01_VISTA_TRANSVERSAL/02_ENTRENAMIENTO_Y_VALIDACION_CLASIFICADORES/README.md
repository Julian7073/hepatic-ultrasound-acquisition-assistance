# Training and validation: transversal

train_validate.py delegates to scripts/14_reproduce_development_only.py. It retains the archived variants, classifier factories, LOPO selection, tie-breaking, and threshold functions. P005 rows are removed before temporal construction. Selected models are fitted only on development clear/blurry anchors. No independent predictions are generated.

```powershell
python "03_CODIGO_ENTRENAMIENTO_Y_TEST/01_DESARROLLO/02_OTRAS_VISTAS/01_VISTA_TRANSVERSAL/02_ENTRENAMIENTO_Y_VALIDACION_CLASIFICADORES/train_validate.py" --artifact-root "PATH_TO_PRIVATE_THESIS_ARTIFACTS"
```

Existing embedding caches are required; their extraction remains in canonical scripts 03 and 07. New training is a reproduction, not a replacement for the archived thesis run. Outputs default to outputs/stage_reproduction/development.
