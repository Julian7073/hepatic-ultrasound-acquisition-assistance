# Independent recorded-video test: transversal

reproduce_evaluation.py replays frozen window predictions, joins each of the five input frames to the Kaggle label manifest, and reports window and video metrics separately. It never calls fit. Use --artifact-root for private local artifacts and --manifest for a downloaded frames_manifest.csv if needed.

run_video.py delegates to the archived sequential-video predictor (use --help for video and model paths). Its sliding buffer and warm-up differ from non-overlapping offline windows; its GUI updates are not interchangeable with offline evaluation samples.

```powershell
python "03_CODIGO_ENTRENAMIENTO_Y_TEST/02_VALIDACION_INDEPENDIENTE/02_OTRAS_VISTAS/01_VISTA_TRANSVERSAL/01_PIPELINE_PROCESAMIENTO_VIDEO_TEST_INDEPENDIENTE/reproduce_evaluation.py" --artifact-root "PATH_TO_PRIVATE_THESIS_ARTIFACTS"
```
