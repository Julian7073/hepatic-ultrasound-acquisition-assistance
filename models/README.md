# Final model artifacts

The three small DINOv2 classifier bundles are stored in `classifiers/`:

- `transversal__binary_dinov2.joblib`
- `oblicua__binary_dinov2.joblib`
- `hepatorrenal__binary_dinov2.joblib`

The three segmentation checkpoints are distributed as assets of release `v1.0.0-thesis` because each file exceeds GitHub's ordinary web-upload limit. Download and place them in `segmentation_checkpoints/`:

- `best_roi_model.pth`
- `best_higado_model.pth`
- `best_la_model.pth`

Do not rename the files. Verify every artifact against `model_sha256.csv` before inference.

DINOv2-Small itself is referenced by the official model identifier `facebook/dinov2-small`; it is not duplicated in this repository.
