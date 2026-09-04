# Archived and audited result tables

The original thesis tables are preserved. The files below close the final record-level audit without changing a trained model, threshold, prediction, or frozen artifact:

- `segmentation_training_record_audit.csv`: all nine pretrained segmentation runs, completed epochs, common settings, checkpoint presence, shared 101-image normalized test identity, and the maximum difference between per-image means and stored summaries.
- `longitudinal_sample_reconciliation.csv`: the 303 unique P005 longitudinal records and the clear/blurry/medium inclusion counts.
- `longitudinal_leave_one_gate_out.csv`: raw acceptance counts after removing each deterministic gate while retaining all others.
- `dino_label_prediction_audit.csv`: every independent DINOv2-Small window joined to five source-frame labels and their `expert_radiologist_review` provenance in the Kaggle manifest.
- `dino_window_prediction_audit.csv`: the 36 independent window probabilities, labels, and binary decisions reconstructed from saved embeddings and classifiers.
- `dino_metrics_by_evaluation_unit.csv`: window-level and video-level binary metrics reported separately.

These tables use coded participant/video identifiers and contain no ultrasound pixels. Clear and blurry are the positive and negative binary anchors; medium remains excluded from binary metrics. Window results must not be multiplied by five or presented as five independent predictions. Longitudinal frame metrics and DINOv2-Small window metrics are not pooled into a global value.

The corresponding read-only entry points are:

- `Codigos_Pipeline_Experimental_Segmentacion/scripts/13_audit_frozen_evidence.py`
- `Codigos_DINO_Experimental/scripts/15_reproduce_frozen_stage.py`

Both scripts write to a new `outputs/stage_reproduction` directory by default and never overwrite this archive.
