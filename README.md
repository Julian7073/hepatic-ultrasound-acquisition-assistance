# Hepatic ultrasound acquisition assistance

Research prototype for assisting users with limited ultrasound experience in the acquisition of four hepatic views: *longitudinal*, *transverse*, *oblique*, and *hepatorenal*. The system evaluates whether a video frame satisfies the acquisition criteria defined for the study protocol. It does **not** diagnose disease, determine normality, or replace specialist review.

## Implemented strategies

1. **Longitudinal view.** DeepLabV3+ segments the ultrasound region of interest (ROI) and liver; U-Net segments the anechoic lumen (LA). The deterministic decision uses the implemented area, intensity-dispersion, GLCM-texture, and border-evidence rules.
2. **Other three views.** DINOv2-Small extracts 384-feature frame embeddings. Five consecutive embeddings are summarized by their mean and population standard deviation (768 features) and classified with Random Forest for the transverse view, logistic regression for the oblique view, and distance-weighted 7-NN for the hepatorenal view.

The interface uses green, yellow, and red operational states, while the expert label and independent metrics remain binary. In the longitudinal branch, yellow identifies adequate ROI/liver evidence without sufficient LA-dependent evidence. In the DINOv2-Small branches, the red/yellow/green actions use the 0.35 and 0.65 limits, whereas independent binary metrics use a separate 0.50 decision threshold. These limits and the temporal confirmation logic are documented in the source and frozen result tables.

The origin, operational purpose, available verification, and limitation of every data-derived or fixed engineering decision are consolidated in [`results/tables/heuristic_design_decisions_audit.csv`](results/tables/heuristic_design_decisions_audit.csv). The development-cohort check of the deployed DINOv2-Small abstention interval is reported separately in [`results/tables/operational_threshold_oof_audit.csv`](results/tables/operational_threshold_oof_audit.csv); it must not be interpreted as independent clinical calibration.

The frozen longitudinal rule was also subjected to a post-hoc diagnostic audit on the independent participant. The sigmoid mask cutoff was swept from 0.10 to 0.90 without selecting a new value from the independent data. No setting recovered an LA prediction in any of the 101 expert-labeled informative frames, so sensitivity remained 0.0%, specificity 100.0%, and balanced accuracy 50.0%. Removing the LA-area requirement alone admitted seven medium-quality frames but no informative or blurry frame; removing all LA-dependent conditions admitted all 303 frames, producing 100.0% sensitivity and 0.0% specificity. These results identify failed LA transfer plus the mandatory LA gate as the mechanism of collapse and show why changing a downstream threshold cannot be presented as a valid correction. The frame-level records, threshold summary, gate analysis, and frozen-rule confusion matrix are available in [`results/tables/longitudinal_mask_threshold_sensitivity_frames.csv`](results/tables/longitudinal_mask_threshold_sensitivity_frames.csv), [`results/tables/longitudinal_mask_threshold_sensitivity_summary.csv`](results/tables/longitudinal_mask_threshold_sensitivity_summary.csv), [`results/tables/longitudinal_gate_ablation_by_mask_threshold.csv`](results/tables/longitudinal_gate_ablation_by_mask_threshold.csv), and [`results/tables/longitudinal_independent_confusion.csv`](results/tables/longitudinal_independent_confusion.csv).

## Repository map

- [`Codigos_Pipeline_Experimental_Segmentacion/`](Codigos_Pipeline_Experimental_Segmentacion/): integrated longitudinal pipeline, final decision rule, evaluation scripts, and Streamlit GUI.
- [`Codigos_Segmentacion_Longitudinal/`](Codigos_Segmentacion_Longitudinal/): derivation and audit of longitudinal image features and thresholds.
- [`Codigos_Entrenamiento_Segmentacion/`](Codigos_Entrenamiento_Segmentacion/): segmentation architecture comparison and training scripts.
- [`Codigos_DINO_Experimental/`](Codigos_DINO_Experimental/): DINOv2 extraction, classifier selection, temporal aggregation, and independent evaluation.
- [`03_CODIGO_ENTRENAMIENTO_Y_TEST/`](03_CODIGO_ENTRENAMIENTO_Y_TEST/): thesis-oriented organization that separates development/internal testing from independent Patient 4 validation and then separates every hepatic view.
- [`models/`](models/): three small view-specific classifier bundles, manifests, hashes, and instructions for the segmentation checkpoints distributed with the tagged release.
- [`results/`](results/): frozen aggregate CSV tables and non-identifying result figures.
- [`data/`](data/): expected dataset layout and access restrictions. No ultrasound video, frame, or annotation is published.
- [`docs/thesis.pdf`](docs/thesis.pdf): final technical document associated with this repository snapshot.

## Installation

The audited thesis environment used Python 3.13.2, PyTorch 2.12.1, and CUDA 12.6. CUDA is optional for execution, although processing rates will differ on CPU-only systems.

```powershell
git clone https://github.com/Julian7073/hepatic-ultrasound-acquisition-assistance.git
cd hepatic-ultrasound-acquisition-assistance
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Download the three segmentation checkpoints from the GitHub release and place them in `models/segmentation_checkpoints/` as described in [`models/README.md`](models/README.md). DINOv2-Small is loaded through the official Hugging Face identifier `facebook/dinov2-small` when it is not available in the local cache.

## Run the interface

```powershell
python -m streamlit run .\Codigos_Pipeline_Experimental_Segmentacion\gui_adquisicion_hepatica.py
```

The GUI accepts a previously recorded MP4, AVI, MOV, or MKV file. Select one of the four views, process the sequence, follow the green/yellow/red acquisition guidance, and download confirmed informative images when available. Yellow requests a fine adjustment and is not a third expert-defined clinical class. New outputs are written under `outputs/`.

If the project is stored in a nonstandard layout, set `THESIS_PROJECT_ROOT` to the repository root before running a script.

## Reproduction commands

```powershell
# Independent longitudinal evaluation (requires authorized local data)
python .\Codigos_Pipeline_Experimental_Segmentacion\evaluate_p005_longitudinal_final.py --frame_stride 3 --save_overlays --save_csv --decision_config .\Codigos_Pipeline_Experimental_Segmentacion\configs\longitudinal_decision_config.json

# Post-hoc diagnostic sweep; does not select a replacement threshold
python .\results\analysis\audit_longitudinal_independent_sensitivity.py

# DINOv2 inference example (requires a local video)
python .\Codigos_DINO_Experimental\scripts\10_run_binary_video_inference.py --video .\data\example_video.mp4 --view transversal
```

Training and full evaluation require the non-public COCO annotations and prepared frame dataset. The frozen aggregate outputs needed to audit the reported results are provided in `results/tables/`.

## Reproducibility and limitations

- The final software was tested on Windows 11; CUDA acceleration was optional.
- The study used a small single-environment sample and is not a multicenter or clinical validation.
- The independent evaluation must remain separate from internal model-selection results.
- Timing depends on the hardware and processing stride; the repository does not claim guaranteed real-time performance on arbitrary systems.
- Dataset access remains subject to the study protocol and institutional authorization.

## Citation

Use [`CITATION.cff`](CITATION.cff) when citing the software and thesis. No open-source license is granted by this repository; reuse requires permission from the author and institution.
