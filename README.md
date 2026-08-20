# Hepatic ultrasound acquisition assistance

Research prototype for assisting users with limited ultrasound experience in the acquisition of four hepatic views: *longitudinal*, *transverse*, *oblique*, and *hepatorenal*. The system evaluates whether a video frame satisfies the acquisition criteria defined for the study protocol. It does **not** diagnose disease, determine normality, or replace specialist review.

## Implemented strategies

1. **Longitudinal view.** DeepLabV3+ segments the ultrasound region of interest (ROI) and liver; U-Net segments the anechoic lumen (LA). The deterministic decision uses the implemented area, intensity-dispersion, GLCM-texture, and border-evidence rules.
2. **Other three views.** DINOv2-Small extracts 384-feature frame embeddings. Five consecutive embeddings are summarized by their mean and population standard deviation (768 features) and classified with Random Forest for the transverse view, logistic regression for the oblique view, and distance-weighted 7-NN for the hepatorenal view.

The auxiliary interface states use 0.35 and 0.65 probability limits, while independent binary metrics use a 0.50 decision threshold. These operational limits and the temporal confirmation logic are documented in the source and frozen result tables.

## Repository map

- [`Codigos_Pipeline_Experimental_Segmentacion/`](Codigos_Pipeline_Experimental_Segmentacion/): integrated longitudinal pipeline, final decision rule, evaluation scripts, and Streamlit GUI.
- [`Codigos_Segmentacion_Longitudinal/`](Codigos_Segmentacion_Longitudinal/): derivation and audit of longitudinal image features and thresholds.
- [`Codigos_Entrenamiento_Segmentacion/`](Codigos_Entrenamiento_Segmentacion/): segmentation architecture comparison and training scripts.
- [`Codigos_DINO_Experimental/`](Codigos_DINO_Experimental/): DINOv2 extraction, classifier selection, temporal aggregation, and independent evaluation.
- [`models/`](models/): three small view-specific classifier bundles, manifests, hashes, and instructions for the segmentation checkpoints distributed with the tagged release.
- [`results/`](results/): frozen aggregate CSV tables and non-identifying result figures.
- [`data/`](data/): expected dataset layout and access restrictions. No ultrasound video, frame, or annotation is published.
- [`docs/thesis.pdf`](docs/thesis.pdf): final technical document associated with this repository snapshot.

## Installation

Python 3.11 or 3.12 is recommended. CUDA is optional.

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

The GUI accepts a previously recorded MP4, AVI, MOV, or MKV file. Select one of the four views, process the sequence, review the binary acquisition guidance, and download confirmed informative images when available. New outputs are written under `outputs/`.

If the project is stored in a nonstandard layout, set `THESIS_PROJECT_ROOT` to the repository root before running a script.

## Reproduction commands

```powershell
# Independent longitudinal evaluation (requires authorized local data)
python .\Codigos_Pipeline_Experimental_Segmentacion\evaluate_p005_longitudinal_final.py --frame_stride 3 --save_overlays --save_csv --decision_config .\Codigos_Pipeline_Experimental_Segmentacion\configs\longitudinal_decision_config.json

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
