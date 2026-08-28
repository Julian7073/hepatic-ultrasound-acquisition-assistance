# Training, internal testing, and independent validation code

This directory separates the development evidence used to fit and select the models from the independent evaluation performed with Patient 4 (repository code P005). The separation mirrors the methodology and results chapters of the thesis.

## 01_DESARROLLO

- `01_VISTA_LONGITUDINAL/01_ENTRENAMIENTO_Y_VALIDACION_MODELOS_IA`: ROI, liver, and LA segmentation training, validation, transfer-learning comparison, and internal test.
- `01_VISTA_LONGITUDINAL/02_IDENTIFICACION_DE_UMBRALES`: derivation of the LA area, intensity, GLCM, and border-rule thresholds from development data.
- `01_VISTA_LONGITUDINAL/03_PIPELINE_PROCESAMIENTO_TEST_INTERNO`: post-processing, quantitative rules, temporal stabilization, internal benchmarks, and GUI integration.
- `02_OTRAS_VISTAS/<view>/01_EXTRACCION_Y_CLASIFICACION_TEST_INTERNO`: DINOv2-Small embedding extraction, five-embedding aggregation, leave-one-patient-out classifier comparison, and internal threshold calibration for the transverse, oblique, or hepatorenal view.

## 02_VALIDACION_INDEPENDIENTE

- `01_VISTA_LONGITUDINAL`: frozen-model evaluation scripts for Patient 4, including the longitudinal state and evidence files.
- `02_OTRAS_VISTAS/<view>`: frozen DINOv2-Small and view-specific classifier inference for Patient 4.

Patient 4 is not used to train weights, select an architecture or classifier, or derive thresholds. The radiologist label remains binary. Green, yellow, and red are operational interface states; yellow is an abstention/fine-adjustment state rather than a third clinical label.

The same source modules are retained in the original top-level folders for backward-compatible reproduction of the tagged thesis release. The organized copies in this directory make the development/independent-validation boundary explicit without changing the audited algorithms.
