# Data access and expected layout

Ultrasound videos, extracted frames, COCO annotations, and radiologist labels are **not distributed in this public repository**. They are study data and may only be transferred under the applicable protocol and institutional authorization.

The code expects the following local layout when authorized data are available:

```text
Dataset/
Dataset_Frames_Processed/
Dataset_Roboflow_Longitudinal/
  ROI_COCO/
  Higado_COCO/
  LA_COCO/
```

The audited study organization distinguished development patients P001, P002, and P003 from the independently evaluated patient P005. P004 was excluded according to the documented acquisition-quality review. Do not alter this separation when reproducing the reported experiments.

The frozen aggregate result tables in `results/tables/` are sufficient to audit the numerical claims in the thesis without publishing individual ultrasound images.
