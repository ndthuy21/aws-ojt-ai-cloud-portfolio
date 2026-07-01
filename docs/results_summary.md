# Results Summary

Latest local verification: 2026-07-01.

## Metrics

| Project | Dataset Type | Key Metrics |
|---|---|---|
| TACO Waste Dataset Preparation | Public TACO object-detection dataset | 1,500 images, 4,784 annotations, 4,241 YOLO boxes, 6 classes, validation passed |
| Air Quality Forecasting | Public UCI Air Quality dataset | 6,919 usable rows, MAE 1.1350, RMSE 1.3775 |
| Weather Traffic Volume Prediction | Public UCI Metro Interstate Traffic Volume dataset | 48,204 rows, MAE 893.9778, RMSE 1145.3156 |

## Interpretation

- The TACO project does not claim model accuracy; it proves public dataset
  preparation, YOLO conversion, split validation, and class-distribution checks.
- The forecasting projects use public tabular datasets and report non-zero
  regression errors that are suitable for honest technical discussion.
- These projects are meant to prove reproducible implementation and AWS readiness,
  not to claim production ML performance.
