# Results Summary

Latest local verification: 2026-07-01.

## Metrics

| Project | Dataset Type | Key Metrics |
|---|---|---|
| Image Waste Classifier | Generated sample images with feature noise | Accuracy 0.8889, Macro F1 0.8886 |
| Air Quality Forecasting | Generated sensor time series | MAE 3.3245, RMSE 4.1356 |
| Weather Traffic Risk | Generated weather/traffic sample | Accuracy 0.7262, Precision 0.8750, Recall 0.5122, F1 0.6462 |

## Interpretation

- The image classifier is intentionally simple and uses generated class patterns
  with feature noise, so the score validates the pipeline rather than real-world
  waste-recognition performance.
- The forecasting and risk-analysis projects are better examples for discussing
  evaluation tradeoffs because their errors are non-zero.
- These projects are meant to prove reproducible implementation and AWS readiness,
  not to claim production ML performance.
