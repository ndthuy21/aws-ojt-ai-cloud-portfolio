# Air Quality Forecasting

This project uses the public UCI Air Quality dataset and builds a small
regression model to estimate benzene concentration `C6H6(GT)` from sensor,
weather, and lag features. It reports MAE and RMSE so the project can be
discussed with concrete evaluation metrics.

## Run

```bash
python projects/air_quality_forecasting/forecast.py
```

Outputs:

- `outputs/predictions.csv`
- `outputs/metrics.json`
- `outputs/model_weights.json`

## Evidence

- Source dataset: UCI Air Quality.
- Public source rows after target cleaning: 8,991.
- Usable rows after feature cleaning and lag creation: 6,919.
- Train/test split: 5,535 / 1,384 rows.
- Latest local result: MAE 1.1350, RMSE 1.3775.

## AWS Extension

- Store public CSV inputs and model artifacts in S3.
- Use Glue or scheduled scripts for ETL.
- Run model training/inference on EC2 or SageMaker.
- Send model logs and drift checks to CloudWatch.
