# Air Quality Forecasting

This project generates a reproducible sensor-like time-series dataset and builds
a small regression model for PM2.5 forecasting. It reports MAE and RMSE so the
project can be discussed with concrete evaluation metrics.

## Run

```bash
python projects/air_quality_forecasting/forecast.py
```

Outputs:

- `outputs/sample_sensor_readings.csv`
- `outputs/predictions.csv`
- `outputs/metrics.json`

## AWS Extension

- Store sensor readings in S3.
- Use Glue or scheduled scripts for ETL.
- Run model training/inference on EC2 or SageMaker.
- Send model logs and drift checks to CloudWatch.

