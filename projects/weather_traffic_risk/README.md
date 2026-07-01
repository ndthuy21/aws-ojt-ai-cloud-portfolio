# Weather-Based Traffic Volume Prediction

This project uses the public UCI Metro Interstate Traffic Volume dataset and
builds a small regression model to estimate hourly `traffic_volume` from weather,
holiday, and time features.

## Run

```bash
python projects/weather_traffic_risk/analyze.py
```

Outputs:

- `outputs/predictions.csv`
- `outputs/metrics.json`
- `outputs/model_weights.json`

## Evidence

- Source dataset: UCI Metro Interstate Traffic Volume.
- Source size: 48,204 hourly records.
- Train/test split: 38,563 / 9,641 rows.
- Latest local result: MAE 893.9778, RMSE 1145.3156.

## AWS Extension

- Store public CSV inputs and model artifacts in S3.
- Process scheduled batches on EC2 or AWS Batch.
- Send alert summaries to CloudWatch logs.
- Use IAM least-privilege access for data reads.
