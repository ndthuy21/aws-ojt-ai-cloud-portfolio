# Weather Traffic Risk Analysis

This project generates a weather and traffic sample dataset, trains a simple
logistic classifier with NumPy, and reports accuracy, precision, recall, F1, and
confusion matrix.

## Run

```bash
python projects/weather_traffic_risk/analyze.py
```

Outputs:

- `outputs/sample_weather_accidents.csv`
- `outputs/predictions.csv`
- `outputs/metrics.json`
- `outputs/confusion_matrix.csv`

## AWS Extension

- Store daily CSV inputs in S3.
- Process scheduled batches on EC2 or AWS Batch.
- Send alert summaries to CloudWatch logs.
- Use IAM least-privilege access for data reads.

