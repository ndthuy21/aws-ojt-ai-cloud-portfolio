# Cloud-Ready ML Inference API Mini Lab

This mini lab exposes a small model artifact through an HTTP API. It uses only
Python standard library so it is easy to run on a clean Linux machine.

## Run Locally

```bash
python cloud_ready_ml_api/app.py --port 8000
```

Health check:

```bash
curl http://127.0.0.1:8000/health
```

Prediction:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features":[0.82,0.15,0.12,0.42]}'
```

## Why This Fits AWS OJT

- Shows a deployable service shape.
- Separates model artifact from application code.
- Can be hosted on EC2 Linux.
- Can move model artifacts to S3.
- Can send logs to CloudWatch.
- Keeps cost-control and Free Tier practice in mind.

