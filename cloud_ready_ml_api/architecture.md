# Cloud-Ready ML Inference API

## Purpose

This mini lab demonstrates how a small ML-style prediction service can be moved
from local development to an AWS learning environment.

## Local Components

- `app.py`: lightweight HTTP API using Python standard library.
- `model.json`: small model artifact loaded at runtime.
- `sample_requests.json`: example prediction payloads.

## AWS Mapping

- Store `model.json` in S3.
- Run the API on EC2 Linux.
- Use IAM role permissions for S3 read-only access.
- Send application logs to CloudWatch.
- Track cost with AWS Budgets and Free Tier monitoring.

## API Contract

Endpoint:

```text
POST /predict
```

Input:

```json
{"features":[0.82,0.15,0.12,0.42]}
```

Output:

```json
{"label":"plastic_rigid","score":0.71}
```

