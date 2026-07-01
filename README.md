# AWS OJT AI Cloud Portfolio

This repository is a compact, reproducible portfolio for AWS OJT Fall 2026.
It focuses on practical AI/data work, AWS readiness, documentation, and cost-aware
cloud deployment planning.

## What This Repo Shows

- Python, data processing, and ML fundamentals.
- Reproducible mini-projects with generated sample data.
- Evaluation metrics such as accuracy, F1 score, confusion matrix, MAE, and RMSE.
- A small HTTP inference API that can be deployed on an EC2 Linux instance.
- AWS mapping for S3 artifacts, IAM least-privilege access, CloudWatch logs, and
  Free Tier cost control.

## Project Structure

```text
cloud_ready_ml_api/
  app.py                    # Standard-library HTTP inference API
  model.json                # Small model artifact used by the API
  architecture.md           # AWS deployment architecture

projects/
  image_waste_classifier/   # Synthetic image classification pipeline
  air_quality_forecasting/  # Time-series forecasting pipeline
  weather_traffic_risk/     # Weather-risk classification pipeline

scripts/
  run_all.py                # Rebuilds all outputs and runs smoke checks

docs/
  aws_deployment_notes.md   # EC2/S3/IAM/CloudWatch/Free Tier plan
```

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/run_all.py
python cloud_ready_ml_api/app.py --port 8000
```

Then test the API:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features":[0.82,0.15,0.12,0.42]}'
```

## Important Honesty Note

The datasets in this repository are generated sample datasets. They are used so
the project is easy to run, review, and reproduce without uploading large private
or third-party data. In a real AWS deployment, the same structure can be connected
to real project data stored in Amazon S3.

## AWS OJT Relevance

This repo is designed to support the following AWS OJT expectations:

- Basic programming and IT foundation.
- English technical documentation.
- Linux/CLI-based practice.
- AWS account preparation with billing awareness.
- Hands-on project thinking, not only theory.
- AI/ML and Data & Analytics track alignment.

