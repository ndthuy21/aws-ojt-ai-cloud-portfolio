# AWS OJT AI Cloud Portfolio

This repository is a compact, reproducible portfolio for AWS OJT Fall 2026.
It focuses on practical AI/data work, AWS readiness, documentation, and cost-aware
cloud deployment planning.

## Project Links And Results

| Project | Direct Link | Evidence |
|---|---|---|
| ML Inference API | [cloud_ready_ml_api](cloud_ready_ml_api/) | Local `/health` and `/predict` endpoints tested; unit tests included |
| TACO Waste Dataset Preparation | [projects/image_waste_classifier](projects/image_waste_classifier/) | Public TACO dataset summary: 1,500 images, 4,784 annotations, 6 output classes, 4,241 YOLO boxes, 80/10/10 split validated |
| Air Quality Forecasting | [projects/air_quality_forecasting](projects/air_quality_forecasting/) | Public UCI Air Quality dataset: 6,919 usable rows after cleaning, 1,384-test rows, MAE 1.1350, RMSE 1.3775 |
| Weather Traffic Volume Prediction | [projects/weather_traffic_risk](projects/weather_traffic_risk/) | Public UCI Metro Interstate Traffic Volume dataset: 48,204 rows, 9,641-test rows, MAE 893.9778, RMSE 1145.3156 |

## Current Status

- Implemented and tested local prototypes.
- Exported metrics, compact public-dataset evidence, predictions, and model weights.
- Documented AWS deployment path as the next step: EC2, S3, IAM, CloudWatch, AWS CLI, and Free Tier cost control.
- No AWS production deployment is claimed in this repository.

## What This Repo Shows

- Python, data processing, and ML fundamentals.
- Reproducible mini-projects using public datasets and compact dataset reports.
- Evaluation metrics such as MAE and RMSE, plus dataset preparation checks for object-detection data.
- A small HTTP inference API with a documented EC2 deployment path.
- AWS learning map for S3 artifacts, IAM least-privilege access, CloudWatch logs, and
  Free Tier cost control.

## Project Structure

```text
cloud_ready_ml_api/
  app.py                    # Standard-library HTTP inference API
  model.json                # Small model artifact used by the API
  architecture.md           # AWS deployment architecture

projects/
  image_waste_classifier/   # Public TACO dataset preparation evidence
  air_quality_forecasting/  # Public UCI Air Quality regression pipeline
  weather_traffic_risk/     # Public UCI traffic-volume regression pipeline

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

## Public Dataset Sources

- TACO: https://github.com/pedropro/TACO
- UCI Air Quality: https://archive.ics.uci.edu/dataset/360/air+quality
- UCI Metro Interstate Traffic Volume: https://archive.ics.uci.edu/dataset/492/metro+interstate+traffic+volume

The TACO raw images are not committed because they are large. This repo stores a
compact conversion and validation report so the evidence can be reviewed quickly.
The UCI CSV files are small enough to keep in the repository for direct reruns.

## AWS OJT Relevance

This repo is designed to support the following AWS OJT expectations:

- Basic programming and IT foundation.
- English technical documentation.
- Linux/CLI-based practice.
- AWS account preparation with billing awareness.
- Hands-on project thinking, not only theory.
- AI/ML and Data & Analytics track alignment.
