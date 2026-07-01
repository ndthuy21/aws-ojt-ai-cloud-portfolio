# Image Waste Classifier

This project is a reproducible computer-vision-style classification demo. It
generates small synthetic images for three classes, extracts simple image
features, trains a softmax classifier with NumPy, and reports accuracy, macro F1
score, and confusion matrix.

## Run

```bash
python projects/image_waste_classifier/train.py
```

Outputs are written to:

```text
projects/image_waste_classifier/outputs/
```

## AWS Extension

- Store generated image data and model artifacts in S3.
- Train or run inference on EC2.
- Expose inference through a small API.
- Log predictions and errors to CloudWatch.

## Note

The dataset is synthetic and intentionally small for reproducibility. The value
of this project is the end-to-end pipeline structure, not the dataset itself.

