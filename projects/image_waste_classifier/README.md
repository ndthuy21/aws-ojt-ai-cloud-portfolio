# TACO Waste Dataset Preparation

This project documents a public computer-vision dataset preparation workflow
using TACO (Trash Annotations in Context). It summarizes the conversion from raw
TACO annotations into a YOLO-style object-detection dataset with six practical
waste classes.

## Run

```bash
python projects/image_waste_classifier/train.py
```

Outputs are written to:

```text
projects/image_waste_classifier/outputs/
```

## Evidence

- Source dataset: TACO public dataset.
- Source size: 1,500 images, 4,784 annotations, 60 raw categories.
- Output format: YOLO-style labels grouped into 6 classes.
- Usable boxes written: 4,241.
- Split: 1,200 train images, 150 validation images, 150 test images.
- Validation checks passed: split size, label count, box count, label parsing,
  and no duplicate image hashes between splits.

## AWS Extension

- Store raw images, YOLO labels, and reports in S3.
- Run conversion and validation scripts on EC2 or AWS Batch.
- Track dataset versions and validation reports before model training.
- Send conversion logs and validation failures to CloudWatch.
