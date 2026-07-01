from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parents[1]
OUT_DIR = ROOT / "outputs"
CONVERSION_REPORT = REPO_ROOT / "data" / "public" / "taco" / "taco_yolo_conversion_report.json"
VALIDATION_REPORT = REPO_ROOT / "data" / "public" / "taco" / "taco_yolo_validation_report.json"


def load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Download TACO or add the compact conversion report; see README.md."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    conversion = load_json(CONVERSION_REPORT)
    validation = load_json(VALIDATION_REPORT)

    source = conversion["source"]
    summary = conversion["summary"]
    split_summary = validation["summary"]
    checks = validation["checks"]
    class_distribution = validation["class_distribution"]

    metrics = {
        "dataset": "TACO (Trash Annotations in Context)",
        "dataset_url": "https://github.com/pedropro/TACO",
        "task_type": "object_detection_dataset_preparation",
        "source_images": int(source["images"]),
        "source_annotations": int(source["annotations"]),
        "source_categories": int(source["categories"]),
        "output_classes": int(summary["output_classes"]),
        "yolo_boxes_written": int(summary["written_bboxes"]),
        "dropped_boxes": int(summary["dropped_bboxes"]),
        "train_images": int(summary["train_images"]),
        "val_images": int(summary["val_images"]),
        "test_images": int(summary["test_images"]),
        "train_boxes": int(split_summary["train"]["boxes"]),
        "val_boxes": int(split_summary["val"]["boxes"]),
        "test_boxes": int(split_summary["test"]["boxes"]),
        "duplicate_hashes_between_splits": int(validation["duplicate_hashes_between_splits"]),
        "max_class_ratio_error_pct": float(validation["max_class_ratio_error_pct"]),
        "validation_passed": bool(all(checks.values()) and not validation["errors"]),
        "validation_checks": checks,
        "classes": [row["class"] for row in class_distribution],
    }

    pd.DataFrame(class_distribution).to_csv(OUT_DIR / "class_distribution.csv", index=False)
    pd.DataFrame(
        [
            {
                "split": split,
                "images": values["images"],
                "labels": values["labels"],
                "boxes": values["boxes"],
                "classes_present": values["classes_present"],
            }
            for split, values in split_summary.items()
        ]
    ).to_csv(OUT_DIR / "split_summary.csv", index=False)
    (OUT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
