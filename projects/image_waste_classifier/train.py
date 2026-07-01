from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "outputs"
SAMPLE_DIR = OUT_DIR / "sample_images"
LABELS = ["plastic_rigid", "paper_cardboard", "metal"]
FEATURE_NAMES = ["blue_ratio", "white_ratio", "gray_ratio", "edge_density"]


def make_image(label: str, rng: np.random.Generator, size: int = 32) -> Image.Image:
    if label == "plastic_rigid":
        base = (35 + rng.integers(0, 25), 135 + rng.integers(0, 35), 215 + rng.integers(0, 30))
    elif label == "paper_cardboard":
        base = (210 + rng.integers(0, 25), 195 + rng.integers(0, 25), 150 + rng.integers(0, 25))
    else:
        tone = 120 + rng.integers(0, 55)
        base = (tone, tone + rng.integers(-8, 8), tone + rng.integers(-8, 8))

    img = Image.new("RGB", (size, size), base)
    draw = ImageDraw.Draw(img)
    for _ in range(4):
        x1 = int(rng.integers(0, size - 8))
        y1 = int(rng.integers(0, size - 8))
        x2 = x1 + int(rng.integers(5, 14))
        y2 = y1 + int(rng.integers(5, 14))
        if label == "plastic_rigid":
            draw.ellipse((x1, y1, x2, y2), fill=(25, 95, 190))
        elif label == "paper_cardboard":
            draw.rectangle((x1, y1, x2, y2), fill=(235, 225, 190))
        else:
            draw.rounded_rectangle((x1, y1, x2, y2), radius=3, fill=(165, 168, 170))
    return img


def extract_features(img: Image.Image) -> list[float]:
    arr = np.asarray(img).astype(np.float32) / 255.0
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    blue_ratio = float(np.mean((b > r + 0.10) & (b > g + 0.05)))
    white_ratio = float(np.mean((r > 0.72) & (g > 0.68) & (b > 0.50)))
    gray_ratio = float(np.mean((np.abs(r - g) < 0.06) & (np.abs(g - b) < 0.06)))
    gray = arr.mean(axis=2)
    edge_density = float((np.abs(np.diff(gray, axis=0)).mean() + np.abs(np.diff(gray, axis=1)).mean()) / 2)
    return [blue_ratio, white_ratio, gray_ratio, edge_density]


def generate_dataset(n_per_class: int = 90, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    rows = []
    targets = []
    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    for label_index, label in enumerate(LABELS):
        for i in range(n_per_class):
            img = make_image(label, rng)
            rows.append(extract_features(img))
            targets.append(label_index)
            if i < 3:
                img.save(SAMPLE_DIR / f"{label}_{i}.png")
    return np.array(rows, dtype=np.float32), np.array(targets, dtype=np.int64)


def train_softmax(X: np.ndarray, y: np.ndarray, epochs: int = 1200, lr: float = 0.7) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(7)
    n_samples, n_features = X.shape
    n_classes = len(LABELS)
    weights = rng.normal(0, 0.05, size=(n_classes, n_features))
    bias = np.zeros(n_classes)
    one_hot = np.eye(n_classes)[y]

    for _ in range(epochs):
        logits = X @ weights.T + bias
        logits -= logits.max(axis=1, keepdims=True)
        probs = np.exp(logits)
        probs /= probs.sum(axis=1, keepdims=True)
        error = (probs - one_hot) / n_samples
        weights -= lr * (error.T @ X)
        bias -= lr * error.sum(axis=0)
    return weights, bias


def predict(X: np.ndarray, weights: np.ndarray, bias: np.ndarray) -> np.ndarray:
    return np.argmax(X @ weights.T + bias, axis=1)


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    matrix = np.zeros((len(LABELS), len(LABELS)), dtype=int)
    for true, pred in zip(y_true, y_pred):
        matrix[true, pred] += 1
    return matrix


def macro_f1(matrix: np.ndarray) -> float:
    f1_scores = []
    for i in range(len(LABELS)):
        tp = matrix[i, i]
        fp = matrix[:, i].sum() - tp
        fn = matrix[i, :].sum() - tp
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1_scores.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return float(np.mean(f1_scores))


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    X, y = generate_dataset()
    rng = np.random.default_rng(11)
    indices = rng.permutation(len(y))
    split = int(len(y) * 0.8)
    train_idx, test_idx = indices[:split], indices[split:]
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    weights, bias = train_softmax(X_train, y_train)
    y_pred = predict(X_test, weights, bias)
    matrix = confusion_matrix(y_test, y_pred)
    metrics = {
        "dataset_type": "generated_sample_images",
        "train_samples": int(len(y_train)),
        "test_samples": int(len(y_test)),
        "accuracy": round(float(np.mean(y_test == y_pred)), 4),
        "macro_f1": round(macro_f1(matrix), 4),
        "feature_names": FEATURE_NAMES,
        "labels": LABELS,
    }

    pd.DataFrame(X, columns=FEATURE_NAMES).assign(label=[LABELS[i] for i in y]).to_csv(
        OUT_DIR / "features.csv", index=False
    )
    pd.DataFrame(matrix, index=LABELS, columns=LABELS).to_csv(OUT_DIR / "confusion_matrix.csv")
    (OUT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    model = {
        "labels": LABELS,
        "weights": weights.round(6).tolist(),
        "bias": bias.round(6).tolist(),
        "feature_names": FEATURE_NAMES,
    }
    (OUT_DIR / "model.json").write_text(json.dumps(model, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

