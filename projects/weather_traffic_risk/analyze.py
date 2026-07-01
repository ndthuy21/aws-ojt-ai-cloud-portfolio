from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "outputs"
FEATURES = ["rain_mm", "wind_speed", "visibility_km", "temperature", "is_weekend"]


def generate_data(days: int = 420, seed: int = 2026) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rain_mm = rng.gamma(shape=1.6, scale=4.0, size=days)
    wind_speed = rng.normal(18, 6, days).clip(2, 55)
    visibility_km = (12 - 0.22 * rain_mm - 0.05 * wind_speed + rng.normal(0, 1.2, days)).clip(1, 15)
    temperature = rng.normal(30, 4, days)
    is_weekend = (np.arange(days) % 7 >= 5).astype(int)

    risk_logit = (
        -1.6
        + 0.18 * rain_mm
        + 0.055 * wind_speed
        - 0.12 * visibility_km
        + 0.45 * is_weekend
        + 0.025 * np.maximum(temperature - 32, 0)
        + rng.normal(0, 0.55, days)
    )
    probability = 1 / (1 + np.exp(-risk_logit))
    high_risk = rng.binomial(1, probability)
    accident_count = rng.poisson(1.2 + 5.0 * probability)
    return pd.DataFrame(
        {
            "day": np.arange(days),
            "rain_mm": rain_mm.round(2),
            "wind_speed": wind_speed.round(2),
            "visibility_km": visibility_km.round(2),
            "temperature": temperature.round(2),
            "is_weekend": is_weekend,
            "accident_count": accident_count,
            "high_risk": high_risk,
        }
    )


def standardize(train: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = train.mean(axis=0)
    std = train.std(axis=0)
    std[std == 0] = 1
    return (train - mean) / std, (test - mean) / std, mean, std


def fit_logistic(X: np.ndarray, y: np.ndarray, epochs: int = 1600, lr: float = 0.12) -> tuple[np.ndarray, float]:
    weights = np.zeros(X.shape[1])
    bias = 0.0
    for _ in range(epochs):
        logits = X @ weights + bias
        probs = 1 / (1 + np.exp(-logits))
        error = probs - y
        weights -= lr * (X.T @ error) / len(y)
        bias -= lr * float(error.mean())
    return weights, bias


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "accuracy": round(float((tp + tn) / len(y_true)), 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1": round(float(f1), 4),
        "true_positive": tp,
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = generate_data()
    df.to_csv(OUT_DIR / "sample_weather_accidents.csv", index=False)

    X = df[FEATURES].to_numpy(float)
    y = df["high_risk"].to_numpy(int)
    split = int(len(df) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    X_train_s, X_test_s, mean, std = standardize(X_train, X_test)
    weights, bias = fit_logistic(X_train_s, y_train)
    probabilities = 1 / (1 + np.exp(-(X_test_s @ weights + bias)))
    y_pred = (probabilities >= 0.5).astype(int)
    report = {
        "dataset_type": "generated_weather_traffic_sample",
        "train_samples": int(len(y_train)),
        "test_samples": int(len(y_test)),
        "features": FEATURES,
        **metrics(y_test, y_pred),
    }

    pred_df = df.iloc[split:].copy()
    pred_df["risk_probability"] = probabilities.round(4)
    pred_df["predicted_high_risk"] = y_pred
    pred_df.to_csv(OUT_DIR / "predictions.csv", index=False)
    matrix = pd.DataFrame(
        [[report["true_negative"], report["false_positive"]], [report["false_negative"], report["true_positive"]]],
        index=["actual_low", "actual_high"],
        columns=["pred_low", "pred_high"],
    )
    matrix.to_csv(OUT_DIR / "confusion_matrix.csv")
    (OUT_DIR / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (OUT_DIR / "model_weights.json").write_text(
        json.dumps(
            {
                "bias": round(float(bias), 6),
                "weights": weights.round(6).tolist(),
                "feature_mean": mean.round(6).tolist(),
                "feature_std": std.round(6).tolist(),
                "features": FEATURES,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
