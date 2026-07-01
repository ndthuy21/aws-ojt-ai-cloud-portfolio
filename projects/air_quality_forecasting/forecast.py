from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "outputs"


def generate_sensor_data(hours: int = 360, seed: int = 123) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    t = np.arange(hours)
    temperature = 29 + 4 * np.sin(2 * np.pi * t / 24) + rng.normal(0, 0.7, hours)
    humidity = 70 - 8 * np.sin(2 * np.pi * t / 24) + rng.normal(0, 2.0, hours)
    wind_speed = 2.5 + 0.8 * np.sin(2 * np.pi * t / 48) + rng.normal(0, 0.25, hours)
    traffic_index = 45 + 25 * ((t % 24 >= 7) & (t % 24 <= 9)) + 20 * ((t % 24 >= 17) & (t % 24 <= 19))
    traffic_index = traffic_index + rng.normal(0, 4.0, hours)
    pm25 = (
        18
        + 0.34 * humidity
        + 0.42 * traffic_index
        - 4.2 * wind_speed
        + 2.5 * np.sin(2 * np.pi * t / 168)
        + rng.normal(0, 3.5, hours)
    )
    return pd.DataFrame(
        {
            "hour": t,
            "temperature": temperature.round(2),
            "humidity": humidity.round(2),
            "wind_speed": wind_speed.round(2),
            "traffic_index": traffic_index.round(2),
            "pm25": pm25.round(2),
        }
    )


def make_features(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[str], pd.DataFrame]:
    work = df.copy()
    work["pm25_lag_1"] = work["pm25"].shift(1)
    work["pm25_lag_24"] = work["pm25"].shift(24)
    work["hour_sin"] = np.sin(2 * np.pi * work["hour"] / 24)
    work["hour_cos"] = np.cos(2 * np.pi * work["hour"] / 24)
    work = work.dropna().reset_index(drop=True)
    features = [
        "temperature",
        "humidity",
        "wind_speed",
        "traffic_index",
        "pm25_lag_1",
        "pm25_lag_24",
        "hour_sin",
        "hour_cos",
    ]
    return work[features].to_numpy(float), work["pm25"].to_numpy(float), features, work


def fit_ridge_regression(X: np.ndarray, y: np.ndarray, alpha: float = 0.1) -> np.ndarray:
    Xb = np.c_[np.ones(len(X)), X]
    penalty = np.eye(Xb.shape[1]) * alpha
    penalty[0, 0] = 0.0
    return np.linalg.solve(Xb.T @ Xb + penalty, Xb.T @ y)


def predict(X: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return np.c_[np.ones(len(X)), X] @ weights


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = generate_sensor_data()
    df.to_csv(OUT_DIR / "sample_sensor_readings.csv", index=False)

    X, y, features, work = make_features(df)
    split = int(len(y) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    weights = fit_ridge_regression(X_train, y_train)
    y_pred = predict(X_test, weights)
    errors = y_test - y_pred
    metrics = {
        "dataset_type": "generated_sensor_timeseries",
        "train_samples": int(len(y_train)),
        "test_samples": int(len(y_test)),
        "mae": round(float(np.mean(np.abs(errors))), 4),
        "rmse": round(float(np.sqrt(np.mean(errors**2))), 4),
        "features": features,
    }

    pred_df = work.iloc[split:].copy()
    pred_df["prediction"] = y_pred.round(2)
    pred_df["absolute_error"] = np.abs(errors).round(2)
    pred_df.to_csv(OUT_DIR / "predictions.csv", index=False)
    (OUT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (OUT_DIR / "model_weights.json").write_text(
        json.dumps({"bias": round(float(weights[0]), 6), "weights": weights[1:].round(6).tolist(), "features": features}, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

