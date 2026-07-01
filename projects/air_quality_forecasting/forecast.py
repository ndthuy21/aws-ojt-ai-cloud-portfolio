from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parents[1]
OUT_DIR = ROOT / "outputs"
DATA_PATH = REPO_ROOT / "data" / "public" / "air_quality" / "AirQualityUCI.csv"
TARGET = "C6H6(GT)"


def load_air_quality(path: Path = DATA_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Download UCI Air Quality dataset first; see README.md."
        )
    df = pd.read_csv(path, sep=";", decimal=",")
    df = df.dropna(axis=1, how="all").dropna(axis=0, how="all")
    df = df.replace(-200, np.nan)
    df["datetime"] = pd.to_datetime(
        df["Date"].astype(str) + " " + df["Time"].astype(str),
        format="%d/%m/%Y %H.%M.%S",
        errors="coerce",
    )
    df = df.dropna(subset=["datetime", TARGET]).sort_values("datetime").reset_index(drop=True)
    return df


def make_features(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[str], pd.DataFrame]:
    work = df.copy()
    work["target_lag_1"] = work[TARGET].shift(1)
    work["target_lag_24"] = work[TARGET].shift(24)
    work["hour"] = work["datetime"].dt.hour
    work["day_of_week"] = work["datetime"].dt.dayofweek
    work["hour_sin"] = np.sin(2 * np.pi * work["hour"] / 24)
    work["hour_cos"] = np.cos(2 * np.pi * work["hour"] / 24)
    features = [
        "CO(GT)",
        "PT08.S1(CO)",
        "PT08.S2(NMHC)",
        "NOx(GT)",
        "PT08.S3(NOx)",
        "NO2(GT)",
        "PT08.S4(NO2)",
        "PT08.S5(O3)",
        "T",
        "RH",
        "AH",
        "target_lag_1",
        "target_lag_24",
        "hour_sin",
        "hour_cos",
        "day_of_week",
    ]
    work = work.dropna(subset=features + [TARGET]).reset_index(drop=True)
    return work[features].to_numpy(float), work[TARGET].to_numpy(float), features, work


def standardize(train: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = train.mean(axis=0)
    std = train.std(axis=0)
    std[std == 0] = 1
    return (train - mean) / std, (test - mean) / std, mean, std


def fit_ridge_regression(X: np.ndarray, y: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    Xb = np.c_[np.ones(len(X)), X]
    penalty = np.eye(Xb.shape[1]) * np.sqrt(alpha)
    penalty[0, 0] = 0.0
    X_aug = np.vstack([Xb, penalty])
    y_aug = np.concatenate([y, np.zeros(Xb.shape[1])])
    return np.linalg.lstsq(X_aug, y_aug, rcond=None)[0]


def predict(X: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return np.sum(X * weights[1:], axis=1) + weights[0]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_air_quality()
    X, y, features, work = make_features(df)

    split = int(len(y) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    X_train_s, X_test_s, mean, std = standardize(X_train, X_test)
    weights = fit_ridge_regression(X_train_s, y_train)
    y_pred = predict(X_test_s, weights)
    errors = y_test - y_pred
    metrics = {
        "dataset": "UCI Air Quality",
        "dataset_url": "https://archive.ics.uci.edu/dataset/360/air+quality",
        "source_rows": int(len(df)),
        "usable_rows_after_cleaning": int(len(y)),
        "train_samples": int(len(y_train)),
        "test_samples": int(len(y_test)),
        "target": TARGET,
        "mae": round(float(np.mean(np.abs(errors))), 4),
        "rmse": round(float(np.sqrt(np.mean(errors**2))), 4),
        "features": features,
    }

    pred_df = work.iloc[split:].copy()
    pred_df["prediction"] = y_pred.round(4)
    pred_df["absolute_error"] = np.abs(errors).round(4)
    pred_df[["datetime", TARGET, "prediction", "absolute_error"]].to_csv(
        OUT_DIR / "predictions.csv", index=False
    )
    (OUT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (OUT_DIR / "model_weights.json").write_text(
        json.dumps(
            {
                "bias": round(float(weights[0]), 6),
                "weights": weights[1:].round(6).tolist(),
                "feature_mean": mean.round(6).tolist(),
                "feature_std": std.round(6).tolist(),
                "features": features,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
