"""Small HTTP inference API for AWS OJT portfolio practice.

The app uses only Python standard library so it can run on a fresh EC2 Linux
instance after cloning the repo. It is intentionally simple, but the deployment
shape is the same as a larger ML API: load an artifact, accept JSON input, return
a prediction, and write logs.
"""

from __future__ import annotations

import argparse
import json
import math
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "model.json"


def load_model(path: Path = MODEL_PATH) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        model = json.load(f)
    if not {"labels", "weights", "bias"}.issubset(model):
        raise ValueError("Model artifact must contain labels, weights, and bias")
    return model


def softmax(values: list[float]) -> list[float]:
    max_value = max(values)
    exps = [math.exp(v - max_value) for v in values]
    total = sum(exps)
    return [v / total for v in exps]


def predict(features: list[float], model: dict[str, Any]) -> dict[str, Any]:
    labels = model["labels"]
    weights = model["weights"]
    bias = model["bias"]

    if len(features) != len(weights[0]):
        raise ValueError(f"Expected {len(weights[0])} features, got {len(features)}")

    logits = []
    for class_weights, class_bias in zip(weights, bias):
        score = sum(w * x for w, x in zip(class_weights, features)) + class_bias
        logits.append(score)

    probabilities = softmax(logits)
    best_index = max(range(len(probabilities)), key=probabilities.__getitem__)
    return {
        "label": labels[best_index],
        "score": round(probabilities[best_index], 4),
        "probabilities": {
            label: round(prob, 4) for label, prob in zip(labels, probabilities)
        },
    }


MODEL = load_model()


class InferenceHandler(BaseHTTPRequestHandler):
    server_version = "AWSOJTInferenceAPI/1.0"

    def _send_json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path == "/health":
            self._send_json(200, {"status": "ok"})
            return
        self._send_json(404, {"error": "Use POST /predict or GET /health"})

    def do_POST(self) -> None:
        if self.path != "/predict":
            self._send_json(404, {"error": "Use POST /predict"})
            return

        try:
            content_length = int(self.headers.get("Content-Length", "0"))
            raw_body = self.rfile.read(content_length)
            payload = json.loads(raw_body.decode("utf-8"))
            features = payload["features"]
            if not isinstance(features, list):
                raise ValueError("features must be a list")
            features = [float(x) for x in features]
            result = predict(features, MODEL)
            self._send_json(200, result)
        except Exception as exc:  # Keep API feedback clear for a learning lab.
            self._send_json(400, {"error": str(exc)})


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the portfolio inference API")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), InferenceHandler)
    print(f"Serving on http://{args.host}:{args.port}")
    print("Try GET /health or POST /predict")
    server.serve_forever()


if __name__ == "__main__":
    main()

