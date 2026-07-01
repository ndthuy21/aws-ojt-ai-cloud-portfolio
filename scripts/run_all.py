from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run(command: list[str]) -> None:
    print("$", " ".join(command))
    subprocess.run(command, cwd=ROOT, check=True)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    python = sys.executable
    run([python, "projects/image_waste_classifier/train.py"])
    run([python, "projects/air_quality_forecasting/forecast.py"])
    run([python, "projects/weather_traffic_risk/analyze.py"])

    run([python, "-m", "unittest", "discover", "-s", "tests"])

    summary = {
        "image_waste_classifier": load_json(ROOT / "projects" / "image_waste_classifier" / "outputs" / "metrics.json"),
        "air_quality_forecasting": load_json(ROOT / "projects" / "air_quality_forecasting" / "outputs" / "metrics.json"),
        "weather_traffic_risk": load_json(ROOT / "projects" / "weather_traffic_risk" / "outputs" / "metrics.json"),
    }
    summary_path = ROOT / "docs" / "latest_metrics_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote {summary_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
