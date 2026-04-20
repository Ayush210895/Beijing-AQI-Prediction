"""Command-line interface for training Beijing AQI models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from beijing_aqi.data import build_feature_frame
from beijing_aqi.models import TrainingConfig, train_all_models


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Beijing AQI prediction models.")
    parser.add_argument(
        "--data-dir",
        default="data/PRSA_Data_20130301-20170228",
        help="Directory containing the extracted UCI station CSV files.",
    )
    parser.add_argument(
        "--output-dir",
        default="reports",
        help="Directory for metrics and model artifacts.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Optional row sample for faster local smoke runs.",
    )
    parser.add_argument("--test-size", type=float, default=0.30)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    frame = build_feature_frame(args.data_dir)
    if args.sample_size:
        frame = frame.sample(
            n=min(args.sample_size, len(frame)),
            random_state=args.random_state,
        )

    config = TrainingConfig(test_size=args.test_size, random_state=args.random_state)
    metrics = train_all_models(frame, output_dir, config)

    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"Saved metrics to {metrics_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
