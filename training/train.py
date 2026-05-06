"""Command-line entry point for model training."""

from __future__ import annotations

import argparse
import json

from training.pipeline import train_all_states
from utils.logger import configure_logging, get_logger


configure_logging()
logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train sales forecasting models.")
    parser.add_argument("--excel-path", default=None, help="Path to Excel sales file.")
    parser.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="Optional subset of models: sarima prophet xgboost lstm",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger.info("CLI training started excel_path=%s models=%s", args.excel_path, args.models)
    registry = train_all_states(args.excel_path, args.models)
    logger.info("CLI training completed states=%s", len(registry.get("states", {})))
    print(json.dumps(registry, indent=2))


if __name__ == "__main__":
    main()
