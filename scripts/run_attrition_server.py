#!/usr/bin/env python
"""
Helper script that preloads the IBM HR Attrition dataset and launches the MCP server.

Example:
    RPT_API_TOKEN=... python scripts/run_attrition_server.py \
        --dataset data/reference/WA_Fn-UseC_-HR-Employee-Attrition.csv
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

from rpt_mcp_server import load_reference_dataset, main

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Start the MCP server with a pre-loaded IBM HR dataset.")
    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="Path to WA_Fn-UseC_-HR-Employee-Attrition.csv",
    )
    parser.add_argument(
        "--dataset-id",
        default="ibm_hr_attrition",
        help="Identifier used when exposing the dataset through MCP resources/tools.",
    )
    parser.add_argument(
        "--target-column",
        default="Attrition",
        help="Binary target column that indicates churn risk.",
    )
    return parser.parse_args()


def main_cli() -> None:
    args = parse_args()
    if not args.dataset.exists():
        raise FileNotFoundError(f"Dataset not found: {args.dataset}")

    logger.info("Loading dataset '%s' from %s", args.dataset_id, args.dataset)
    load_reference_dataset(
        dataset_id=args.dataset_id,
        filepath=str(args.dataset),
        target_column=args.target_column,
    )

    logger.info("Starting MCP server after preloading IBM HR reference data.")
    main()


if __name__ == "__main__":
    main_cli()
