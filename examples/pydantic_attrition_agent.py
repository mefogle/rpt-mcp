#!/usr/bin/env python
"""
Batch attrition analysis example that uses the MCP server + Pydantic AI.

Usage:
    OPENAI_API_KEY=... RPT_API_TOKEN=... \
        python examples/pydantic_attrition_agent.py \
            --survey data/new_employee_survey.csv \
            --reference data/reference/WA_Fn-UseC_-HR-Employee-Attrition.csv
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel

from examples.attrition_utils import (
    DATASET_ID,
    HIGH_RISK_THRESHOLD,
    TARGET_COLUMN,
    probability_for_label,
    risk_factor_rules,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze multiple employees for attrition risk.")
    parser.add_argument(
        "--survey",
        type=Path,
        required=True,
        help="CSV file with new employee survey data (same schema as IBM HR dataset).",
    )
    parser.add_argument(
        "--reference",
        type=Path,
        required=True,
        help="Path to the IBM HR dataset used as MCP reference context.",
    )
    parser.add_argument(
        "--server-cmd",
        nargs="+",
        default=["python", "scripts/run_attrition_server.py"],
        help="Command for launching the MCP server.",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        help="OpenAI/Pydantic AI model used for narration.",
    )
    return parser.parse_args()


async def call_predict_classification(session: ClientSession, rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    response = await session.call_tool(
        "predict_classification",
        arguments={
            "dataset_id": DATASET_ID,
            "target_column": TARGET_COLUMN,
            "test_data_json": json.dumps(rows),
            "return_probabilities": True,
        },
    )
    for block in response.content:
        if block.type == "text":
            return json.loads(block.text)
    raise RuntimeError("Unexpected MCP response payload.")


@dataclass
class HighRiskEmployee:
    employee_id: str
    probability: float
    prediction: str
    risk_factors: List[str]


def build_high_risk_list(
    rows: List[Dict[str, Any]],
    predictions: List[Any],
    probabilities: List[Any],
    threshold: float = HIGH_RISK_THRESHOLD,
) -> List[HighRiskEmployee]:
    high_risk: List[HighRiskEmployee] = []
    for idx, row in enumerate(rows):
        probability = probability_for_label(probabilities[idx])
        if probability < threshold:
            continue
        prediction = predictions[idx] if idx < len(predictions) else "Yes"
        employee_id = str(row.get("EmployeeNumber", f"row-{idx+1}"))
        high_risk.append(
            HighRiskEmployee(
                employee_id=employee_id,
                probability=probability,
                prediction=prediction,
                risk_factors=risk_factor_rules(row),
            )
        )
    return high_risk


async def run_analysis(args: argparse.Namespace) -> None:
    survey_df = pd.read_csv(args.survey)
    survey_rows = survey_df.to_dict(orient="records")

    env = dict(os.environ)
    env.setdefault("RPT_API_TOKEN", os.environ.get("RPT_API_TOKEN", ""))

    server_params = StdioServerParameters(
        command=args.server_cmd[0],
        args=args.server_cmd[1:] + ["--dataset", str(args.reference)],
        env=env,
    )

    agent = Agent(model=OpenAIModel(args.model))

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            prediction_payload = await call_predict_classification(session, survey_rows)

    predictions = prediction_payload.get("predictions", [])
    probabilities = prediction_payload.get("probabilities", [])

    high_risk = build_high_risk_list(survey_rows, predictions, probabilities)

    summary_context = {
        "total_rows": len(survey_rows),
        "high_risk_count": len(high_risk),
        "high_risk": [
            {
                "employee_id": emp.employee_id,
                "risk_indicator": "flagged as high attrition risk",
                "relative_confidence": "model classification only",
                "prediction": emp.prediction,
                "risk_factors": emp.risk_factors,
            }
            for emp in high_risk
        ],
    }

    summary_prompt = (
        "You are preparing an HR-facing update based on attrition predictions. "
        "Use the provided JSON payload to craft a concise narrative that:\n"
        "1. States how many employees were evaluated.\n"
        "2. Identifies the employees flagged as high risk and explain why (no numeric probabilities; "
        "describe them qualitatively instead).\n"
        "3. Summarizes the predominant drivers across those employees.\n"
        "4. Lists concrete next steps the HR team can take. At least one recommendation should explicitly "
        "mention scheduling retention (stay) conversations with the high-risk employees.\n"
        f"Payload:\n{json.dumps(summary_context, indent=2)}"
    )

    result = await agent.run(summary_prompt)
    output_text = getattr(result, "text", None) or getattr(result, "output_text", None) or str(result)
    print(output_text)


def main() -> None:
    args = parse_args()
    asyncio.run(run_analysis(args))


if __name__ == "__main__":
    main()
