#!/usr/bin/env python
"""
Conversational single-employee attrition analysis using MCP + OpenAI Responses.

Example:
    OPENAI_API_KEY=... RPT_API_TOKEN=... \
        python examples/interactive_attrition_agent.py \
            --reference data/reference/WA_Fn-UseC_-HR-Employee-Attrition.csv
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from examples.attrition_utils import (
    DATASET_ID,
    TARGET_COLUMN,
    probability_for_label,
    risk_factor_rules,
)
from examples.openai_utils import generate_summary_text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Have a conversation about a single at-risk employee.")
    parser.add_argument(
        "--reference",
        type=Path,
        required=True,
        help="Path to the IBM HR dataset (CSV).",
    )
    parser.add_argument(
        "--server-cmd",
        nargs="+",
        default=["python", "scripts/run_attrition_server.py"],
        help="Command used to launch the MCP server.",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        help="OpenAI/Pydantic AI model identifier.",
    )
    return parser.parse_args()


def build_baseline_row(df: pd.DataFrame) -> Dict[str, Any]:
    baseline: Dict[str, Any] = {}
    for column in df.columns:
        if column == TARGET_COLUMN:
            continue
        series = df[column]
        if pd.api.types.is_numeric_dtype(series):
            baseline[column] = float(series.mean())
        else:
            baseline[column] = series.mode().iloc[0]
    return baseline


def normalize_choice(value: str, options: List[str]) -> str:
    for option in options:
        if value.lower() == option.lower():
            return option
    raise ValueError(f"Value '{value}' is not one of {options}")


def clamp_int(value: str | int, min_value: int, max_value: int) -> int:
    val = int(value)
    return max(min_value, min(max_value, val))


@dataclass
class Question:
    column: Optional[str]
    prompt: str
    converter: Any = str
    choices: Optional[List[str]] = None
    default: Any = None


def ask_user(question: Question) -> Any:
    while True:
        agent_text = f"Agent: {question.prompt}"
        if question.choices:
            agent_text += f" ({'/'.join(question.choices)})"
        if question.default:
            agent_text += f" [default: {question.default}]"
        agent_text += "\n> "
        raw = input(agent_text).strip()
        if not raw and question.default is not None:
            return question.default
        if not raw:
            print("Agent: Please provide a response so I can assess the employee.")
            continue
        try:
            value = question.converter(raw)
            if question.choices and str(value) not in question.choices:
                print(f"Agent: Please choose one of {question.choices}.")
                continue
            return value
        except Exception:
            print("Agent: I couldn't interpret that. Let's try again.")


def gather_employee_profile(df: pd.DataFrame, baseline: Dict[str, Any]) -> tuple[Dict[str, Any], str]:
    print("Agent: Tell me about the employee. I'll ask a few quick questions.")
    departments = sorted(df["Department"].unique())
    job_roles = sorted(df["JobRole"].unique())
    marital_statuses = sorted(df["MaritalStatus"].unique())
    travel_types = sorted(df["BusinessTravel"].unique())

    next_employee_id = int(df["EmployeeNumber"].max()) + 1
    questions = [
        Question(
            column="EmployeeNumber",
            prompt="What's the employee ID or a memorable number?",
            converter=int,
            default=next_employee_id,
        ),
        Question(column=None, prompt="What's the employee's name or nickname?", converter=str),
        Question(column="Age", prompt="How old are they?", converter=int),
        Question(
            column="JobRole",
            prompt="What's their job role?",
            converter=lambda v: normalize_choice(v, job_roles),
            choices=job_roles,
            default=job_roles[0],
        ),
        Question(
            column="Department",
            prompt="Which department are they in?",
            converter=lambda v: normalize_choice(v, departments),
            choices=departments,
            default=departments[0],
        ),
        Question(
            column="JobLevel",
            prompt="What job level (1-5)?",
            converter=lambda v: clamp_int(v, 1, 5),
            default=int(round(df["JobLevel"].mean())),
        ),
        Question(
            column="MonthlyIncome",
            prompt="Monthly income (USD)?",
            converter=int,
            default=int(df["MonthlyIncome"].median()),
        ),
        Question(
            column="JobSatisfaction",
            prompt="Job satisfaction 1 (low) - 4 (high)?",
            converter=lambda v: clamp_int(v, 1, 4),
            default=2,
        ),
        Question(
            column="WorkLifeBalance",
            prompt="Work-life balance 1 (poor) - 4 (excellent)?",
            converter=lambda v: clamp_int(v, 1, 4),
            default=2,
        ),
        Question(
            column="EnvironmentSatisfaction",
            prompt="Environment satisfaction 1-4?",
            converter=lambda v: clamp_int(v, 1, 4),
            default=3,
        ),
        Question(column="YearsAtCompany", prompt="How many years at the company?", converter=int, default=3),
        Question(column="YearsSinceLastPromotion", prompt="Years since last promotion?", converter=int, default=2),
        Question(
            column="TotalWorkingYears",
            prompt="Total working years?",
            converter=int,
            default=int(df["TotalWorkingYears"].median()),
        ),
        Question(
            column="OverTime",
            prompt="Do they frequently work overtime?",
            converter=lambda v: "Yes" if v.lower().startswith("y") else "No",
            choices=["Yes", "No"],
            default="No",
        ),
        Question(
            column="DistanceFromHome",
            prompt="Distance from home to office (km)?",
            converter=int,
            default=int(df["DistanceFromHome"].median()),
        ),
        Question(
            column="MaritalStatus",
            prompt="Marital status?",
            converter=lambda v: normalize_choice(v, marital_statuses),
            choices=marital_statuses,
            default=marital_statuses[0],
        ),
        Question(
            column="BusinessTravel",
            prompt="Typical business travel frequency?",
            converter=lambda v: normalize_choice(v, travel_types),
            choices=travel_types,
            default=travel_types[0],
        ),
    ]

    filled_row = dict(baseline)
    display_name = ""
    for question in questions:
        value = ask_user(question)
        if question.column:
            filled_row[question.column] = value
        else:
            display_name = value
    return filled_row, display_name


async def call_predict_classification(session: ClientSession, row: Dict[str, Any]) -> Dict[str, Any]:
    response = await session.call_tool(
        "predict_classification",
        arguments={
            "dataset_id": DATASET_ID,
            "test_data_json": json.dumps([row]),
            "return_probabilities": True,
        },
    )
    for block in response.content:
        if block.type == "text":
            return json.loads(block.text)
    raise RuntimeError("Unexpected MCP response payload.")


async def assess_employee(args: argparse.Namespace) -> None:
    df = pd.read_csv(args.reference)
    baseline = build_baseline_row(df)
    profile, display_name = gather_employee_profile(df, baseline)
    profile[TARGET_COLUMN] = None

    env = dict(os.environ)
    env.setdefault("RPT_API_TOKEN", os.environ.get("RPT_API_TOKEN", ""))

    server_params = StdioServerParameters(
        command=args.server_cmd[0],
        args=args.server_cmd[1:] + ["--dataset", str(args.reference)],
        env=env,
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            payload = await call_predict_classification(session, profile)

    prediction_rows = payload.get("predictions", [{}])
    probability_rows = payload.get("probabilities", [{}])
    prediction = (prediction_rows[0] or {}).get(TARGET_COLUMN, "No")
    probability = probability_for_label((probability_rows[0] or {}).get(TARGET_COLUMN))
    risk_factors = risk_factor_rules(profile)

    summary_payload = {
        "employee_name": display_name or profile.get("EmployeeNumber"),
        "employee_number": profile.get("EmployeeNumber"),
        "probability": round(probability * 100, 2),
        "prediction": prediction,
        "risk_factors": risk_factors,
        "profile": profile,
    }

    summary_prompt = (
        "You are advising an HR manager about a single employee. "
        "Use the JSON payload below to:\n"
        "1. Provide the attrition probability and classification.\n"
        "2. Explain the main drivers of the risk based on the profile.\n"
        "3. Offer concrete next steps to retain the employee.\n"
        f"Payload:\n{json.dumps(summary_payload, indent=2)}"
    )

    output_text = await generate_summary_text(args.model, summary_prompt)
    print("\n--- Attrition Assessment ---")
    print(output_text)


def main() -> None:
    args = parse_args()
    asyncio.run(assess_employee(args))


if __name__ == "__main__":
    main()
