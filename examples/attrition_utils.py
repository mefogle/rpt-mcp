from __future__ import annotations

from typing import Any, Dict, Iterable, List, Sequence

DATASET_ID = "ibm_hr_attrition"
TARGET_COLUMN = "Attrition"
POSITIVE_LABEL = "Yes"


def risk_factor_rules(row: Dict[str, Any]) -> List[str]:
    """Heuristic explanations that highlight contributing factors."""
    factors: List[str] = []
    if row.get("JobSatisfaction") in (1, 2):
        factors.append("Low job satisfaction score")
    if row.get("OverTime") == "Yes":
        factors.append("Frequent overtime workload")
    if row.get("YearsSinceLastPromotion", 0) >= 3:
        factors.append(f"{row['YearsSinceLastPromotion']} years since last promotion")
    if row.get("DistanceFromHome", 0) >= 25:
        factors.append(f"Long commute ({row['DistanceFromHome']} km)")
    if row.get("WorkLifeBalance") in (1, 2):
        factors.append("Low work-life balance score")
    if row.get("MonthlyIncome", 0) <= 3000:
        factors.append("Low monthly income band")
    if row.get("EnvironmentSatisfaction") in (1, 2):
        factors.append("Environment satisfaction concerns")
    if row.get("JobRole") in {"Sales Executive", "Sales Representative"} and row.get("JobLevel", 1) <= 2:
        factors.append("Sales compensation/pressure risk")
    if row.get("TotalWorkingYears", 0) >= 15 and row.get("YearsAtCompany", 0) <= 3:
        factors.append("Experienced hire still early in tenure")
    return factors


def probability_for_label(
    probability_entry: Sequence[Any] | float | int | None,
    label: str = POSITIVE_LABEL,
) -> float:
    """Extract probability for the requested label from the MCP response."""
    if probability_entry is None:
        return 0.0

    if isinstance(probability_entry, (int, float)):
        return float(probability_entry)

    if isinstance(probability_entry, dict):
        if probability_entry.get("prediction") == label:
            return float(probability_entry.get("probability", 0.0))
        return 0.0

    if isinstance(probability_entry, Iterable):
        for entry in probability_entry:
            value = probability_for_label(entry, label)
            if value:
                return value
    return 0.0

