from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples import attrition_utils as utils


def test_risk_factor_rules_detects_multiple_signals():
    row = {
        "JobSatisfaction": 1,
        "OverTime": "Yes",
        "YearsSinceLastPromotion": 4,
        "DistanceFromHome": 32,
        "WorkLifeBalance": 1,
        "MonthlyIncome": 2500,
        "EnvironmentSatisfaction": 2,
        "JobRole": "Sales Executive",
        "JobLevel": 2,
        "TotalWorkingYears": 18,
        "YearsAtCompany": 2,
    }
    factors = utils.risk_factor_rules(row)
    assert "Low job satisfaction score" in factors
    assert any("3 years since last promotion" in f or "4 years" in f for f in factors)
    assert any("Long commute" in f for f in factors)
    assert "Sales compensation/pressure risk" in factors
    assert "Experienced hire still early in tenure" in factors
    assert len(factors) >= 7  # ensure multiple heuristics triggered


@pytest.mark.parametrize(
    "payload,label,expected",
    [
        (0.42, "Yes", 0.42),
        ({"prediction": "Yes", "probability": 0.81}, "Yes", 0.81),
        ({"prediction": "No", "probability": 0.9}, "Yes", 0.0),
        ([{"prediction": "No", "probability": 0.2}, {"prediction": "Yes", "probability": 0.8}], "Yes", 0.8),
        ([{"prediction": "Yes", "probability": 0.55}], "Yes", 0.55),
        (None, "Yes", 0.0),
    ],
)
def test_probability_for_label_handles_various_payloads(payload, label, expected):
    assert utils.probability_for_label(payload, label) == pytest.approx(expected)


def test_probability_for_label_handles_nested_sequences():
    payload = [
        [
            {"prediction": "No", "probability": 0.4},
            {"prediction": "Yes", "probability": 0.6},
        ]
    ]
    assert utils.probability_for_label(payload) == pytest.approx(0.6)


def test_high_risk_mask_thresholding():
    probabilities = [
        {"prediction": "Yes", "probability": 0.85},
        {"prediction": "Yes", "probability": 0.65},
        {"prediction": "No", "probability": 0.9},
        0.72,
    ]
    mask = utils.high_risk_mask(probabilities, threshold=0.70)
    assert mask == [True, False, False, True]
