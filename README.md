# rpt-mcp

MCP server that exposes SAP RPT cloud tabular predictions through FastMCP resources and tools.

## Requirements

* **SAP RPT API token** – set the `RPT_API_TOKEN` environment variable with your personal token (see https://rpt.cloud.sap/docs). Optional overrides:
  * `RPT_API_BASE_URL` for pointing to non-production endpoints.
* **Reference data** – populate `REFERENCE_DATA_DIR` with parquet/CSV datasets that act as the in-context examples for predictions.

## Installation (with [`uv`](https://github.com/astral-sh/uv))

```bash
# from the repository root
uv sync --extra dev --extra examples
```

`uv sync` creates (or refreshes) a local `.venv/` using the dependencies declared in `pyproject.toml`, including the optional `dev` + `examples` extras required for pytest and the MCP demonstration agents. You can either activate the environment (`source .venv/bin/activate`) or simply prefix commands with `uv run ...`.

## Running

Launch the server in-place via `uv run` (which ensures the project is on `PYTHONPATH`):

```bash
export RPT_API_TOKEN=...         # required
export REFERENCE_DATA_DIR=...    # directory containing your parquet/csv files

uv run python -m rpt_mcp_server.server
```

## IBM HR Attrition Examples

1. Download `WA_Fn-UseC_-HR-Employee-Attrition.csv` from the [Kaggle dataset](https://www.kaggle.com/pavansubhasht/ibm-hr-analytics-attrition-dataset).
2. Place it somewhere under your workspace, e.g. `data/reference/WA_Fn-UseC_-HR-Employee-Attrition.csv`.
3. Install the optional dependencies used by the examples: `uv sync --extra dev --extra examples`.
4. A synthetic survey file (`data/new_employee_survey.csv`) is included so you can immediately test the workflow without publishing Kaggle data.

### Batch scenario with 50 employees

```
# Terminal 1 – start the MCP server and preload the dataset
RPT_API_TOKEN=... uv run python -m scripts.run_attrition_server \
  --dataset data/reference/WA_Fn-UseC_-HR-Employee-Attrition.csv

# Terminal 2 – run an agent that calls predict_classification with new survey rows
OPENAI_API_KEY=... RPT_API_TOKEN=... uv run python -m examples.pydantic_attrition_agent \
  --survey data/new_employee_survey.csv \
  --reference data/reference/WA_Fn-UseC_-HR-Employee-Attrition.csv
```

The agent loads the 1,470-row IBM dataset as context, scores the 50 survey entries via `predict_classification`, flags everyone above the 70 % threshold, and produces an HR-friendly summary with risk factors and recommendations.

> The synthetic survey rows capture a mix of scenarios (new grads, mid-level scientists, sales executives with long commutes, etc.) so the classifier produces both low- and high-risk predictions that showcase how the agent surfaces risk factors.

### Conversational one-off inquiry

```
OPENAI_API_KEY=... RPT_API_TOKEN=... uv run python -m examples.interactive_attrition_agent \
  --reference data/reference/WA_Fn-UseC_-HR-Employee-Attrition.csv
```

The script greets you as the agent, asks follow-up questions (job satisfaction, work-life balance, commute, income, etc.), and once it has enough detail it calls `predict_classification` for that single employee. The response includes the attrition probability, the key drivers (e.g., “Low job satisfaction, 3 years without promotion”), and concrete retention steps so you can act quickly.

## Testing

```bash
uv run pytest
```
