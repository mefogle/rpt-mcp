# rpt-mcp

MCP server that exposes SAP RPT cloud tabular predictions through FastMCP resources and tools.

## Requirements

* **SAP RPT API token** – set the `RPT_API_TOKEN` environment variable with your personal token (see https://rpt.cloud.sap/docs). Optional overrides:
  * `RPT_API_BASE_URL` for pointing to non-production endpoints.
* **Reference data** – optionally set the `RPT_DATASETS` environment variable with a JSON object that maps dataset IDs to file paths. Example:
  * `export RPT_DATASETS='{"ibm_hr": {"path": "data/reference/WA_Fn-UseC_-HR-Employee-Attrition.csv"}}'`

## Installation (with [`uv`](https://github.com/astral-sh/uv))

```bash
# from the repository root
uv sync --extra dev --extra examples
```

`uv sync` creates (or refreshes) a local `.venv/` using the dependencies declared in `pyproject.toml`, including the optional `dev` + `examples` extras required for pytest and the MCP demonstration agents. You can either activate the environment (`source .venv/bin/activate`) or simply prefix commands with `uv run ...`.

## Running

### Local stdio transport

```bash
export RPT_API_TOKEN=...  # required
export RPT_DATASETS='{"ibm_hr": {"path": "data/reference/WA_Fn-UseC_-HR-Employee-Attrition.csv"}}'

uv run python -m rpt_mcp_server --transport stdio
```

### SSE transport (non-container)

```bash
uv run python -m rpt_mcp_server \
  --transport sse \
  --host 0.0.0.0 \
  --port 8080 \
  --allowed-origins https://your-agent.example \
  --dataset ibm_hr=data/reference/WA_Fn-UseC_-HR-Employee-Attrition.csv
```

Each `--dataset` flag registers an ID/path pair. You can also configure the same mapping via `RPT_DATASETS`.

If you skip `RPT_DATASETS`/`--dataset`, the server still runs—every MCP call will be treated as query-only input, and any missing values are automatically replaced with `[PREDICT]` before hitting the SAP API.

### Docker (SSE transport)

The container image only bundles the MCP server package—examples and dev extras stay outside the
image. Build and run it like this:

```bash
docker build -t rpt-mcp .
docker run \
  -p 8080:8080 \
  -e RPT_API_TOKEN=... \
  -e RPT_DATASETS='{"ibm_hr": {"path": "/data/reference/WA_Fn-UseC_-HR-Employee-Attrition.csv"}}' \
  -v $(pwd)/data/reference:/data/reference:ro \
  rpt-mcp
```

The container entrypoint defaults to `--transport sse --host 0.0.0.0 --port 8080`, so remote MCP
clients can connect over Server-Sent Events without a separate wrapper process.

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
OPENAI_API_KEY=... RPT_API_TOKEN=... uv run python -m examples.batch_attrition_agent \
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
