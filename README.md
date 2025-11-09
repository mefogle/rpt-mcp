# rpt-mcp

MCP server that exposes SAP RPT cloud tabular predictions through FastMCP resources and tools.

## Requirements

* **SAP RPT API token** – set the `RPT_API_TOKEN` environment variable with your personal token (see https://rpt.cloud.sap/docs). Optional overrides:
  * `RPT_API_BASE_URL` for pointing to non-production endpoints.
* **Reference data** – the repo ships with a trimmed IBM HR dataset under `examples/data/reference/`. Use it as-is or swap in your own file by pointing `RPT_DATASETS` (e.g. `export RPT_DATASETS='{"ibm_hr": {"path": "examples/data/reference/WA_Fn-UseC_-HR-Employee-Attrition.csv"}}'`).

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
export RPT_DATASETS='{"ibm_hr": {"path": "examples/data/reference/WA_Fn-UseC_-HR-Employee-Attrition.csv"}}'

uv run python -m rpt_mcp_server --transport stdio
```

### SSE transport (non-container)

```bash
uv run python -m rpt_mcp_server \
  --transport sse \
  --host 0.0.0.0 \
  --port 8080 \
  --allowed-origins https://your-agent.example \
  --dataset ibm_hr=examples/data/reference/WA_Fn-UseC_-HR-Employee-Attrition.csv
```

Each `--dataset` flag registers an ID/path pair. You can also configure the same mapping via `RPT_DATASETS`.

If you skip `RPT_DATASETS`/`--dataset`, the server still runs—every MCP call will be treated as query-only input, and any missing values are automatically replaced with `[PREDICT]` before hitting the SAP API.

### Docker (SSE transport)

The root `Dockerfile` is a reusable template that expects datasets to be mounted at runtime:

```bash
docker build -t rpt-mcp .
docker run \
  -p 8080:8080 \
  -v $(pwd)/examples/data:/data:ro \
  -e RPT_API_TOKEN=... \
  -e RPT_DATASETS='{"ibm_hr": {"path": "/data/reference/WA_Fn-UseC_-HR-Employee-Attrition.csv"}}' \
  rpt-mcp
```

For a self-contained HR demo, an example-specific Dockerfile lives under `examples/Dockerfile.hr` and bundles the sample dataset directly:

```bash
docker build -f examples/Dockerfile.hr -t hr-rpt-mcp-server --build-arg BASE_IMAGE=python:3.12-slim .
docker run \
  -p 8080:8080 \
  -e RPT_API_TOKEN=... \
  -e RPT_DATASETS='{"ibm_hr": {"path": "/app/examples/data/reference/WA_Fn-UseC_-HR-Employee-Attrition.csv"}}' \
  hr-rpt-mcp-server
```

Both images default to `--transport sse --host 0.0.0.0 --port 8080`, so remote MCP clients can connect over Server-Sent Events without a separate wrapper process.

## IBM HR Attrition Examples

1. Install the optional dependencies used by the examples: `uv sync --extra dev --extra examples`.
2. Use the bundled sample dataset (`examples/data/reference/WA_Fn-UseC_-HR-Employee-Attrition.csv`) or point `--reference` at your own Kaggle export.
3. A synthetic survey file (`examples/data/new_employee_survey.csv`) is included so you can immediately test the workflow without publishing Kaggle data.

### Batch scenario 

```
OPENAI_API_KEY=... RPT_API_TOKEN=... uv run python -m examples.batch_attrition_agent \
  --survey examples/data/new_employee_survey.csv \
  --reference examples/data/reference/WA_Fn-UseC_-HR-Employee-Attrition.csv
```

The agent loads the 1,470-row IBM dataset as context, scores the 12 survey entries via `predict_classification`, flags everyone who seems likely to leave and produces an HR-friendly summary with risk factors and recommendations.

> The synthetic survey rows capture a mix of scenarios (new grads, mid-level scientists, sales executives with long commutes, etc.) so the classifier produces both low- and high-risk predictions that showcase how the agent surfaces risk factors.

## Testing

```bash
uv run pytest
```
