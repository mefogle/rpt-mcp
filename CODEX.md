# rpt-mcp – Codex/Agent Quickstart

This repository hosts an MCP server that wraps SAP's hosted RPT tabular prediction API. The agent-facing pieces live under `src/rpt_mcp_server/`, while runnable examples and helper scripts live in `examples/` and `scripts/`.

## Environment & Tooling

- Prefer [`uv`](https://github.com/astral-sh/uv) for dependency management. From the repo root: `uv sync --extra dev --extra examples` and run commands via `uv run ...`.
- Secrets:
  - `RPT_API_TOKEN` – required for all server/example runs.
  - `RPT_DATASETS` – JSON mapping of dataset IDs to file paths (optional when running query-only). The repo ships with a trimmed HR dataset under `examples/data/reference/` so the examples work out of the box.
- Essential binaries: `python3.12+`, `uv`, `pytest` (via `uv run pytest`).

## Key Components

| Path | Purpose |
| --- | --- |
| `src/rpt_mcp_server/server.py` | MCP server; exposes dataset resources and prediction tools that proxy to SAP's `/api/predict`. |
| `scripts/run_attrition_server.py` | Helper that loads the IBM HR dataset and starts the MCP server. |
| `examples/attrition_utils.py` | Shared heuristics and constants for attrition agents. |
| `examples/batch_attrition_agent.py` | Batch agent: loads survey CSV, calls MCP, summarizes high-risk employees. Spawns the server over stdio. |
| `tests/test_server.py` | Unit tests for MCP resources/tools using mocked SAP client. |
| `tests/test_attrition_utils.py` | Tests for attrition helper heuristics/probability extraction. |
| `examples/data/reference/WA_Fn-UseC_-HR-Employee-Attrition.csv` | Sample dataset bundled for the HR demo. |
| `examples/data/new_employee_survey.csv` | Synthetic survey rows suitable for demos. |

## Running the Server

```
export RPT_API_TOKEN=...
export RPT_DATASETS='{"ibm_hr": {"path": "examples/data/reference/WA_Fn-UseC_-HR-Employee-Attrition.csv"}}'
uv run python -m rpt_mcp_server --transport stdio

# SSE transport, useful for remote agents (inline dataset mapping)
uv run python -m rpt_mcp_server \
  --transport sse \
  --host 0.0.0.0 \
  --port 8080 \
  --dataset ibm_hr=examples/data/reference/WA_Fn-UseC_-HR-Employee-Attrition.csv

# or preload IBM HR data and run via helper script (stdio by default)
RPT_API_TOKEN=... uv run python -m scripts.run_attrition_server --dataset examples/data/reference/WA_Fn-UseC_-HR-Employee-Attrition.csv
```

Skipping dataset registration is valid—the server will simply pass all caller-provided rows directly to SAP, filling empty fields with `[PREDICT]` automatically.

Docker options (SSE-only):

```
docker build -t rpt-mcp .
docker run -p8080:8080 \
  -v $(pwd)/examples/data:/data:ro \
  -e RPT_API_TOKEN=... \
  -e RPT_DATASETS='{"ibm_hr": {"path": "/data/reference/WA_Fn-UseC_-HR-Employee-Attrition.csv"}}' \
  rpt-mcp

docker build -f examples/Dockerfile.hr -t hr-rpt-mcp-server --build-arg BASE_IMAGE=python:3.12-slim .
docker run -p8080:8080 \
  -e RPT_API_TOKEN=... \
  -e RPT_DATASETS='{"ibm_hr": {"path": "/app/examples/data/reference/WA_Fn-UseC_-HR-Employee-Attrition.csv"}}' \
  hr-rpt-mcp-server
```

## Running Examples

All examples manage their own MCP subprocess via stdio; no separate server process is required.

```
OPENAI_API_KEY=... RPT_API_TOKEN=... uv run python -m examples.batch_attrition_agent \
  --survey examples/data/new_employee_survey.csv \
  --reference examples/data/reference/WA_Fn-UseC_-HR-Employee-Attrition.csv
```

The script defaults to the bundled dataset; pass a different `--reference` path if you have the full Kaggle export. It relies on the `examples` extra (`mcp`, `openai`).

### MCP Transport Notes

- The stdio transport is used; each agent spawns the MCP server so the stdin/stdout pipes are wired up. Reusing an already-running server would require a socket transport.

### Attrition Risk Output

- The SAP API does not expose class probabilities yet. `predict_classification` returns placeholder probability structures, so prompts have been tuned to speak qualitatively (e.g., “high risk” and “schedule retention conversations”).

## Testing & Validation

```
uv run pytest
```

The test suite covers MCP resources/tools (with stubbed SAP client) and the attrition utilities. Examples are intentionally not integration-tested; run them manually when needed.

## Misc

- `pyproject.toml` defines extras (`dev`, `examples`).
- The bundled HR dataset lives under `examples/data/reference/` and is tracked so the examples run without any extra downloads.
- Commit history wraps up SAP API integration, uv migration, example agents, and documentation updates.
