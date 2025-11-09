from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import requests

try:  # pragma: no cover - exercised indirectly in tests via fallback
    from fastmcp import FastMCP
except ImportError:  # pragma: no cover
    class FastMCP:  # minimal stub to keep the module importable in lean envs
        def __init__(self, name: str):
            self.name = name

        def resource(self, *args, **kwargs):
            def decorator(func):
                return func

            return decorator

        def tool(self, *args, **kwargs):
            def decorator(func):
                return func

            return decorator

        def run(self, *args, **kwargs):
            raise RuntimeError("FastMCP dependency missing; install fastmcp to run the server.")


mcp = FastMCP("sap-rpt-tabular-predictor")

# Reference datasets keyed by dataset_id
_reference_data_cache: Dict[str, Dict[str, Any]] = {}

logger = logging.getLogger(__name__)

PREDICT_TOKEN = "[PREDICT]"
DEFAULT_ROW_ID_FIELD = "__row_id"
MAX_CONTEXT_ROWS = 2048


class SAPRPTError(RuntimeError):
    """Base error for SAP RPT API failures."""

    def __init__(self, message: str, status_code: int, retry_after: Optional[str] = None):
        super().__init__(message)
        self.status_code = status_code
        self.retry_after = retry_after


class SAPRPTAuthenticationError(SAPRPTError):
    """Raised when authentication fails."""


class SAPRPTRateLimitError(SAPRPTError):
    """Raised when API rate limiting occurs."""


class SAPRPTClient:
    """HTTP client for interacting with SAP RPT cloud predictions."""

    def __init__(
        self,
        api_token: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: int = 60,
        session: Optional[requests.Session] = None,
    ):
        self.api_token = api_token or os.getenv("RPT_API_TOKEN")
        if not self.api_token:
            raise RuntimeError(
                "SAP RPT API token not configured. Set RPT_API_TOKEN in the environment."
            )
        self.base_url = (base_url or os.getenv("RPT_API_BASE_URL", "https://rpt.cloud.sap")).rstrip("/")
        self.timeout = timeout
        self.session = session or requests.Session()

    def predict(
        self,
        rows: List[Dict[str, Any]],
        index_column: Optional[str] = None,
        parameters: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"rows": rows}
        if index_column:
            payload["index_column"] = index_column
        if parameters:
            payload["parameters"] = dict(parameters)
        return self._post("/api/predict", payload)

    def _post(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        url = f"{self.base_url}{path}"
        headers = {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json",
        }
        response = self.session.post(url, headers=headers, json=payload, timeout=self.timeout)
        if response.status_code >= 400:
            self._raise_api_error(response)
        try:
            return response.json()
        except ValueError as exc:  # pragma: no cover - unexpected non-JSON response
            raise SAPRPTError("Invalid JSON response from SAP RPT API", response.status_code) from exc

    def _raise_api_error(self, response: requests.Response) -> None:
        retry_after = response.headers.get("Retry-After")
        message = response.text or response.reason
        try:
            payload = response.json()
            message = payload.get("message") or payload.get("error") or message
        except ValueError:
            pass

        if response.status_code == 401:
            raise SAPRPTAuthenticationError(message, response.status_code, retry_after)
        if response.status_code in (429, 503):
            raise SAPRPTRateLimitError(message, response.status_code, retry_after)

        raise SAPRPTError(message, response.status_code, retry_after)


_sap_client: Optional[SAPRPTClient] = None

__all__ = [
    "mcp",
    "load_reference_dataset",
    "list_available_datasets",
    "get_dataset_schema",
    "get_dataset_sample",
    "predict_tabular",
    "initialize_reference_datasets",
    "set_rpt_client",
    "get_rpt_client",
    "PREDICT_TOKEN",
    "MAX_CONTEXT_ROWS",
]


def get_rpt_client() -> SAPRPTClient:
    global _sap_client
    if _sap_client is None:
        _sap_client = SAPRPTClient()
    return _sap_client


def set_rpt_client(client: Optional[SAPRPTClient]) -> None:
    global _sap_client
    _sap_client = client


def _resolve_reference_dataframe(dataset_id: Optional[str]) -> Optional[pd.DataFrame]:
    if not dataset_id:
        return None
    info = _reference_data_cache.get(dataset_id)
    if info is None:
        raise KeyError(dataset_id)
    return info["dataframe"]


def _rows_from_json(rows_json: str) -> pd.DataFrame:
    payload = json.loads(rows_json)
    if isinstance(payload, Mapping):
        payload = payload.get("rows") or payload.get("data") or payload
    if isinstance(payload, Sequence) and not isinstance(payload, (str, bytes)):
        return pd.DataFrame(list(payload))
    raise ValueError("rows_json must encode a list of row dictionaries.")


def _chunk_dataframe(df: pd.DataFrame, size: int) -> Sequence[pd.DataFrame]:
    if len(df) == 0:
        return []
    if size <= 0:
        return [df]
    return [df.iloc[start : start + size] for start in range(0, len(df), size)]


def _sanitize_record(record: Dict[str, Any]) -> Dict[str, Any]:
    sanitized: Dict[str, Any] = {}
    for key, value in record.items():
        if value is None:
            sanitized[key] = PREDICT_TOKEN
            continue
        if isinstance(value, str):
            sanitized[key] = value if value.strip() else PREDICT_TOKEN
            continue
        if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
            sanitized[key] = PREDICT_TOKEN
            continue
        if pd.isna(value):
            sanitized[key] = PREDICT_TOKEN
            continue
        sanitized[key] = value
    return sanitized


def _build_prediction_payload(
    context_records: List[Dict[str, Any]],
    query_records: List[Dict[str, Any]],
    query_offset: int,
    row_id_field: str,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    rows: List[Dict[str, Any]] = []
    query_ids: List[str] = []

    for idx, record in enumerate(context_records):
        row = _sanitize_record(record)
        value = row.get(row_id_field)
        if value is None or (isinstance(value, str) and not value.strip()):
            row[row_id_field] = f"context-{idx}"
        rows.append(row)

    for idx, record in enumerate(query_records):
        row = _sanitize_record(record)
        value = row.get(row_id_field)
        if value is None or (isinstance(value, str) and not value.strip()):
            row_id = f"query-{query_offset + idx}"
        else:
            row_id = str(value)
        row[row_id_field] = row_id
        rows.append(row)
        query_ids.append(row_id)

    return rows, query_ids


def _decode_prediction_cell(value: Any) -> Any:
    if isinstance(value, list) and value:
        top = value[0]
        if isinstance(top, dict):
            return top.get("prediction")
        return top
    if isinstance(value, dict) and "prediction" in value:
        return value.get("prediction")
    return value


def _extract_predictions(report: Dict[str, Any], row_id_field: str) -> Dict[str, Dict[str, Any]]:
    prediction_block = report.get("prediction", {})
    entries = prediction_block.get("predictions", []) if isinstance(prediction_block, dict) else []
    mapping: Dict[str, Dict[str, Any]] = {}

    for entry in entries:
        if not isinstance(entry, dict):
            continue
        row_id = entry.get(row_id_field)
        if row_id is None:
            continue
        decoded: Dict[str, Any] = {}
        for column, value in entry.items():
            if column == row_id_field:
                continue
            decoded[column] = _decode_prediction_cell(value)
        mapping[str(row_id)] = decoded

    return mapping


def _invoke_prediction_pipeline(
    reference_df: Optional[pd.DataFrame],
    query_df: pd.DataFrame,
    *,
    row_id_field: str,
    context_rows_limit: int,
    client: Optional[SAPRPTClient] = None,
    parameters: Optional[Mapping[str, Any]] = None,
    chunk_size: int = MAX_CONTEXT_ROWS,
) -> Tuple[List[Dict[str, Any]], float]:
    if query_df.empty:
        return [], 0.0

    limit = max(0, min(context_rows_limit, MAX_CONTEXT_ROWS))
    context_records: List[Dict[str, Any]] = []
    if reference_df is not None and not reference_df.empty and limit:
        context_records = reference_df.head(limit).to_dict(orient="records")

    client = client or get_rpt_client()

    predictions: List[Dict[str, Any]] = []
    total_delay = 0.0
    query_offset = 0

    for chunk_df in _chunk_dataframe(query_df, max(1, chunk_size)):
        query_records = chunk_df.to_dict(orient="records")
        rows, query_ids = _build_prediction_payload(
            context_records,
            query_records,
            query_offset,
            row_id_field,
        )
        report = client.predict(rows=rows, index_column=row_id_field, parameters=parameters)

        delay = report.get("delay")
        if isinstance(delay, (int, float)):
            total_delay += float(delay)

        mapping = _extract_predictions(report, row_id_field)
        for row_id in query_ids:
            predictions.append(mapping.get(row_id, {}))
        query_offset += len(query_ids)

    return predictions, total_delay


def load_reference_dataset(dataset_id: str, filepath: str):
    """
    Load reference dataset at server initialization.

    Args:
        dataset_id: Unique identifier for this dataset
        filepath: Path to the data file (CSV, Parquet, Feather, etc.)
    """
    file_ext = Path(filepath).suffix.lower()

    if file_ext == ".parquet":
        df = pd.read_parquet(filepath)
    elif file_ext == ".feather":
        df = pd.read_feather(filepath)
    elif file_ext == ".csv":
        df = pd.read_csv(filepath)
    else:
        raise ValueError(f"Unsupported file format: {file_ext}")

    _reference_data_cache[dataset_id] = {
        "dataframe": df,
        "filepath": filepath,
        "shape": df.shape,
        "columns": list(df.columns),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
    }

    logger.info("Loaded dataset '%s': %s rows, %s columns", dataset_id, df.shape[0], df.shape[1])
    return df


# ============================================================================
# MCP RESOURCES - Dataset Discovery and Metadata
# ============================================================================

def list_available_datasets() -> str:
    """
    Lists all available reference datasets loaded in the server.

    Returns:
        JSON string with dataset metadata including ID, shape, columns, and types
    """
    datasets = []
    for dataset_id, info in _reference_data_cache.items():
        datasets.append(
            {
                "id": dataset_id,
                "rows": info["shape"][0],
                "columns": info["shape"][1],
                "column_names": info["columns"],
                "filepath": info["filepath"],
            }
        )

    return json.dumps({"datasets": datasets, "count": len(datasets)}, indent=2)


def get_dataset_schema(dataset_id: str) -> str:
    """
    Get detailed schema information for a specific dataset.
    """
    if dataset_id not in _reference_data_cache:
        return json.dumps({"error": f"Dataset '{dataset_id}' not found"})

    info = _reference_data_cache[dataset_id]
    df = info["dataframe"]

    schema = {
        "dataset_id": dataset_id,
        "shape": {"rows": info["shape"][0], "columns": info["shape"][1]},
        "columns": {},
    }

    for col in df.columns:
        schema["columns"][col] = {
            "dtype": str(df[col].dtype),
            "non_null_count": int(df[col].count()),
            "null_count": int(df[col].isnull().sum()),
            "unique_values": int(df[col].nunique()),
            "sample_values": df[col].dropna().head(5).tolist() if df[col].notna().any() else [],
        }

    return json.dumps(schema, indent=2)


def get_dataset_sample(dataset_id: str, n_rows: int = 10) -> str:
    """
    Get a sample of rows from the dataset.
    """
    if dataset_id not in _reference_data_cache:
        return json.dumps({"error": f"Dataset '{dataset_id}' not found"})

    df = _reference_data_cache[dataset_id]["dataframe"]
    sample_df = df.head(n_rows)

    return json.dumps(
        {
            "dataset_id": dataset_id,
            "sample_size": len(sample_df),
            "data": sample_df.to_dict(orient="records"),
        },
        indent=2,
        default=str,
    )


def predict_tabular(
    dataset_id: Optional[str] = None,
    *,
    rows_json: str,
    index_column: Optional[str] = None,
    context_rows: Optional[int] = None,
    max_rows: Optional[int] = None,
    max_context_size: Optional[int] = None,
    bagging: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Unified prediction tool that proxies directly to SAP's /api/predict endpoint.

    Args:
        dataset_id: Optional reference dataset identifier for context rows.
        rows_json: JSON-encoded list of query rows.
        index_column: Column used as the SAP index (defaults to an internal __row_id).
        context_rows: Cap on how many reference rows to attach (<= MAX_CONTEXT_ROWS, default 2048). Most callers should use the default so SAP receives the richest possible context.
        max_rows: Optional limit on the number of query rows processed.
        max_context_size: Parameter forwarded to SAP (if provided).
        bagging: Parameter forwarded to SAP (if provided).

    Guidance for callers:
        * Always include the full schema. Do NOT drop columns even if every value is unknown.
        * Represent missing values as actual JSON nulls or Python None (e.g., {"Attrition": null}). Do NOT emit the literal string "null" or remove the column.
        * Empty strings, whitespace-only strings, NaN, or None are all acceptable ways to mark unknown cells.
    """
    try:
        query_df = _rows_from_json(rows_json)
    except (ValueError, json.JSONDecodeError) as exc:
        return {"error": f"Failed to parse rows_json: {exc}"}

    if max_rows and max_rows > 0:
        query_df = query_df.head(max_rows)

    if query_df.empty:
        return {
            "dataset_id": dataset_id,
            "predictions": [],
            "num_predictions": 0,
            "context_rows_included": 0,
            "delay_seconds": 0.0,
            "request_metadata": {
                "index_column": index_column or DEFAULT_ROW_ID_FIELD,
                "context_rows_requested": context_rows or 0,
            },
        }

    reference_df = None
    if dataset_id:
        try:
            reference_df = _resolve_reference_dataframe(dataset_id)
        except KeyError:
            return {
                "error": f"Dataset '{dataset_id}' not found",
                "available_datasets": list(_reference_data_cache.keys()),
            }

    requested_context_rows = MAX_CONTEXT_ROWS if context_rows is None else context_rows
    context_limit = max(0, min(int(requested_context_rows), MAX_CONTEXT_ROWS))
    context_rows_included = min(len(reference_df), context_limit) if reference_df is not None else 0

    row_id_field = (index_column or DEFAULT_ROW_ID_FIELD).strip() or DEFAULT_ROW_ID_FIELD

    parameters: Dict[str, Any] = {}
    if max_context_size is not None:
        parameters["max_context_size"] = max_context_size
    if bagging is not None:
        parameters["bagging"] = bagging

    try:
        predictions, total_delay = _invoke_prediction_pipeline(
            reference_df,
            query_df,
            row_id_field=row_id_field,
            context_rows_limit=context_limit,
            client=get_rpt_client(),
            parameters=parameters or None,
        )
    except (SAPRPTError, RuntimeError) as exc:
        logger.error("Prediction failed: %s", exc)
        payload: Dict[str, Any] = {
            "error": f"Prediction failed: {exc}",
            "error_type": type(exc).__name__,
        }
        if isinstance(exc, SAPRPTError):
            payload["status_code"] = exc.status_code
            if exc.retry_after:
                payload["retry_after"] = exc.retry_after
        return payload

    request_metadata: Dict[str, Any] = {
        "index_column": row_id_field,
        "context_rows_included": context_rows_included,
    }
    if bagging is not None:
        request_metadata["bagging"] = bagging
    if max_context_size is not None:
        request_metadata["max_context_size"] = max_context_size
    request_metadata["context_rows_requested"] = requested_context_rows

    return {
        "dataset_id": dataset_id,
        "predictions": predictions,
        "num_predictions": len(predictions),
        "context_rows_included": context_rows_included,
        "delay_seconds": total_delay,
        "request_metadata": request_metadata,
    }


# ============================================================================
# SERVER INITIALIZATION AND STARTUP
# ============================================================================

def initialize_reference_datasets(dataset_map: Optional[Mapping[str, Mapping[str, Any]]] = None) -> None:
    """Load reference datasets defined by dataset_id -> path mappings."""
    if not dataset_map:
        logger.info("No reference datasets configured; running in query-only mode.")
        return

    for dataset_id, info in dataset_map.items():
        if not dataset_id:
            raise ValueError("Dataset IDs must be non-empty strings.")
        if not isinstance(info, Mapping):
            raise ValueError(f"Dataset config for '{dataset_id}' must be a mapping.")

        path = info.get("path")
        if not path:
            raise ValueError(f"Dataset '{dataset_id}' is missing a 'path' entry.")

        load_reference_dataset(dataset_id=dataset_id, filepath=str(path))

    logger.info("Loaded %s reference datasets", len(_reference_data_cache))


# Register MCP resources and tools without shadowing underlying callables
mcp.resource("datasets://available")(list_available_datasets)
mcp.resource("datasets://{dataset_id}/schema")(get_dataset_schema)
mcp.resource("datasets://{dataset_id}/sample")(get_dataset_sample)

mcp.tool()(predict_tabular)
