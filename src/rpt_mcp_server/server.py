from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

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

# Global caches for models and reference datasets
_model_cache = {}
_reference_data_cache = {}

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PREDICT_TOKEN = "[PREDICT]"
INDEX_COLUMN = "__row_id"
MAX_CONTEXT_ROWS = 2048
MAX_QUERY_ROWS = 25


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

    def predict(self, rows: List[Dict[str, object]], index_column: Optional[str] = None) -> Dict[str, object]:
        payload: Dict[str, object] = {"rows": rows}
        if index_column:
            payload["index_column"] = index_column
        return self._post("/api/predict", payload)

    def _post(self, path: str, payload: Dict[str, object]) -> Dict[str, object]:
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
    "list_cached_models",
    "predict_classification",
    "predict_regression",
    "predict_batch_from_file",
    "clear_model_cache",
    "initialize_reference_datasets",
    "main",
    "set_rpt_client",
]


def get_rpt_client() -> SAPRPTClient:
    global _sap_client
    if _sap_client is None:
        _sap_client = SAPRPTClient()
    return _sap_client


def set_rpt_client(client: Optional[SAPRPTClient]) -> None:
    global _sap_client
    _sap_client = client


def _select_context_rows(reference_df: pd.DataFrame, target_column: str) -> pd.DataFrame:
    if target_column not in reference_df.columns:
        raise ValueError(f"Target column '{target_column}' not found in reference dataset")
    context_df = reference_df.dropna(subset=[target_column])
    if len(context_df) < 2:
        raise ValueError("At least two context rows with known target values are required.")
    return context_df.head(MAX_CONTEXT_ROWS)


def _chunk_dataframe(df: pd.DataFrame, size: int) -> Sequence[pd.DataFrame]:
    if len(df) == 0:
        return []
    return [df.iloc[start : start + size] for start in range(0, len(df), size)]


def _build_prediction_payload(
    context_records: List[Dict[str, object]],
    query_records: List[Dict[str, object]],
    target_columns: Sequence[str],
    query_offset: int,
) -> Tuple[List[Dict[str, object]], List[str]]:
    rows: List[Dict[str, object]] = []
    query_ids: List[str] = []

    for idx, record in enumerate(context_records):
        row = dict(record)
        row[INDEX_COLUMN] = f"context-{idx}"
        rows.append(row)

    for idx, record in enumerate(query_records):
        row_id = f"query-{query_offset + idx}"
        row = dict(record)
        row[INDEX_COLUMN] = row_id
        for column in target_columns:
            row[column] = PREDICT_TOKEN
        rows.append(row)
        query_ids.append(row_id)

    return rows, query_ids


def _extract_predictions(report: Dict[str, object], target_column: str) -> Dict[str, object]:
    prediction_block = report.get("prediction", {})
    entries = prediction_block.get("predictions", []) if isinstance(prediction_block, dict) else []
    mapping: Dict[str, object] = {}

    for entry in entries:
        if not isinstance(entry, dict):
            continue
        row_id = entry.get(INDEX_COLUMN)
        column_predictions = entry.get(target_column, [])
        value = None
        if isinstance(column_predictions, list) and column_predictions:
            top_prediction = column_predictions[0]
            if isinstance(top_prediction, dict):
                value = top_prediction.get("prediction")
            else:
                value = top_prediction
        if row_id:
            mapping[row_id] = value

    return mapping


def _invoke_prediction_pipeline(
    reference_df: pd.DataFrame,
    query_df: pd.DataFrame,
    target_column: str,
    client: Optional[SAPRPTClient] = None,
) -> Tuple[List[object], float]:
    if query_df.empty:
        return [], 0.0

    context_df = _select_context_rows(reference_df, target_column)
    context_records = context_df.to_dict(orient="records")
    client = client or get_rpt_client()

    predictions: List[object] = []
    total_delay = 0.0
    query_offset = 0

    for chunk_df in _chunk_dataframe(query_df, MAX_QUERY_ROWS):
        query_records = chunk_df.to_dict(orient="records")
        rows, query_ids = _build_prediction_payload(context_records, query_records, [target_column], query_offset)
        report = client.predict(rows=rows, index_column=INDEX_COLUMN)

        delay = report.get("delay")
        if isinstance(delay, (int, float)):
            total_delay += float(delay)

        mapping = _extract_predictions(report, target_column)
        for row_id in query_ids:
            predictions.append(mapping.get(row_id))
        query_offset += len(query_ids)

    return predictions, total_delay


def load_reference_dataset(dataset_id: str, filepath: str, target_column: Optional[str] = None):
    """
    Load reference dataset at server initialization.
    
    Args:
        dataset_id: Unique identifier for this dataset
        filepath: Path to the data file (CSV, Parquet, Feather, etc.)
        target_column: Optional target column name for automatic splitting
    """
    file_ext = Path(filepath).suffix.lower()
    
    if file_ext == '.parquet':
        df = pd.read_parquet(filepath)
    elif file_ext == '.feather':
        df = pd.read_feather(filepath)
    elif file_ext == '.csv':
        df = pd.read_csv(filepath)
    else:
        raise ValueError(f"Unsupported file format: {file_ext}")
    
    _reference_data_cache[dataset_id] = {
        'dataframe': df,
        'target_column': target_column,
        'filepath': filepath,
        'shape': df.shape,
        'columns': list(df.columns),
        'dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()}
    }
    
    logger.info(f"Loaded dataset '{dataset_id}': {df.shape[0]} rows, {df.shape[1]} columns")
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
        datasets.append({
            "id": dataset_id,
            "rows": info['shape'][0],
            "columns": info['shape'][1],
            "column_names": info['columns'],
            "target_column": info.get('target_column'),
            "filepath": info['filepath']
        })
    
    return json.dumps({"datasets": datasets, "count": len(datasets)}, indent=2)


def get_dataset_schema(dataset_id: str) -> str:
    """
    Get detailed schema information for a specific dataset.
    
    Args:
        dataset_id: The ID of the dataset
        
    Returns:
        JSON string with column-level metadata including types, null counts, unique values
    """
    if dataset_id not in _reference_data_cache:
        return json.dumps({"error": f"Dataset '{dataset_id}' not found"})
    
    info = _reference_data_cache[dataset_id]
    df = info['dataframe']
    
    schema = {
        "dataset_id": dataset_id,
        "shape": {"rows": info['shape'][0], "columns": info['shape'][1]},
        "target_column": info.get('target_column'),
        "columns": {}
    }
    
    for col in df.columns:
        schema["columns"][col] = {
            "dtype": str(df[col].dtype),
            "non_null_count": int(df[col].count()),
            "null_count": int(df[col].isnull().sum()),
            "unique_values": int(df[col].nunique()),
            "sample_values": df[col].dropna().head(5).tolist() if len(df[col].dropna()) > 0 else []
        }
    
    return json.dumps(schema, indent=2)


def get_dataset_sample(dataset_id: str, n_rows: int = 10) -> str:
    """
    Get a sample of rows from the dataset.
    
    Args:
        dataset_id: The ID of the dataset
        n_rows: Number of sample rows to return (default: 10)
        
    Returns:
        JSON string with sample data
    """
    if dataset_id not in _reference_data_cache:
        return json.dumps({"error": f"Dataset '{dataset_id}' not found"})
    
    df = _reference_data_cache[dataset_id]['dataframe']
    sample_df = df.head(n_rows)
    
    return json.dumps({
        "dataset_id": dataset_id,
        "sample_size": len(sample_df),
        "data": sample_df.to_dict(orient='records')
    }, indent=2, default=str)


def list_cached_models() -> str:
    """
    Lists all currently cached (fitted) models in memory.
    
    Returns:
        JSON string with information about cached models
    """
    models = []
    for model_key, model_info in _model_cache.items():
        models.append({
            "key": model_key,
            "type": model_info['type'],
            "dataset_id": model_info['dataset_id'],
            "target_column": model_info['target_column'],
            "config": model_info['config']
        })
    
    return json.dumps({"cached_models": models, "count": len(models)}, indent=2)


# ============================================================================
# MCP TOOLS - Prediction Operations
# ============================================================================

def predict_classification(
    dataset_id: str,
    target_column: str,
    test_data_json: str,
    max_context_size: int = 8192,
    bagging: int = 8,
    return_probabilities: bool = True
) -> dict:
    """
    Predict categorical values in tabular data via the SAP RPT cloud API.
    """
    try:
        if dataset_id not in _reference_data_cache:
            return {
                "error": f"Dataset '{dataset_id}' not found",
                "available_datasets": list(_reference_data_cache.keys())
            }

        info = _reference_data_cache[dataset_id]
        reference_df = info['dataframe']

        if target_column not in reference_df.columns:
            return {
                "error": f"Target column '{target_column}' not found in dataset",
                "available_columns": list(reference_df.columns)
            }

        try:
            test_data = json.loads(test_data_json)
            if isinstance(test_data, dict) and 'data' in test_data:
                test_data = test_data['data']
            test_df = pd.DataFrame(test_data)
        except Exception as exc:
            return {"error": f"Failed to parse test_data_json: {exc}"}

        features_df = reference_df.drop(columns=[target_column])
        test_features = test_df.drop(columns=[target_column], errors='ignore')
        missing_cols = set(features_df.columns) - set(test_features.columns)
        if missing_cols:
            return {
                "error": f"Test data is missing columns: {list(missing_cols)}",
                "required_columns": list(features_df.columns)
            }

        if target_column not in test_df.columns:
            test_df[target_column] = None

        client = get_rpt_client()
        predictions, _ = _invoke_prediction_pipeline(reference_df, test_df, target_column, client=client)
        predictions_list = list(predictions)

        model_key = f"{dataset_id}_{target_column}_clf_{max_context_size}_{bagging}"
        api_base_url = getattr(client, "base_url", "sap-rpt-api")
        _model_cache[model_key] = {
            'model': 'sap-rpt-api',
            'type': 'classifier',
            'dataset_id': dataset_id,
            'target_column': target_column,
            'config': {
                'max_context_size': max_context_size,
                'bagging': bagging,
                'api_base_url': api_base_url,
                'context_rows': min(len(reference_df), MAX_CONTEXT_ROWS),
            }
        }

        result = {
            "predictions": predictions_list,
            "target_column": target_column,
            "num_predictions": len(predictions_list),
            "model_config": _model_cache[model_key]['config'],
        }

        if return_probabilities:
            probability_payload = []
            classes = set()
            for prediction in predictions_list:
                if prediction is None:
                    probability_payload.append([])
                    continue
                classes.add(prediction)
                probability_payload.append([{"prediction": prediction, "probability": 1.0}])
            result["probabilities"] = probability_payload
            result["classes"] = sorted(classes)

        return result

    except (SAPRPTError, RuntimeError) as exc:
        logger.error("Classification prediction failed: %s", exc)
        payload = {
            "error": f"Prediction failed: {exc}",
            "error_type": type(exc).__name__,
        }
        if isinstance(exc, SAPRPTError):
            payload["status_code"] = exc.status_code
            if exc.retry_after:
                payload["retry_after"] = exc.retry_after
        return payload


def predict_regression(
    dataset_id: str,
    target_column: str,
    test_data_json: str,
    max_context_size: int = 8192,
    bagging: int = 8
) -> dict:
    """
    Predict numerical target values via the SAP RPT cloud prediction API.
    """
    try:
        if dataset_id not in _reference_data_cache:
            return {
                "error": f"Dataset '{dataset_id}' not found",
                "available_datasets": list(_reference_data_cache.keys())
            }

        info = _reference_data_cache[dataset_id]
        reference_df = info['dataframe']

        if target_column not in reference_df.columns:
            return {
                "error": f"Target column '{target_column}' not found in dataset",
                "available_columns": list(reference_df.columns)
            }

        try:
            test_data = json.loads(test_data_json)
            if isinstance(test_data, dict) and 'data' in test_data:
                test_data = test_data['data']
            test_df = pd.DataFrame(test_data)
        except Exception as exc:
            return {"error": f"Failed to parse test_data_json: {exc}"}

        features_df = reference_df.drop(columns=[target_column])
        test_features = test_df.drop(columns=[target_column], errors='ignore')
        missing_cols = set(features_df.columns) - set(test_features.columns)
        if missing_cols:
            return {
                "error": f"Test data is missing columns: {list(missing_cols)}",
                "required_columns": list(features_df.columns)
            }

        if target_column not in test_df.columns:
            test_df[target_column] = None

        client = get_rpt_client()
        predictions, _ = _invoke_prediction_pipeline(reference_df, test_df, target_column, client=client)
        predictions_list = []
        numeric_values = []
        for prediction in predictions:
            if prediction is None:
                predictions_list.append(None)
                continue
            try:
                numeric_value = float(prediction)
                predictions_list.append(numeric_value)
                numeric_values.append(numeric_value)
            except (TypeError, ValueError):
                predictions_list.append(prediction)

        statistics = {}
        if numeric_values:
            pred_array = np.array(numeric_values, dtype=float)
            statistics = {
                "mean": float(np.mean(pred_array)),
                "std": float(np.std(pred_array)),
                "min": float(np.min(pred_array)),
                "max": float(np.max(pred_array)),
                "median": float(np.median(pred_array)),
            }
        else:
            statistics = {"mean": None, "std": None, "min": None, "max": None, "median": None}

        model_key = f"{dataset_id}_{target_column}_reg_{max_context_size}_{bagging}"
        api_base_url = getattr(client, "base_url", "sap-rpt-api")
        _model_cache[model_key] = {
            'model': 'sap-rpt-api',
            'type': 'regressor',
            'dataset_id': dataset_id,
            'target_column': target_column,
            'config': {
                'max_context_size': max_context_size,
                'bagging': bagging,
                'api_base_url': api_base_url,
                'context_rows': min(len(reference_df), MAX_CONTEXT_ROWS),
            }
        }

        return {
            "predictions": predictions_list,
            "target_column": target_column,
            "num_predictions": len(predictions_list),
            "statistics": statistics,
            "model_config": _model_cache[model_key]['config'],
        }

    except (SAPRPTError, RuntimeError) as exc:
        logger.error("Regression prediction failed: %s", exc)
        payload = {
            "error": f"Prediction failed: {exc}",
            "error_type": type(exc).__name__,
        }
        if isinstance(exc, SAPRPTError):
            payload["status_code"] = exc.status_code
            if exc.retry_after:
                payload["retry_after"] = exc.retry_after
        return payload


def predict_batch_from_file(
    dataset_id: str,
    target_column: str,
    input_file_path: str,
    output_file_path: str,
    task_type: str = "classification",
    max_context_size: int = 8192,
    bagging: int = 8
) -> dict:
    """
    Process large test files directly without passing data through JSON.
    
    This is more efficient for large datasets as it avoids JSON serialization overhead.
    
    Args:
        dataset_id: ID of the reference dataset
        target_column: Column to predict
        input_file_path: Path to input CSV/Parquet file with test data
        output_file_path: Path where predictions should be saved
        task_type: Either "classification" or "regression"
        max_context_size: Maximum context size (default: 8192)
        bagging: Bagging factor (default: 8)
    
    Returns:
        Dictionary with processing status and output file path
    """
    try:
        if dataset_id not in _reference_data_cache:
            return {
                "error": f"Dataset '{dataset_id}' not found",
                "available_datasets": list(_reference_data_cache.keys())
            }

        input_path = Path(input_file_path)
        if not input_path.exists():
            return {"error": f"Input file not found: {input_file_path}"}

        if input_path.suffix.lower() == '.csv':
            test_df = pd.read_csv(input_file_path)
        elif input_path.suffix.lower() == '.parquet':
            test_df = pd.read_parquet(input_file_path)
        else:
            return {"error": f"Unsupported file format: {input_path.suffix}"}

        info = _reference_data_cache[dataset_id]
        reference_df = info['dataframe']

        features_df = reference_df.drop(columns=[target_column])
        test_features = test_df.drop(columns=[target_column], errors='ignore')
        missing_cols = set(features_df.columns) - set(test_features.columns)
        if missing_cols:
            return {
                "error": f"Test data is missing columns: {list(missing_cols)}",
                "required_columns": list(features_df.columns)
            }

        if target_column not in test_df.columns:
            test_df[target_column] = None

        client = get_rpt_client()
        predictions, _ = _invoke_prediction_pipeline(reference_df, test_df, target_column, client=client)

        if len(predictions) != len(test_df):
            return {"error": "Prediction count does not match input rows."}

        result_df = test_df.copy()
        result_df[f"{target_column}_predicted"] = predictions

        output_path = Path(output_file_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if output_path.suffix.lower() == '.csv':
            result_df.to_csv(output_file_path, index=False)
        elif output_path.suffix.lower() == '.parquet':
            result_df.to_parquet(output_file_path, index=False)
        else:
            return {"error": f"Unsupported output format: {output_path.suffix}"}

        model_key = f"{dataset_id}_{target_column}_{task_type[:3]}_{max_context_size}_{bagging}"
        api_base_url = getattr(client, "base_url", "sap-rpt-api")
        _model_cache[model_key] = {
            'model': 'sap-rpt-api',
            'type': task_type,
            'dataset_id': dataset_id,
            'target_column': target_column,
            'config': {
                'max_context_size': max_context_size,
                'bagging': bagging,
                'api_base_url': api_base_url,
                'context_rows': min(len(reference_df), MAX_CONTEXT_ROWS),
            }
        }

        return {
            "status": "success",
            "input_file": input_file_path,
            "output_file": output_file_path,
            "num_predictions": len(predictions),
            "task_type": task_type
        }

    except (SAPRPTError, RuntimeError) as exc:
        logger.error("Batch prediction failed: %s", exc)
        payload = {
            "error": f"Batch prediction failed: {exc}",
            "error_type": type(exc).__name__,
        }
        if isinstance(exc, SAPRPTError):
            payload["status_code"] = exc.status_code
            if exc.retry_after:
                payload["retry_after"] = exc.retry_after
        return payload


def clear_model_cache(model_key: Optional[str] = None) -> dict:
    """
    Clear cached models from memory.
    
    Args:
        model_key: Specific model key to clear, or None to clear all models
    
    Returns:
        Dictionary with clearance status
    """
    global _model_cache
    
    if model_key:
        if model_key in _model_cache:
            del _model_cache[model_key]
            return {"status": "success", "cleared": model_key}
        else:
            return {"error": f"Model key '{model_key}' not found in cache"}
    else:
        num_cleared = len(_model_cache)
        _model_cache = {}
        return {"status": "success", "cleared_count": num_cleared}


# Register MCP resources and tools without shadowing underlying callables
mcp.resource("datasets://available")(list_available_datasets)
mcp.resource("datasets://{dataset_id}/schema")(get_dataset_schema)
mcp.resource("datasets://{dataset_id}/sample")(get_dataset_sample)
mcp.resource("models://cached")(list_cached_models)

mcp.tool()(predict_classification)
mcp.tool()(predict_regression)
mcp.tool()(predict_batch_from_file)
mcp.tool()(clear_model_cache)


# ============================================================================
# SERVER INITIALIZATION AND STARTUP
# ============================================================================

def initialize_reference_datasets(reference_data_dir: Optional[str] = None) -> None:
    """
    Load default reference datasets if they exist on disk.

    Args:
        reference_data_dir: Directory containing parquet/csv reference data.
    """
    data_root = Path(reference_data_dir or os.getenv("REFERENCE_DATA_DIR", "/mnt/dgx/data/reference"))

    try:
        sales_path = data_root / "sales_history_2020_2024.parquet"
        if sales_path.exists():
            load_reference_dataset(
                dataset_id="sales_history",
                filepath=str(sales_path),
                target_column="revenue"
            )

        churn_path = data_root / "customer_churn.parquet"
        if churn_path.exists():
            load_reference_dataset(
                dataset_id="customer_churn",
                filepath=str(churn_path),
                target_column="churned"
            )

        logger.info("Loaded %s reference datasets", len(_reference_data_cache))
    except Exception as exc:
        logger.warning("Failed to load some datasets: %s", exc)


def main() -> None:
    """Entry point for launching the MCP server."""
    initialize_reference_datasets()
    logger.info("Starting SAP RPT-1-OSS MCP Server...")
    mcp.run(transport='stdio')


if __name__ == "__main__":  # pragma: no cover
    main()
