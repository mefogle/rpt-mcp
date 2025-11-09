import json
from collections import deque
from pathlib import Path

import pandas as pd
import pytest

from rpt_mcp_server import server

PREDICT_TOKEN = server.PREDICT_TOKEN


@pytest.fixture(autouse=True)
def reset_caches():
    server._reference_data_cache.clear()
    server.set_rpt_client(None)
    yield
    server._reference_data_cache.clear()
    server.set_rpt_client(None)


def _write_dataset(tmp_path: Path, dataset_id: str, df: pd.DataFrame) -> Path:
    path = tmp_path / f"{dataset_id}.csv"
    df.to_csv(path, index=False)
    server.load_reference_dataset(dataset_id=dataset_id, filepath=str(path))
    return path


@pytest.fixture
def classification_dataset(tmp_path):
    df = pd.DataFrame(
        {
            "region": ["na", "eu", "apac"],
            "segment": ["enterprise", "smb", "enterprise"],
            "churned": ["yes", "no", "no"],
        }
    )
    _write_dataset(tmp_path, "classification", df)
    return df


class DummySAPClient:
    def __init__(self):
        self.column_predictions = {}
        self.requests = []

    def set_predictions(self, column: str, values):
        self.column_predictions[column] = deque(values)

    def predict(self, rows, index_column=None, parameters=None):
        predictions = []
        for row in rows:
            row_id = row.get(index_column) if isinstance(row, dict) else None
            if not row_id:
                continue
            target_columns = [column for column, value in row.items() if value == PREDICT_TOKEN]
            if not target_columns:
                continue
            entry = {index_column: row_id} if index_column else {}
            for column in target_columns:
                queue = self.column_predictions.setdefault(column, deque())
                prediction_value = queue.popleft() if queue else f"{column}-pred"
                entry[column] = [{"prediction": prediction_value}]
            predictions.append(entry)
        self.requests.append({"rows": rows, "index_column": index_column, "parameters": parameters})
        return {
            "prediction": {
                "id": "dummy",
                "metadata": {"num_predict_rows": len(predictions)},
                "predictions": predictions,
            },
            "delay": 1,
        }


@pytest.fixture
def mock_rpt_client():
    client = DummySAPClient()
    server.set_rpt_client(client)
    yield client
    server.set_rpt_client(None)


def test_load_reference_dataset_caches_metadata(tmp_path):
    df = pd.DataFrame(
        {
            "feature_a": [1, 2, 3],
            "feature_b": ["x", "y", "z"],
            "label": ["yes", "no", "yes"],
        }
    )
    path = _write_dataset(tmp_path, "ds", df)

    cached = server._reference_data_cache["ds"]
    assert cached["filepath"] == str(path)
    assert cached["shape"] == df.shape
    assert cached["dtypes"]["feature_a"] == str(df["feature_a"].dtype)


def test_list_available_datasets_returns_metadata(classification_dataset):
    payload = json.loads(server.list_available_datasets())
    assert payload["count"] == 1
    dataset_info = payload["datasets"][0]
    assert dataset_info["id"] == "classification"
    assert dataset_info["rows"] == len(classification_dataset)
    assert "churned" in dataset_info["column_names"]


def test_get_dataset_schema_handles_missing_dataset():
    payload = json.loads(server.get_dataset_schema("missing"))
    assert "Dataset 'missing' not found" in payload["error"]


def test_get_dataset_schema_returns_column_stats(classification_dataset):
    payload = json.loads(server.get_dataset_schema("classification"))
    assert payload["shape"]["columns"] == classification_dataset.shape[1]
    region = payload["columns"]["region"]
    assert region["unique_values"] == len(classification_dataset["region"].unique())
    assert region["non_null_count"] == len(classification_dataset)


def test_predict_tabular_allows_query_only(mock_rpt_client):
    mock_rpt_client.set_predictions("churned", ["yes"])
    test_rows = [{"region": "na", "segment": "enterprise", "churned": None}]
    result = server.predict_tabular(rows_json=json.dumps(test_rows))

    assert result["num_predictions"] == 1
    assert result["context_rows_included"] == 0
    assert result["predictions"][0]["churned"] == "yes"


def test_predict_tabular_uses_reference_dataset(classification_dataset, mock_rpt_client):
    mock_rpt_client.set_predictions("churned", ["yes", "no"])
    test_rows = [
        {"region": "na", "segment": "enterprise", "churned": None},
        {"region": "apac", "segment": "enterprise", "churned": None},
    ]
    result = server.predict_tabular(
        dataset_id="classification",
        rows_json=json.dumps(test_rows),
        context_rows=2,
    )

    assert result["num_predictions"] == len(test_rows)
    predictions = result["predictions"]
    assert predictions[0]["churned"] == "yes"
    assert predictions[1]["churned"] == "no"
    assert result["context_rows_included"] == 2
    assert result["request_metadata"]["index_column"] == server.DEFAULT_ROW_ID_FIELD


def test_predict_tabular_errors_when_dataset_missing():
    result = server.predict_tabular(
        dataset_id="unknown",
        rows_json=json.dumps([{"col": 1}]),
    )
    assert "Dataset 'unknown' not found" in result["error"]


def test_predict_tabular_honors_custom_index_column(mock_rpt_client):
    mock_rpt_client.set_predictions("churned", ["yes"])
    test_rows = [{"EmployeeNumber": "42", "region": "na", "churned": None}]
    result = server.predict_tabular(
        rows_json=json.dumps(test_rows),
        index_column="EmployeeNumber",
    )

    assert result["predictions"][0]["churned"] == "yes"
    assert result["request_metadata"]["index_column"] == "EmployeeNumber"


def test_predict_tabular_respects_max_rows(mock_rpt_client):
    mock_rpt_client.set_predictions("churned", ["yes", "no"])
    test_rows = [
        {"region": "na", "churned": None},
        {"region": "eu", "churned": None},
    ]
    result = server.predict_tabular(rows_json=json.dumps(test_rows), max_rows=1)

    assert result["num_predictions"] == 1
    assert len(mock_rpt_client.requests[-1]["rows"]) == 1
