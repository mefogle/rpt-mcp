"""rpt-mcp server package."""

from .cli import main, serve
from .server import (
    get_rpt_client,
    get_dataset_sample,
    get_dataset_schema,
    initialize_reference_datasets,
    list_available_datasets,
    load_reference_dataset,
    mcp,
    predict_tabular,
    set_rpt_client,
)

__all__ = [
    "get_rpt_client",
    "get_dataset_sample",
    "get_dataset_schema",
    "initialize_reference_datasets",
    "list_available_datasets",
    "load_reference_dataset",
    "main",
    "mcp",
    "predict_tabular",
    "serve",
    "set_rpt_client",
]
