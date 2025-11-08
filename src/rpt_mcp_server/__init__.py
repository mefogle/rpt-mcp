"""rpt-mcp server package."""

from .server import (
    clear_model_cache,
    get_rpt_client,
    get_dataset_sample,
    get_dataset_schema,
    initialize_reference_datasets,
    list_available_datasets,
    list_cached_models,
    load_reference_dataset,
    main,
    mcp,
    predict_batch_from_file,
    predict_classification,
    predict_regression,
    set_rpt_client,
)

__all__ = [
    "clear_model_cache",
    "get_rpt_client",
    "get_dataset_sample",
    "get_dataset_schema",
    "initialize_reference_datasets",
    "list_available_datasets",
    "list_cached_models",
    "load_reference_dataset",
    "main",
    "mcp",
    "predict_batch_from_file",
    "predict_classification",
    "predict_regression",
    "set_rpt_client",
]
