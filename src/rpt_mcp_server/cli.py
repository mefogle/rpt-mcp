from __future__ import annotations

import argparse
import inspect
import json
import logging
import os
from typing import Dict, Mapping, Optional, Sequence

from . import server

logger = logging.getLogger(__name__)

DEFAULT_TRANSPORT = os.getenv("RPT_MCP_TRANSPORT", "stdio")
DEFAULT_HOST = os.getenv("RPT_MCP_HOST", "0.0.0.0")
DEFAULT_PORT = int(os.getenv("RPT_MCP_PORT", "8080"))
DEFAULT_SSE_PATH = os.getenv("RPT_MCP_SSE_PATH", "/sse")
DEFAULT_ALLOWED_ORIGINS = os.getenv("RPT_MCP_ALLOWED_ORIGINS", "*")
DATASET_MAP_ENV = os.getenv("RPT_DATASETS")
DEFAULT_ALLOWED_ORIGIN_LIST = None  # initialized after helper definition


def _split_origins(raw: Optional[str]) -> Optional[Sequence[str]]:
    if raw is None:
        return None
    if not raw.strip():
        return None
    if raw.strip() == "*":
        return ["*"]
    return [part.strip() for part in raw.split(",") if part.strip()]


DEFAULT_ALLOWED_ORIGIN_LIST = _split_origins(DEFAULT_ALLOWED_ORIGINS)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the SAP RPT MCP server")
    parser.add_argument(
        "--transport",
        choices=("stdio", "sse"),
        default=DEFAULT_TRANSPORT,
        help="Transport used to expose the MCP server (default: stdio)",
    )
    parser.add_argument(
        "--host",
        default=DEFAULT_HOST,
        help="Host/interface to bind in SSE mode (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help="Port to listen on when transport=sse (default: 8080)",
    )
    parser.add_argument(
        "--sse-path",
        default=DEFAULT_SSE_PATH,
        help="HTTP path that exposes the SSE endpoint (default: /sse)",
    )
    parser.add_argument(
        "--allowed-origins",
        default=DEFAULT_ALLOWED_ORIGIN_LIST,
        nargs="+",
        help="List of allowed CORS origins for SSE clients (default: '*')",
    )
    parser.add_argument(
        "--dataset",
        action="append",
        dest="datasets",
        default=[],
        metavar="ID=PATH",
        help=(
            "Register a dataset by ID and file path (repeatable)."
        ),
    )
    parser.add_argument(
        "--log-level",
        default=os.getenv("RPT_MCP_LOG_LEVEL", "INFO"),
        help="Logging level (default: INFO)",
    )
    return parser.parse_args(argv)


def _accepted_keywords(run_method) -> set[str]:
    try:
        signature = inspect.signature(run_method)
    except (TypeError, ValueError):
        return set()
    accepted = set(signature.parameters.keys())
    if any(param.kind is inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values()):
        accepted.add("**")
    return accepted


def _apply_kwargs(base_kwargs: dict, extra: dict, accepted: set[str]) -> dict:
    result = dict(base_kwargs)
    accepts_any = "**" in accepted
    for key, value in extra.items():
        if value is None:
            continue
        if accepts_any or key in accepted:
            result[key] = value
    return result


def _normalize_dataset_spec(dataset_id: str, value: object) -> Dict[str, str]:
    if isinstance(value, str):
        path = value
    elif isinstance(value, Mapping):
        path = value.get("path")
    else:
        raise ValueError(f"Dataset '{dataset_id}' must map to a path string or an object with a 'path' entry.")

    if not path or not str(path).strip():
        raise ValueError(f"Dataset '{dataset_id}' is missing a valid path.")

    return {"path": str(path).strip()}


def _dataset_map_from_env(raw: Optional[str]) -> Dict[str, Dict[str, str]]:
    if not raw:
        return {}
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Failed to parse RPT_DATASETS JSON: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError("RPT_DATASETS must be a JSON object mapping dataset IDs to paths.")

    mapping: Dict[str, Dict[str, str]] = {}
    for dataset_id, value in payload.items():
        normalized_id = str(dataset_id).strip()
        if not normalized_id:
            raise ValueError("Dataset IDs in RPT_DATASETS must be non-empty strings.")
        mapping[normalized_id] = _normalize_dataset_spec(normalized_id, value)
    return mapping


def _parse_dataset_argument(arg: str) -> tuple[str, Dict[str, str]]:
    for separator in ("=", ":"):
        if separator in arg:
            dataset_id, remainder = arg.split(separator, 1)
            break
    else:
        raise ValueError("Dataset arguments must look like ID=PATH or ID:PATH")

    dataset_id = dataset_id.strip()
    remainder = remainder.strip()
    if not dataset_id or not remainder:
        raise ValueError("Dataset arguments require both an ID and a path.")

    path = remainder.strip()
    if not path:
        raise ValueError("Dataset path portion cannot be empty.")

    return dataset_id, {"path": path}


def _build_dataset_map(dataset_args: Sequence[str], env_payload: Optional[str]) -> Dict[str, Dict[str, str]]:
    dataset_map = _dataset_map_from_env(env_payload)
    for raw in dataset_args:
        dataset_id, spec = _parse_dataset_argument(raw)
        dataset_map[dataset_id] = spec
    return dataset_map


def serve(
    *,
    transport: str = DEFAULT_TRANSPORT,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    sse_path: str = DEFAULT_SSE_PATH,
    allowed_origins: Optional[Sequence[str]] = DEFAULT_ALLOWED_ORIGIN_LIST,
    dataset_map: Optional[Mapping[str, Mapping[str, str]]] = None,
) -> None:
    """Start the MCP server with the requested transport."""
    log_level = logging.getLogger().level
    logger.setLevel(log_level)

    server.initialize_reference_datasets(dataset_map)

    run_kwargs = {"transport": transport}
    if transport == "sse":
        accepted = _accepted_keywords(server.mcp.run)
        origins_value = list(allowed_origins) if allowed_origins else None
        extra = {
            "host": host,
            "port": port,
            "path": sse_path,
        }
        if origins_value is not None:
            if "allowed_origins" in accepted or "**" in accepted:
                extra["allowed_origins"] = origins_value
            else:
                for alias in ("allow_origins", "allowed_origin", "allow_origin"):
                    if alias in accepted:
                        extra[alias] = origins_value
                        break
        run_kwargs = _apply_kwargs(run_kwargs, extra, accepted)

    logger.info("Starting SAP RPT MCP server via transport=%s", transport)
    server.mcp.run(**run_kwargs)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    level = getattr(logging, str(args.log_level).upper(), logging.INFO)
    logging.basicConfig(level=level)

    try:
        dataset_map = _build_dataset_map(args.datasets, DATASET_MAP_ENV)
    except ValueError as exc:
        raise SystemExit(f"Invalid dataset configuration: {exc}") from exc

    serve(
        transport=args.transport,
        host=args.host,
        port=args.port,
        sse_path=args.sse_path,
        allowed_origins=args.allowed_origins,
        dataset_map=dataset_map,
    )
