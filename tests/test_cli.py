from __future__ import annotations

import pytest

from rpt_mcp_server import cli


@pytest.fixture(autouse=True)
def reset_logging():
    import logging

    root = logging.getLogger()
    handlers = list(root.handlers)
    for handler in handlers:
        root.removeHandler(handler)
        handler.close()
    yield


def test_serve_stdio_runs_fastmcp(monkeypatch):
    called = {}
    initialized = {}

    def fake_run(*, transport, **kwargs):
        called["transport"] = transport
        called["kwargs"] = kwargs

    def fake_init(mapping):
        initialized["mapping"] = mapping

    monkeypatch.setattr(cli.server, "initialize_reference_datasets", fake_init)
    monkeypatch.setattr(cli.server.mcp, "run", fake_run)

    dataset_map = {"demo": {"path": "/tmp/demo.csv"}}
    cli.serve(transport="stdio", dataset_map=dataset_map)

    assert called["transport"] == "stdio"
    assert called["kwargs"] == {}
    assert initialized["mapping"] == dataset_map


def test_serve_sse_passes_host_port_and_origins(monkeypatch):
    recorded = {}
    initialized = {}

    def fake_run(*, transport, host=None, port=None, path=None, allowed_origins=None):
        recorded.update(
            transport=transport,
            host=host,
            port=port,
            path=path,
            allowed_origins=allowed_origins,
        )

    monkeypatch.setattr(cli.server, "initialize_reference_datasets", lambda mapping: initialized.update(mapping=mapping))
    monkeypatch.setattr(cli.server.mcp, "run", fake_run)

    cli.serve(
        transport="sse",
        host="0.0.0.0",
        port=9090,
        sse_path="/events",
        allowed_origins=["https://client.example"],
        dataset_map={"demo": {"path": "/tmp/demo.csv"}},
    )

    assert recorded["transport"] == "sse"
    assert recorded["host"] == "0.0.0.0"
    assert recorded["port"] == 9090
    assert recorded["path"] == "/events"
    assert recorded["allowed_origins"] == ["https://client.example"]
    assert initialized["mapping"] == {"demo": {"path": "/tmp/demo.csv"}}


def test_main_configures_logging_level(monkeypatch):
    captured = {}

    def fake_serve(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(cli, "serve", fake_serve)

    cli.main([
        "--transport",
        "stdio",
        "--log-level",
        "warning",
        "--dataset",
        "ibm=/tmp/data.csv",
    ])

    assert captured["transport"] == "stdio"
    assert captured["dataset_map"] == {"ibm": {"path": "/tmp/data.csv"}}
