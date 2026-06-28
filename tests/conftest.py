"""
conftest.py — Session-scoped fixture that starts the target FastAPI app
on a random free port, runs all tests, then shuts it down.
"""

from __future__ import annotations

import socket
import subprocess
import sys
import time
import os
from pathlib import Path

import pytest
import requests


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="session")
def api_base_url() -> str:  # type: ignore[return]
    """Start the FastAPI target app and return its base URL."""
    port = _find_free_port()
    target_dir = Path(__file__).parent.parent / "target_api"
    proc = subprocess.Popen(
        [
            sys.executable, "-m", "uvicorn",
            "main:app",
            "--host", "127.0.0.1",
            "--port", str(port),
            "--log-level", "warning",
        ],
        cwd=str(target_dir),
    )

    base_url = f"http://127.0.0.1:{port}"
    # Wait until server is ready (max 10 s)
    for _ in range(40):
        try:
            requests.get(f"{base_url}/health", timeout=0.5)
            break
        except Exception:
            time.sleep(0.25)
    else:
        proc.terminate()
        pytest.fail("FastAPI server failed to start in time")

    yield base_url

    proc.terminate()
    proc.wait(timeout=5)