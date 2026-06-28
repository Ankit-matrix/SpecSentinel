"""
test_api.py — pytest integration.

Instead of hard-coding tests, we:
1. Fetch the OpenAPI spec from the running server.
2. Parse it and call the LLM to generate test cases.
3. Parametrize a single pytest test function over every test case.

This means `pytest tests/` always tests against the latest spec.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import uuid
from pathlib import Path

import pytest
import requests

# Make the project root importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from generator import (
    RunLogger,
    RunSummary,
    TestCaseGenerator,
    TestExecutor,
    parse_openapi_spec,
)

log = logging.getLogger(__name__)

# ── Load test cases once per session ─────────────────────────────────────────

def _load_test_cases(base_url: str):
    spec_url = f"{base_url}/openapi.json"
    raw_spec = requests.get(spec_url, timeout=10).json()
    parsed = parse_openapi_spec(raw_spec)

    generator = TestCaseGenerator()
    test_cases = generator.generate(parsed)
    return raw_spec, parsed, test_cases


# ── Pytest parametrize trick: collect at module level via a session-scoped var ─

_BASE_URL_ENV = os.environ.get("API_BASE_URL", "")
_test_cases_cache: list | None = None
_raw_spec_cache: dict | None = None
_parsed_spec_cache: dict | None = None


def pytest_configure(config):
    """Called early — we can't do network calls here safely."""
    pass


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def session_data(api_base_url):
    """Load spec + generate test cases once for the whole session."""
    raw_spec, parsed_spec, test_cases = _load_test_cases(api_base_url)
    return {
        "base_url": api_base_url,
        "raw_spec": raw_spec,
        "parsed_spec": parsed_spec,
        "test_cases": test_cases,
    }


@pytest.fixture(scope="session")
def executor(api_base_url):
    return TestExecutor(base_url=api_base_url)


@pytest.fixture(scope="session")
def all_results(session_data, executor):
    """Execute all test cases and cache results."""
    return executor.run_all(session_data["test_cases"])


# ── Main test: one parametrized item per test case ────────────────────────────

def test_pipeline_generates_cases(session_data):
    """Sanity check: the LLM must produce at least one test case."""
    assert len(session_data["test_cases"]) > 0, "LLM generated zero test cases"


def test_all_cases(session_data, executor, tmp_path):
    """
    Run the full suite, persist logs, and assert pass rate >= 50 %.
    (We deliberately have bugs in the target API so some failures are expected.)
    """
    test_cases = session_data["test_cases"]
    results = executor.run_all(test_cases)

    # Persist results
    run_id = str(uuid.uuid4())[:8]
    summary = RunSummary(
        run_id=run_id,
        base_url=session_data["base_url"],
        spec_title=session_data["raw_spec"].get("info", {}).get("title", "Unknown"),
        spec_version=session_data["raw_spec"].get("info", {}).get("version", "0"),
        total=len(results),
        passed=sum(1 for r in results if r.passed),
        failed=sum(1 for r in results if not r.passed),
        results=results,
    )

    logger = RunLogger(output_dir=str(Path(__file__).parent.parent / "logs"))
    paths = logger.save(summary)
    logger.print_summary(summary)

    print(f"\nLog files written:\n  JSON    → {paths['json']}\n  Parquet → {paths['parquet']}")

    # Store for other tests to inspect
    pytest._run_summary = summary  # type: ignore[attr-defined]
    pytest._log_paths = paths       # type: ignore[attr-defined]

    # Assert pass rate — bugs exist, so we tolerate up to 40% failures
    pass_rate = summary.passed / summary.total if summary.total else 0
    assert pass_rate >= 0.5, (
        f"Pass rate too low: {pass_rate:.0%}. "
        f"{summary.failed}/{summary.total} tests failed."
    )


def test_happy_path_cases_pass(session_data, executor):
    """Happy-path tests should have a very high pass rate (>=80%)."""
    happy = [tc for tc in session_data["test_cases"] if tc.category.value == "happy_path"]
    if not happy:
        pytest.skip("No happy-path test cases generated")

    results = executor.run_all(happy)
    passed = sum(1 for r in results if r.passed)
    rate = passed / len(results)
    assert rate >= 0.8, f"Happy-path pass rate {rate:.0%} is below 80%"