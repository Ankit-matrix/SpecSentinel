"""
test_executor.py — Executes a list of TestCase objects against a live API
and returns TestResult objects with pass/fail info.
"""

from __future__ import annotations

import logging
import time
from typing import Any
from urllib.parse import urljoin

import requests

from .models import FailureReason, TestCase, TestResult

log = logging.getLogger(__name__)

# Map OpenAPI type names → Python types for schema validation
_TYPE_MAP: dict[str, type | tuple] = {
    "string": str,
    "number": (int, float),
    "integer": int,
    "boolean": bool,
    "array": list,
    "object": dict,
    "null": type(None),
}


class TestExecutor:
    def __init__(self, base_url: str, timeout: int = 10):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json", "Accept": "application/json"})

    def run_all(self, test_cases: list[TestCase]) -> list[TestResult]:
        results = []
        for tc in test_cases:
            result = self._run_one(tc)
            status = "PASS" if result.passed else "FAIL"
            log.info(f"[{status}] {tc.test_id} — {tc.method} {tc.path} → {result.actual_status} (expected {tc.expected_status})")
            results.append(result)
        return results

    def _run_one(self, tc: TestCase) -> TestResult:
        # Resolve path parameters
        try:
            resolved_path = tc.path.format(**tc.path_params) if tc.path_params else tc.path
        except KeyError as e:
            resolved_path = tc.path
            log.warning(f"Missing path param {e} for {tc.test_id}")

        url = self.base_url + resolved_path

        base_result = dict(
            test_id=tc.test_id,
            description=tc.description,
            category=tc.category,
            method=tc.method,
            path=tc.path,
            resolved_url=url,
            request_body=tc.request_body,
            expected_status=tc.expected_status,
            expected_schema_keys=tc.expected_schema_keys,
        )

        # Execute request
        try:
            start = time.perf_counter()
            resp = self.session.request(
                method=tc.method,
                url=url,
                json=tc.request_body,
                params=tc.query_params or None,
                timeout=self.timeout,
            )
            elapsed_ms = (time.perf_counter() - start) * 1000
        except requests.RequestException as e:
            return TestResult(
                **base_result,
                passed=False,
                failure_reasons=[FailureReason.REQUEST_ERROR],
                failure_detail=str(e),
            )

        # Parse response body
        try:
            body = resp.json() if resp.content else None
        except ValueError:
            body = resp.text

        # ── Validation checks ─────────────────────────────────────────────────
        failure_reasons: list[FailureReason] = []
        failure_details: list[str] = []

        # 1. Status code check
        if resp.status_code != tc.expected_status:
            failure_reasons.append(FailureReason.STATUS_MISMATCH)
            failure_details.append(
                f"Expected status {tc.expected_status}, got {resp.status_code}"
            )

        # 2. Schema key presence (only for passing status codes)
        if tc.expected_schema_keys and isinstance(body, dict):
            for key in tc.expected_schema_keys:
                if key not in body:
                    failure_reasons.append(FailureReason.SCHEMA_KEY_MISSING)
                    failure_details.append(f"Missing key '{key}' in response body")

        # 3. Schema type checks
        if tc.expected_schema_types and isinstance(body, dict):
            for key, expected_type in tc.expected_schema_types.items():
                if key in body:
                    py_type = _TYPE_MAP.get(expected_type)
                    if py_type and not isinstance(body[key], py_type):
                        failure_reasons.append(FailureReason.SCHEMA_TYPE_MISMATCH)
                        failure_details.append(
                            f"Key '{key}': expected {expected_type}, got {type(body[key]).__name__}"
                        )

        passed = len(failure_reasons) == 0
        return TestResult(
            **base_result,
            actual_status=resp.status_code,
            actual_response_body=body,
            response_time_ms=round(elapsed_ms, 2),
            passed=passed,
            failure_reasons=failure_reasons,
            failure_detail="; ".join(failure_details) if failure_details else None,
        )