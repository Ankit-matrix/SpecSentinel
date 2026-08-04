"""
triage.py — Week 2: Feeds each failed TestResult back to the LLM and asks it
to classify the failure as one of:
  - product_bug        : the API genuinely behaves incorrectly
  - bad_test           : the test case itself is wrong (bad expectation, invalid input)
  - environment_issue  : flaky/infra problem (timeout, connection reset, 5xx from a dependency)

Returns each result annotated with triage_label, triage_confidence, triage_rationale.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import groq
from pydantic import BaseModel, Field, ValidationError

from .models import TestResult

log = logging.getLogger(__name__)

VALID_LABELS = {"product_bug", "bad_test", "environment_issue"}

SYSTEM_PROMPT = """You are an expert SDET performing failure triage on API test results.

For each failed test, you will see:
- The request that was sent (method, path, body)
- The expected status code and expected schema keys
- The actual status code and actual response body
- Any failure detail already computed by the test harness

Classify the failure into EXACTLY one of these three categories:

1. "product_bug" — The API is genuinely behaving incorrectly. The request was well-formed
   and reasonable, but the API returned the wrong status code, is missing expected fields,
   returned wrong types, or violates its own documented contract (e.g. missing referential
   integrity checks, ignoring query parameters, wrong REST status code conventions).

2. "bad_test" — The test case itself has an incorrect expectation. This happens when the
   expected_status or expected_schema_keys the test asserts don't actually match reasonable
   API behavior — e.g. the test expected a 200 for a request that should legitimately 422,
   or the test made up a schema key that was never part of the API's contract.

3. "environment_issue" — The failure is due to infrastructure/flakiness rather than actual
   API logic: connection errors, timeouts, 5xx errors from something unrelated to the
   endpoint's business logic, non-deterministic timing issues.

OUTPUT FORMAT (strict JSON, no markdown, no prose):
{
  "test_id": "<same test_id as input>",
  "label": "product_bug" | "bad_test" | "environment_issue",
  "confidence": 0.0 to 1.0,
  "rationale": "one or two sentences explaining the classification"
}

Output ONLY the JSON object for the single failure provided. No explanation outside the JSON."""

USER_PROMPT_TEMPLATE = """Classify this test failure:

TEST ID: {test_id}
DESCRIPTION: {description}
REQUEST: {method} {resolved_url}
REQUEST BODY: {request_body}

EXPECTED STATUS: {expected_status}
EXPECTED SCHEMA KEYS: {expected_schema_keys}

ACTUAL STATUS: {actual_status}
ACTUAL RESPONSE BODY: {actual_response_body}

FAILURE DETAIL (from test harness): {failure_detail}

Output ONLY the JSON classification object."""


class TriageResult(BaseModel):
    test_id: str
    label: str = Field(..., description="One of product_bug, bad_test, environment_issue")
    confidence: float = Field(..., ge=0.0, le=1.0)
    rationale: str


class FailureTriager:
    def __init__(self, model: str = "llama-3.3-70b-versatile", max_retries: int = 2):
        self.client = groq.Groq()
        self.model = model
        self.max_retries = max_retries

    def triage_all(self, results: list[TestResult]) -> list[TestResult]:
        """Triage every failed result in-place (returns new list, originals untouched)."""
        annotated = []
        for r in results:
            if r.passed:
                annotated.append(r)
                continue
            triage = self._triage_one(r)
            r_copy = r.model_copy(update={
                "triage_label": triage.label,
                "triage_confidence": triage.confidence,
                "triage_rationale": triage.rationale,
            })
            log.info(f"[{triage.label:20}] {r.test_id} (confidence={triage.confidence:.2f})")
            annotated.append(r_copy)
        return annotated

    def _triage_one(self, result: TestResult) -> TriageResult:
        user_msg = USER_PROMPT_TEMPLATE.format(
            test_id=result.test_id,
            description=result.description,
            method=result.method,
            resolved_url=result.resolved_url,
            request_body=json.dumps(result.request_body) if result.request_body else "null",
            expected_status=result.expected_status,
            expected_schema_keys=result.expected_schema_keys,
            actual_status=result.actual_status,
            actual_response_body=json.dumps(result.actual_response_body)[:1000],
            failure_detail=result.failure_detail or "N/A",
        )

        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            raw = self._call_llm(user_msg)
            try:
                data = self._parse_json(raw)
                triage = TriageResult.model_validate(data)
                if triage.label not in VALID_LABELS:
                    raise ValueError(f"Invalid label: {triage.label}")
                return triage
            except (json.JSONDecodeError, ValidationError, ValueError) as e:
                last_error = e
                log.warning(f"Triage parse error for {result.test_id} attempt {attempt + 1}: {e}")
                user_msg += f"\n\nPREVIOUS ATTEMPT FAILED: {e}\nFix and output only the corrected JSON."

        # Fallback: don't crash the whole run over one bad classification
        log.error(f"Triage failed for {result.test_id} after retries: {last_error}")
        return TriageResult(
            test_id=result.test_id,
            label="bad_test",
            confidence=0.0,
            rationale=f"Triage LLM failed to produce valid output: {last_error}",
        )

    def _call_llm(self, user_message: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            max_tokens=512,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
        )
        return response.choices[0].message.content.strip()

    def _parse_json(self, raw: str) -> Any:
        cleaned = raw
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            cleaned = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
        return json.loads(cleaned)