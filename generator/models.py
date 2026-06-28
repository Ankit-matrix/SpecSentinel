"""
Core Pydantic models used throughout the pipeline.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional
from pydantic import BaseModel, Field


# ── Test-case generation ──────────────────────────────────────────────────────

class TestCategory(str, Enum):
    HAPPY_PATH = "happy_path"
    BOUNDARY = "boundary"
    ERROR_PATH = "error_path"
    AUTHENTICATION = "authentication"


class TestCase(BaseModel):
    test_id: str = Field(..., description="Unique slug, e.g. 'create_post_happy_path'")
    description: str = Field(..., description="One-line human description of what this test checks")
    category: TestCategory
    method: str = Field(..., description="HTTP method in UPPER CASE")
    path: str = Field(..., description="URL path, e.g. '/posts/{post_id}'")
    path_params: dict[str, Any] = Field(default_factory=dict, description="Values to substitute into path template")
    query_params: dict[str, Any] = Field(default_factory=dict)
    request_body: Optional[dict[str, Any]] = Field(default=None)
    expected_status: int = Field(..., description="Expected HTTP status code")
    expected_schema_keys: list[str] = Field(
        default_factory=list,
        description="Top-level keys that must be present in the JSON response body",
    )
    expected_schema_types: dict[str, str] = Field(
        default_factory=dict,
        description="Mapping of key → expected JSON type (string, number, boolean, array, object, null)",
    )
    notes: Optional[str] = None


class TestCaseBatch(BaseModel):
    """LLM returns this as the top-level JSON object."""
    test_cases: list[TestCase]


# ── Test execution ────────────────────────────────────────────────────────────

class FailureReason(str, Enum):
    STATUS_MISMATCH = "status_mismatch"
    SCHEMA_KEY_MISSING = "schema_key_missing"
    SCHEMA_TYPE_MISMATCH = "schema_type_mismatch"
    REQUEST_ERROR = "request_error"
    JSON_DECODE_ERROR = "json_decode_error"


class TestResult(BaseModel):
    test_id: str
    description: str
    category: TestCategory
    method: str
    path: str
    resolved_url: str
    request_body: Optional[dict[str, Any]]

    # Actual outcomes
    actual_status: Optional[int] = None
    actual_response_body: Optional[Any] = None
    response_time_ms: Optional[float] = None

    # Expected values (copied from TestCase for traceability)
    expected_status: int
    expected_schema_keys: list[str]

    # Pass / fail
    passed: bool
    failure_reasons: list[FailureReason] = Field(default_factory=list)
    failure_detail: Optional[str] = None

    # Week-2 triage fields (filled in later)
    triage_label: Optional[str] = None        # "product_bug" | "bad_test" | "environment_issue"
    triage_confidence: Optional[float] = None
    triage_rationale: Optional[str] = None
    manual_label: Optional[str] = None        # ground truth for evaluation


class RunSummary(BaseModel):
    run_id: str
    base_url: str
    spec_title: str
    spec_version: str
    total: int
    passed: int
    failed: int
    results: list[TestResult]