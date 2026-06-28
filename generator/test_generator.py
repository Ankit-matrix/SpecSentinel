"""
test_generator.py — Sends parsed OpenAPI spec to Llama and gets back
a validated list of TestCase objects.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

import groq
from pydantic import ValidationError

from .models import TestCase, TestCaseBatch

log = logging.getLogger(__name__)

# ── Prompt ────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert API test engineer. You will receive a summary of a REST API's OpenAPI spec and must produce a comprehensive set of test cases.

OUTPUT FORMAT (strict JSON, no markdown, no prose):
{
  "test_cases": [
    {
      "test_id": "unique_snake_case_slug",
      "description": "one-line description of what is being tested",
      "category": "happy_path" | "boundary" | "error_path" | "authentication",
      "method": "GET" | "POST" | "PUT" | "DELETE" | "PATCH",
      "path": "/path/with/{param}",
      "path_params": {"param": "concrete_value"},
      "query_params": {},
      "request_body": null | { ... },
      "expected_status": 200,
      "expected_schema_keys": ["id", "title"],
      "expected_schema_types": {"id": "string", "title": "string"},
      "notes": "optional extra context"
    }
  ]
}

RULES:
1. Generate at minimum 3 test cases per endpoint: one happy path, one boundary/edge, one error path.
2. For POST/PUT endpoints also generate: missing required fields (expect 422), empty strings where min_length > 0 (expect 422).
3. For GET /{id} endpoints: valid ID happy path, non-existent UUID (expect 404).
4. For list endpoints: default call (happy path), limit=0 or limit=-1 if limit param exists (expect 422 or 400).
5. path_params must use concrete string values, not placeholders. Use realistic UUIDs like "00000000-0000-0000-0000-000000000001" for non-existent resources.
6. expected_schema_keys should list the keys you expect in a successful response body (skip for error responses).
7. expected_schema_types maps each key to its JSON type: "string", "number", "boolean", "array", "object", "null".
8. Do NOT include Authorization headers or API keys — this is a public API.
9. Output ONLY the JSON object. No explanation, no markdown fences."""

USER_PROMPT_TEMPLATE = """Here is the API spec summary. Generate test cases for ALL endpoints.

SPEC:
{spec_json}

Remember: output ONLY the raw JSON object with a "test_cases" array."""


# ── Generator ─────────────────────────────────────────────────────────────────

class TestCaseGenerator:
    def __init__(self, model: str = "llama-3.3-70b-versatile", max_retries: int = 2):
        self.client = groq.Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.model = model
        self.max_retries = max_retries

    def generate(self, parsed_spec: dict) -> list[TestCase]:
        """
        Feed the parsed spec to Llama and return validated TestCase objects.
        Retries up to max_retries times on JSON/validation errors.
        """
        spec_json = json.dumps(parsed_spec, indent=2)
        user_msg = USER_PROMPT_TEMPLATE.format(spec_json=spec_json)

        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            log.info(f"LLM generation attempt {attempt + 1}/{self.max_retries + 1}")
            raw = self._call_llm(user_msg)
            log.debug(f"Raw LLM response (first 500 chars):\n{raw[:500]}")

            try:
                data = self._parse_json(raw)
                batch = TestCaseBatch.model_validate(data)
                log.info(f"Generated {len(batch.test_cases)} test cases successfully.")
                return batch.test_cases
            except (json.JSONDecodeError, ValidationError) as e:
                last_error = e
                log.warning(f"Parse/validation error on attempt {attempt + 1}: {e}")
                # On retry, append the error to the user message so the LLM can self-correct
                user_msg = (
                    USER_PROMPT_TEMPLATE.format(spec_json=spec_json)
                    + f"\n\nPREVIOUS ATTEMPT FAILED. Error: {e}\n"
                    + "Fix the JSON and try again. Output ONLY the corrected JSON object."
                )

        raise ValueError(f"Failed to generate valid test cases after {self.max_retries + 1} attempts. Last error: {last_error}")

    def _call_llm(self, user_message: str) -> str:
        message = self.client.chat.completions.create(
            model=self.model,
            max_tokens=4096,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message}
            ]
        )
        return message.choices[0].message.content.strip()

    def _parse_json(self, raw: str) -> Any:
        """Strip accidental markdown fences then parse JSON."""
        cleaned = raw
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            # Drop first and last fence lines
            cleaned = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
        return json.loads(cleaned)