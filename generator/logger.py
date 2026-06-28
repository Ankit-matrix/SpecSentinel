"""
logger.py — Persists RunSummary to both JSON (human-readable) and Parquet (analytics).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from datetime import datetime

import pandas as pd

from .models import RunSummary, TestResult

log = logging.getLogger(__name__)


def _result_to_flat_dict(r: TestResult) -> dict:
    """Flatten a TestResult into a dict suitable for a DataFrame row."""
    return {
        "test_id": r.test_id,
        "description": r.description,
        "category": r.category.value,
        "method": r.method,
        "path": r.path,
        "resolved_url": r.resolved_url,
        "request_body": json.dumps(r.request_body) if r.request_body else None,
        "actual_status": r.actual_status,
        "expected_status": r.expected_status,
        "actual_response_body": json.dumps(r.actual_response_body) if r.actual_response_body is not None else None,
        "response_time_ms": r.response_time_ms,
        "passed": r.passed,
        "failure_reasons": ",".join(fr.value for fr in r.failure_reasons),
        "failure_detail": r.failure_detail,
        "triage_label": r.triage_label,
        "triage_confidence": r.triage_confidence,
        "triage_rationale": r.triage_rationale,
        "manual_label": r.manual_label,
    }


class RunLogger:
    def __init__(self, output_dir: str = "logs"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def save(self, summary: RunSummary) -> dict[str, Path]:
        """Save run summary to JSON + Parquet. Returns dict of paths."""
        timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
        stem = f"run_{summary.run_id}_{timestamp}"

        json_path = self._save_json(summary, stem)
        parquet_path = self._save_parquet(summary, stem)

        log.info(f"Saved run log → JSON: {json_path}  Parquet: {parquet_path}")
        return {"json": json_path, "parquet": parquet_path}

    def _save_json(self, summary: RunSummary, stem: str) -> Path:
        path = self.output_dir / f"{stem}.json"
        payload = {
            "run_id": summary.run_id,
            "base_url": summary.base_url,
            "spec_title": summary.spec_title,
            "spec_version": summary.spec_version,
            "total": summary.total,
            "passed": summary.passed,
            "failed": summary.failed,
            "pass_rate": round(summary.passed / summary.total * 100, 1) if summary.total else 0,
            "results": [r.model_dump() for r in summary.results],
        }
        path.write_text(json.dumps(payload, indent=2, default=str))
        return path

    def _save_parquet(self, summary: RunSummary, stem: str) -> Path:
        path = self.output_dir / f"{stem}.parquet"
        rows = [_result_to_flat_dict(r) for r in summary.results]
        df = pd.DataFrame(rows)
        df["run_id"] = summary.run_id
        df["spec_title"] = summary.spec_title
        df.to_parquet(path, index=False, engine="pyarrow")
        return path

    def load_parquet(self, path: str | Path) -> pd.DataFrame:
        return pd.read_parquet(path, engine="pyarrow")

    def print_summary(self, summary: RunSummary) -> None:
        print("\n" + "═" * 60)
        print(f"  RUN SUMMARY  [{summary.spec_title} v{summary.spec_version}]")
        print("═" * 60)
        print(f"  Total tests : {summary.total}")
        print(f"  Passed      : {summary.passed}  ✓")
        print(f"  Failed      : {summary.failed}  ✗")
        if summary.total:
            print(f"  Pass rate   : {summary.passed / summary.total * 100:.1f}%")
        print()

        failures = [r for r in summary.results if not r.passed]
        if failures:
            print("  FAILURES:")
            for r in failures:
                print(f"    ✗ [{r.method}] {r.path}  →  {r.test_id}")
                print(f"      Expected {r.expected_status}, got {r.actual_status}")
                if r.failure_detail:
                    print(f"      {r.failure_detail}")
        print("═" * 60 + "\n")