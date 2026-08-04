"""
label_ground_truth.py — Interactive CLI for manually labeling failed test results.

Walk through every failure in one or more run logs and assign a ground-truth label.
Labels are saved to a persistent CSV so you can build up 50-100 labeled cases
across multiple runs over time, then use evaluate_triage.py to score the LLM triager.

Usage:
    python label_ground_truth.py logs/run_4b21ab01_20260628T075458.json
    python label_ground_truth.py --all          # label every run_*.json in logs/
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from generator import TestResult

LABELS = ["product_bug", "bad_test", "environment_issue"]
GROUND_TRUTH_CSV = Path(__file__).parent / "logs" / "ground_truth_labels.csv"


def _load_existing_labels() -> dict[str, str]:
    if not GROUND_TRUTH_CSV.exists():
        return {}
    with open(GROUND_TRUTH_CSV) as f:
        reader = csv.DictReader(f)
        return {row["test_id"] + "|" + row["run_id"]: row["manual_label"] for row in reader}


def _append_label(run_id: str, test_id: str, label: str, method: str, path: str, failure_detail: str):
    GROUND_TRUTH_CSV.parent.mkdir(parents=True, exist_ok=True)
    is_new = not GROUND_TRUTH_CSV.exists()
    with open(GROUND_TRUTH_CSV, "a", newline="") as f:
        writer = csv.writer(f)
        if is_new:
            writer.writerow(["run_id", "test_id", "method", "path", "failure_detail", "manual_label"])
        writer.writerow([run_id, test_id, method, path, failure_detail, label])


def _prompt_label(result: TestResult) -> str | None:
    print("\n" + "─" * 60)
    print(f"  Test ID       : {result.test_id}")
    print(f"  Description   : {result.description}")
    print(f"  Request       : {result.method} {result.resolved_url}")
    if result.request_body:
        print(f"  Request body  : {json.dumps(result.request_body)}")
    print(f"  Expected      : status={result.expected_status}  keys={result.expected_schema_keys}")
    print(f"  Actual        : status={result.actual_status}")
    body_preview = json.dumps(result.actual_response_body)
    print(f"  Actual body   : {body_preview[:200]}")
    print(f"  Failure detail: {result.failure_detail}")
    if result.triage_label:
        print(f"  [LLM triage was: {result.triage_label} (confidence={result.triage_confidence})]")
        print(f"  [LLM rationale: {result.triage_rationale}]")
    print("─" * 60)
    print("  1) product_bug       2) bad_test       3) environment_issue")
    print("  s) skip              q) quit")

    while True:
        choice = input("  Your label> ").strip().lower()
        if choice == "q":
            return "QUIT"
        if choice == "s":
            return None
        mapping = {"1": "product_bug", "2": "bad_test", "3": "environment_issue"}
        if choice in mapping:
            return mapping[choice]
        if choice in LABELS:
            return choice
        print("  Invalid input. Enter 1, 2, 3, s, or q.")


def label_run_log(path: Path, existing: dict[str, str]) -> int:
    payload = json.loads(path.read_text())
    run_id = payload["run_id"]
    results = [TestResult(**r) for r in payload["results"] if not r["passed"]]

    if not results:
        print(f"  {path.name}: no failures to label.")
        return 0

    labeled_count = 0
    for r in results:
        key = f"{r.test_id}|{run_id}"
        if key in existing:
            continue  # already labeled
        label = _prompt_label(r)
        if label == "QUIT":
            print(f"\nStopped. Labeled {labeled_count} new case(s) this session.")
            return labeled_count
        if label is None:
            continue
        _append_label(run_id, r.test_id, label, r.method, r.path, r.failure_detail or "")
        labeled_count += 1

    return labeled_count


def main():
    parser = argparse.ArgumentParser(description="Manually label failures for ground-truth triage evaluation")
    parser.add_argument("run_log", nargs="?", help="Path to a specific run_*.json log")
    parser.add_argument("--all", action="store_true", help="Label every run_*.json in logs/")
    args = parser.parse_args()

    logs_dir = Path(__file__).parent / "logs"
    existing = _load_existing_labels()
    print(f"Loaded {len(existing)} existing ground-truth label(s) from {GROUND_TRUTH_CSV}")

    if args.all:
        paths = sorted(logs_dir.glob("run_*.json"))
    elif args.run_log:
        paths = [Path(args.run_log)]
    else:
        paths = sorted(logs_dir.glob("run_*.json"))[-1:]  # most recent

    total_labeled = 0
    for p in paths:
        # Skip already-triaged output files to avoid re-labeling triage annotations
        if "-triaged" in p.stem:
            continue
        total_labeled += label_run_log(p, existing)
        existing = _load_existing_labels()  # refresh after each file

    print(f"\nDone. {total_labeled} new label(s) added.")
    print(f"Ground truth CSV: {GROUND_TRUTH_CSV}")


if __name__ == "__main__":
    main()