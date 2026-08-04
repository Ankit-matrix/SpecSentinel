"""
triage_run.py — Week 2 CLI.

Loads a saved run's JSON log, re-triages every failed TestResult via the LLM,
and writes a new annotated JSON + Parquet file (does not overwrite the original).

Usage:
    python triage_run.py logs/run_4b21ab01_20260628T075458.json
    python triage_run.py --latest        # auto-picks most recent run_*.json in logs/
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("triage_run")

sys.path.insert(0, str(Path(__file__).parent))

from generator import FailureTriager, RunLogger, RunSummary, TestResult


def _latest_run_log(logs_dir: Path) -> Path:
    candidates = sorted(logs_dir.glob("run_*.json"))
    if not candidates:
        raise FileNotFoundError(f"No run_*.json files found in {logs_dir}")
    return candidates[-1]


def load_summary_from_json(path: Path) -> RunSummary:
    payload = json.loads(path.read_text())
    results = [TestResult(**r) for r in payload["results"]]
    return RunSummary(
        run_id=payload["run_id"],
        base_url=payload["base_url"],
        spec_title=payload["spec_title"],
        spec_version=payload["spec_version"],
        total=payload["total"],
        passed=payload["passed"],
        failed=payload["failed"],
        results=results,
    )


def main():
    parser = argparse.ArgumentParser(description="Week 2 — triage failures in a saved run log")
    parser.add_argument("run_log", nargs="?", help="Path to a run_*.json log file")
    parser.add_argument("--latest", action="store_true", help="Use the most recent log in logs/")
    args = parser.parse_args()

    logs_dir = Path(__file__).parent / "logs"

    if args.latest or not args.run_log:
        run_log_path = _latest_run_log(logs_dir)
    else:
        run_log_path = Path(args.run_log)

    log.info(f"Loading run log: {run_log_path}")
    summary = load_summary_from_json(run_log_path)

    failures = [r for r in summary.results if not r.passed]
    if not failures:
        log.info("No failures in this run — nothing to triage.")
        return

    log.info(f"Triaging {len(failures)} failure(s) via LLM …")
    triager = FailureTriager()
    annotated_results = triager.triage_all(summary.results)

    annotated_summary = summary.model_copy(update={"results": annotated_results})

    logger = RunLogger(output_dir=str(logs_dir))
    # Give it a distinct run_id suffix so it doesn't collide with the original file
    triaged_summary = annotated_summary.model_copy(update={"run_id": f"{summary.run_id}-triaged"})
    paths = logger.save(triaged_summary)

    print("\n" + "═" * 60)
    print("  TRIAGE SUMMARY")
    print("═" * 60)
    from collections import Counter
    label_counts = Counter(r.triage_label for r in annotated_results if r.triage_label)
    for label, count in label_counts.items():
        print(f"  {label:20} : {count}")
    print()
    for r in annotated_results:
        if r.triage_label:
            print(f"  [{r.triage_label:18} conf={r.triage_confidence:.2f}] {r.test_id}")
            print(f"      {r.triage_rationale}")
    print("═" * 60)
    print(f"\nAnnotated log written to:\n  JSON    → {paths['json']}\n  Parquet → {paths['parquet']}")


if __name__ == "__main__":
    main()