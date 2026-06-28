"""
Usage:
    python main.py                     # starts its own FastAPI server
    python main.py --url http://...    # target an already-running API
    python main.py --spec openapi.json # load spec from file instead of /openapi.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import socket
import subprocess
import sys
import time
import uuid
from pathlib import Path

import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("pipeline")

sys.path.insert(0, str(Path(__file__).parent))

from generator import (
    logger as Logger,
    models,
    test_generator,
    test_executor,
    spec_parser
)


def find_free_port() -> int:
    with socket.socket() as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def start_target_api() -> tuple[subprocess.Popen, str]:
    port = find_free_port()
    target_dir = Path(__file__).parent / "target_api"
    log.info(f"Starting target FastAPI app on port {port} …")
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "main:app",
         "--host", "127.0.0.1", "--port", str(port), "--log-level", "warning"],
        cwd=str(target_dir),
    )
    base_url = f"http://127.0.0.1:{port}"
    for _ in range(40):
        try:
            requests.get(f"{base_url}/health", timeout=0.5)
            log.info(f"Target API ready at {base_url}")
            return proc, base_url
        except Exception:
            time.sleep(0.25)
    proc.terminate()
    raise RuntimeError("FastAPI server did not start in time.")


def fetch_spec(base_url: str) -> dict:
    url = f"{base_url}/openapi.json"
    log.info(f"Fetching OpenAPI spec from {url}")
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    return resp.json()


def run_pipeline(base_url: str, spec_path: str | None = None) -> RunSummary:
    # ── Step 1: Load + parse spec ─────────────────────────────────────────────
    print("\n" + "─" * 60)
    print("  STEP 1 / 3 — Parsing OpenAPI spec")
    print("─" * 60)

    raw_spec = spec_parser.load_openapi_from_file(spec_path) if spec_path else fetch_spec(base_url)
    parsed_spec = spec_parser.parse_openapi_spec(raw_spec)

    info = parsed_spec["info"]
    endpoints = parsed_spec["endpoints"]
    print(f"  API      : {info['title']} v{info['version']}")
    print(f"  Endpoints: {len(endpoints)}")
    for ep in endpoints:
        print(f"    {ep['method']:6} {ep['path']}")

    # ── Step 2: LLM test-case generation ─────────────────────────────────────
    print("\n" + "─" * 60)
    print("  STEP 2 / 3 — Generating test cases via LLM")
    print("─" * 60)

    generator = test_generator.TestCaseGenerator()
    test_cases = generator.generate(parsed_spec)

    # Group by category for display
    from collections import Counter
    cats = Counter(tc.category.value for tc in test_cases)
    print(f"  Generated {len(test_cases)} test cases:")
    for cat, n in sorted(cats.items()):
        print(f"    {cat:20} : {n}")

    print("\n  Sample test cases:")
    for tc in test_cases[:5]:
        print(f"    [{tc.category.value:12}] {tc.test_id}")
        print(f"      {tc.method} {tc.path}  →  expect {tc.expected_status}")

    # ── Step 3: Execute ───────────────────────────────────────────────────────
    print("\n" + "─" * 60)
    print("  STEP 3 / 3 — Executing tests against live API")
    print("─" * 60)

    executor = test_executor.TestExecutor(base_url=base_url)
    results = executor.run_all(test_cases)

    # ── Build summary + persist ───────────────────────────────────────────────
    run_id = str(uuid.uuid4())[:8]
    summary = models.RunSummary(
        run_id=run_id,
        base_url=base_url,
        spec_title=info["title"],
        spec_version=info["version"],
        total=len(results),
        passed=sum(1 for r in results if r.passed),
        failed=sum(1 for r in results if not r.passed),
        results=results,
    )

    logger = Logger.RunLogger(output_dir=str(Path(__file__).parent / "logs"))
    paths = logger.save(summary)
    logger.print_summary(summary)

    print(f"  Artifacts written:")
    print(f"    JSON    → {paths['json']}")
    print(f"    Parquet → {paths['parquet']}")

    # ── Failure breakdown by category ─────────────────────────────────────────
    failures = [r for r in results if not r.passed]
    if failures:
        print(f"\n  Failure breakdown:")
        from collections import Counter as C2
        by_cat = C2(r.category.value for r in failures)
        for cat, n in by_cat.items():
            print(f"    {cat:20} : {n} failures")

        print("\n  Failure details (ready for Week-2 triage):")
        for r in failures:
            print(f"\n    ✗ {r.test_id}")
            print(f"      {r.method} {r.resolved_url}")
            print(f"      Expected {r.expected_status}, got {r.actual_status}")
            if r.failure_detail:
                print(f"      {r.failure_detail}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="LLM-powered API test generator — Week 1 pipeline")
    parser.add_argument("--url", help="Base URL of already-running target API")
    parser.add_argument("--spec", help="Path to OpenAPI spec JSON/YAML file")
    args = parser.parse_args()

    proc = None
    try:
        if args.url:
            base_url = args.url.rstrip("/")
            summary = run_pipeline(base_url, spec_path=args.spec)
        else:
            proc, base_url = start_target_api()
            summary = run_pipeline(base_url)
    finally:
        if proc:
            log.info("Shutting down target API …")
            proc.terminate()
            proc.wait(timeout=5)


if __name__ == "__main__":
    main()