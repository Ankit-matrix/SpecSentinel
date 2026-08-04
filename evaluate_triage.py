"""
evaluate_triage.py — Compares LLM triage predictions against the manually labeled
ground truth CSV and reports precision / recall / F1 per category, plus a confusion matrix.

Usage:
    python evaluate_triage.py
    python evaluate_triage.py --triaged-log logs/run_xxx-triaged_*.json
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from generator import TestResult

LABELS = ["product_bug", "bad_test", "environment_issue"]
GROUND_TRUTH_CSV = Path(__file__).parent / "logs" / "ground_truth_labels.csv"


def load_ground_truth() -> dict[tuple[str, str], str]:
    """Returns {(run_id, test_id): manual_label}"""
    if not GROUND_TRUTH_CSV.exists():
        raise FileNotFoundError(
            f"No ground truth file found at {GROUND_TRUTH_CSV}. "
            f"Run label_ground_truth.py first."
        )
    gt = {}
    with open(GROUND_TRUTH_CSV) as f:
        for row in csv.DictReader(f):
            gt[(row["run_id"], row["test_id"])] = row["manual_label"]
    return gt


def load_triaged_predictions(logs_dir: Path) -> dict[tuple[str, str], dict]:
    """
    Loads every *-triaged*.json run log and returns
    {(base_run_id, test_id): {"label":..., "confidence":..., "rationale":...}}
    base_run_id strips the "-triaged" suffix so it matches the original run_id
    used in the ground-truth CSV.
    """
    predictions = {}
    for path in sorted(logs_dir.glob("run_*-triaged_*.json")):
        payload = json.loads(path.read_text())
        run_id = payload["run_id"].replace("-triaged", "")
        for r in payload["results"]:
            if r.get("triage_label"):
                predictions[(run_id, r["test_id"])] = {
                    "label": r["triage_label"],
                    "confidence": r["triage_confidence"],
                    "rationale": r["triage_rationale"],
                }
    return predictions


def compute_metrics(y_true: list[str], y_pred: list[str]) -> dict:
    """Per-label precision / recall / F1 + confusion matrix."""
    confusion = defaultdict(lambda: defaultdict(int))  # confusion[true][pred] += 1
    for t, p in zip(y_true, y_pred):
        confusion[t][p] += 1

    metrics = {}
    for label in LABELS:
        tp = confusion[label][label]
        fp = sum(confusion[other][label] for other in LABELS if other != label)
        fn = sum(confusion[label][other] for other in LABELS if other != label)

        precision = tp / (tp + fp) if (tp + fp) > 0 else None
        recall = tp / (tp + fn) if (tp + fn) > 0 else None
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision is not None and recall is not None and (precision + recall) > 0
            else None
        )
        support = sum(confusion[label].values())
        metrics[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }

    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    accuracy = correct / len(y_true) if y_true else 0

    return {"per_label": metrics, "accuracy": accuracy, "confusion": confusion, "n": len(y_true)}


def print_report(results: dict):
    print("\n" + "═" * 66)
    print("  TRIAGE EVALUATION REPORT")
    print("═" * 66)
    print(f"  Total labeled cases evaluated: {results['n']}")
    print(f"  Overall accuracy             : {results['accuracy']:.1%}\n")

    print(f"  {'Label':22} {'Precision':>10} {'Recall':>10} {'F1':>8} {'Support':>8}")
    print("  " + "─" * 62)
    for label, m in results["per_label"].items():
        p = f"{m['precision']:.2f}" if m["precision"] is not None else "  n/a"
        r = f"{m['recall']:.2f}" if m["recall"] is not None else "  n/a"
        f1 = f"{m['f1']:.2f}" if m["f1"] is not None else "n/a"
        print(f"  {label:22} {p:>10} {r:>10} {f1:>8} {m['support']:>8}")

    print("\n  Confusion matrix (rows = true label, cols = predicted label):")
    header = " " * 24 + "".join(f"{l[:12]:>14}" for l in LABELS)
    print(" " + header)
    for true_label in LABELS:
        row = f"  {true_label:22}"
        for pred_label in LABELS:
            row += f"{results['confusion'][true_label][pred_label]:>14}"
        print(row)

    print("\n" + "═" * 66)

    # Flag likely misclassifications for review
    print("\n  Mismatches (for manual review):")
    for true_label in LABELS:
        for pred_label in LABELS:
            if true_label != pred_label and results["confusion"][true_label][pred_label] > 0:
                n = results["confusion"][true_label][pred_label]
                print(f"    {n}x  true={true_label}  →  predicted={pred_label}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate LLM triage against manual ground truth")
    args = parser.parse_args()

    logs_dir = Path(__file__).parent / "logs"

    gt = load_ground_truth()
    print(f"Loaded {len(gt)} ground-truth label(s) from {GROUND_TRUTH_CSV}")

    predictions = load_triaged_predictions(logs_dir)
    print(f"Loaded {len(predictions)} LLM prediction(s) from triaged run logs")

    y_true, y_pred, missing = [], [], []
    for key, manual_label in gt.items():
        if key in predictions:
            y_true.append(manual_label)
            y_pred.append(predictions[key]["label"])
        else:
            missing.append(key)

    if missing:
        print(f"\n  WARNING: {len(missing)} ground-truth case(s) have no matching LLM prediction.")
        print("  Run triage_run.py on the corresponding run log(s) first.")
        for run_id, test_id in missing[:10]:
            print(f"    missing: run_id={run_id} test_id={test_id}")

    if not y_true:
        print("\nNo overlapping (ground_truth ∩ predictions) cases to evaluate. Exiting.")
        return

    results = compute_metrics(y_true, y_pred)
    print_report(results)

    if results["n"] < 50:
        print(f"\n  NOTE: Only {results['n']} labeled cases. Aim for 50-100 for a reliable eval,")
        print("  per the Week 2 project spec. Run label_ground_truth.py on more runs to grow the set.")


if __name__ == "__main__":
    main()