"""Compare committed validation metrics with the committed quality baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def quality_gate(
    metrics_path: Path,
    baseline_path: Path,
    minimum_delta: float = 0.01,
) -> int:
    """Return 1 when top-1 falls below baseline minus the allowed delta."""
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    actual = float(metrics["top1"])
    minimum = float(baseline["top1"]) - minimum_delta
    if actual < minimum:
        print(f"quality gate failed: top1={actual} < minimum={minimum}")
        return 1
    print(f"quality gate passed: top1={actual} >= minimum={minimum}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the validation quality gate.")
    parser.add_argument("metrics", type=Path)
    parser.add_argument("baseline", type=Path)
    args = parser.parse_args()
    return quality_gate(args.metrics, args.baseline)


if __name__ == "__main__":
    raise SystemExit(main())
