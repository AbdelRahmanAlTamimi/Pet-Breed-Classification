from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))
# The maintenance script is intentionally kept outside the installable package.
from scripts.quality_gate import quality_gate


def _write(path: Path, top1: float) -> None:
    path.write_text(json.dumps({"top1": top1}) + "\n", encoding="utf-8")


def test_quality_gate_passes_within_margin(tmp_path: Path) -> None:
    metrics = tmp_path / "metrics.json"
    baseline = tmp_path / "baseline.json"
    _write(metrics, 0.941)
    _write(baseline, 0.95)

    assert quality_gate(metrics, baseline) == 0


def test_quality_gate_rejects_large_regression(tmp_path: Path) -> None:
    metrics = tmp_path / "metrics.json"
    baseline = tmp_path / "baseline.json"
    _write(metrics, 0.939)
    _write(baseline, 0.95)

    assert quality_gate(metrics, baseline) == 1
