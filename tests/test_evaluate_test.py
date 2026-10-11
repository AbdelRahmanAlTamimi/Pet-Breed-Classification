from __future__ import annotations

import numpy as np
import pytest

from pet_breed_classification import evaluate_test
from pet_breed_classification.evaluate_test import (
    most_confused_pairs,
    seeded_sample,
    selective_metrics,
)


def test_most_confused_pairs_orders_off_diagonal_counts() -> None:
    matrix = np.array([[5, 3, 0], [1, 4, 0], [0, 2, 6]])

    pairs = most_confused_pairs(matrix, ["a", "b", "c"], limit=5)

    assert pairs == [
        {"true": "a", "predicted": "b", "count": 3},
        {"true": "c", "predicted": "b", "count": 2},
        {"true": "b", "predicted": "a", "count": 1},
    ]


def test_seeded_sample_is_reproducible_and_keeps_order() -> None:
    records = [{"image_id": str(index)} for index in range(50)]

    first = seeded_sample(records, 10, seed=42)

    assert first == seeded_sample(records, 10, seed=42)
    assert len(first) == 10
    assert [int(r["image_id"]) for r in first] == sorted(int(r["image_id"]) for r in first)
    assert seeded_sample(records, 100, seed=42) == records


def test_selective_metrics_uses_the_threshold_on_calibrated_scores() -> None:
    logits = np.array([[4.0, 0.0], [0.0, 4.0], [0.1, 0.0], [3.0, 0.0]])
    labels = np.array([0, 1, 1, 1])

    metrics = selective_metrics(logits, labels, temperature=1.0, threshold=0.9)

    assert metrics["n"] == 4
    assert metrics["top1"] == pytest.approx(0.5)
    # Rows 0, 1 and 3 clear 0.9; row 3 is wrong.
    assert metrics["coverage"] == pytest.approx(0.75)
    assert metrics["selective_accuracy"] == pytest.approx(2 / 3)


def test_main_refuses_to_overwrite_the_final_report(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    report = tmp_path / "final_test_evaluation.json"
    report.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(evaluate_test, "REPORT_PATH", report)
    monkeypatch.setattr(evaluate_test, "_relative", lambda path: path.name)

    with pytest.raises(SystemExit):
        evaluate_test.main([])
