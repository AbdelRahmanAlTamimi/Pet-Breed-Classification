from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

from monitoring import drift_check
from monitoring.detectors import (
    PIXEL_ALPHA,
    check_batch,
    embedding_auc,
    pixel_p_values,
    pixel_score,
)
from monitoring.features import pixel_stats
from monitoring.reference import val_halves


def _features(rng: np.random.Generator, count: int, shift: float = 0.0) -> dict[str, np.ndarray]:
    return {
        "embedding": rng.normal(shift, 1.0, size=(count, 64)).astype(np.float32),
        "brightness": rng.normal(120 + 40 * shift, 10, size=count),
        "contrast": rng.normal(50, 5, size=count),
        "sharpness": rng.normal(300, 30, size=count),
        "max_confidence": np.clip(rng.normal(0.95 - 0.3 * shift, 0.03, size=count), 0, 1),
    }


THRESHOLDS = {"embedding": 0.6, "pixel": -math.log10(PIXEL_ALPHA), "confidence": 0.2}


def test_pixel_stats_of_flat_and_striped_images() -> None:
    flat = Image.new("RGB", (16, 16), (100, 100, 100))
    brightness, contrast, sharpness = pixel_stats(flat)
    assert brightness == pytest.approx(100.0)
    assert contrast == pytest.approx(0.0)
    assert sharpness == pytest.approx(0.0)

    stripes = np.zeros((16, 16, 3), dtype=np.uint8)
    stripes[:, ::2] = 255
    assert pixel_stats(Image.fromarray(stripes))[2] > 0


def test_embedding_auc_separates_shifted_batches_only() -> None:
    rng = np.random.default_rng(0)
    reference = rng.normal(size=(400, 64))

    same = embedding_auc(reference, rng.normal(size=(200, 64)), seed=1)
    shifted = embedding_auc(reference, rng.normal(1.0, 1.0, size=(200, 64)), seed=1)

    assert same < 0.65
    assert shifted > 0.95


def test_pixel_score_is_minus_log10_of_the_smallest_p_value() -> None:
    assert pixel_score({"a": 0.5, "b": 0.001, "c": 0.2}) == pytest.approx(3.0)
    assert pixel_score({"a": 0.0}) == pytest.approx(300.0)


def test_check_batch_alarms_on_shift_and_stays_quiet_on_clean_data() -> None:
    rng = np.random.default_rng(42)
    reference = _features(rng, 400)
    confidence_reference = _features(rng, 200)["max_confidence"]

    clean = check_batch(_features(rng, 200), reference, confidence_reference, THRESHOLDS, 0.9, 0)
    drifted = check_batch(
        _features(rng, 200, shift=1.0), reference, confidence_reference, THRESHOLDS, 0.9, 0
    )

    assert clean["alarms"] == {"embedding": False, "pixel": False, "confidence": False}
    assert drifted["alarms"] == {"embedding": True, "pixel": True, "confidence": True}
    assert drifted["abstain_rate"] > clean["abstain_rate"]
    assert set(pixel_p_values(reference, _features(rng, 50))) == {
        "brightness",
        "contrast",
        "sharpness",
    }


def test_val_halves_are_disjoint_and_cover_everything() -> None:
    first, second = val_halves(11, seed=42)
    assert len(first) == 5
    assert sorted(np.concatenate([first, second]).tolist()) == list(range(11))


def test_batch_slices_drop_only_a_small_trailing_batch() -> None:
    assert [(s.start, s.stop) for s in drift_check.batch_slices(449)] == [(0, 200), (200, 400)]
    assert [(s.start, s.stop) for s in drift_check.batch_slices(460)] == [
        (0, 200),
        (200, 400),
        (400, 460),
    ]
    assert drift_check.batch_slices(49) == []


def test_push_gauges_uses_only_the_detector_label(monkeypatch: pytest.MonkeyPatch) -> None:
    pushed: dict[str, Any] = {}

    def fake_push(url: str, job: str, registry: Any) -> None:
        pushed.update(url=url, job=job, samples=[
            sample for metric in registry.collect() for sample in metric.samples
        ])

    monkeypatch.setattr("prometheus_client.push_to_gateway", fake_push)
    result = {
        "scores": {"embedding": 0.9, "pixel": 5.0, "confidence": 0.3},
        "alarms": {"embedding": True, "pixel": True, "confidence": False},
        "thresholds": THRESHOLDS,
        "abstain_rate": 0.4,
    }

    drift_check.push_gauges("http://pushgateway:9091", result)

    assert pushed["job"] == drift_check.PUSH_JOB
    by_name = {(s.name, tuple(sorted(s.labels.items()))): s.value for s in pushed["samples"]}
    assert by_name[("petbreed_drift_alarm", (("detector", "embedding"),))] == 1
    assert by_name[("petbreed_drift_alarm", (("detector", "confidence"),))] == 0
    assert by_name[("petbreed_confidence_drift_ks", ())] == pytest.approx(0.3)
    assert all(set(s.labels) <= {"detector"} for s in pushed["samples"])


def test_evidently_report_is_written(tmp_path: Path) -> None:
    pytest.importorskip("evidently")
    rng = np.random.default_rng(3)
    path = tmp_path / "evidently.html"

    drift_check.save_evidently_report(_features(rng, 100), _features(rng, 60, shift=1.0), path)

    assert path.stat().st_size > 0
