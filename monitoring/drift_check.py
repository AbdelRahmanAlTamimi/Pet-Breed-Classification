"""Drift check that runs after every batch-scoring run.

Splits the scored images into batches, runs every detector with the frozen thresholds,
writes reports/drift/latest.json, appends one line per batch to reports/drift/history.jsonl,
saves an Evidently data-drift report for the latest batch, pushes gauges to a Prometheus
Pushgateway when ``PETBREED_PUSHGATEWAY_URL`` is set, and hands any alarm to the guarded
retraining loop in ``monitoring.retrain_on_drift``.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from pet_breed_classification.config import cfg

from .detectors import DETECTORS, check_batch
from .features import PIXEL_STATS, FeatureExtractor, Features, subset
from .reference import load_reference
from .retrain_on_drift import on_drift_check
from .thresholds import BATCH_SIZE, THRESHOLDS_PATH

logger = logging.getLogger(__name__)

DRIFT_DIR = cfg.REPORTS_DIR / "drift"
LATEST_PATH = DRIFT_DIR / "latest.json"
HISTORY_PATH = DRIFT_DIR / "history.jsonl"
EVIDENTLY_PATH = cfg.REPORTS_DIR / "evidently_latest_batch.html"
MIN_BATCH_SIZE = 50
TABULAR_COLUMNS = (*PIXEL_STATS, "max_confidence")
PUSH_JOB = "petbreed_batch_drift"


def batch_slices(count: int, batch_size: int = BATCH_SIZE) -> list[slice]:
    """Split ``count`` rows into full batches; a trailing batch below 50 rows is dropped."""
    slices = [slice(start, min(start + batch_size, count)) for start in range(0, count, batch_size)]
    return [part for part in slices if part.stop - part.start >= MIN_BATCH_SIZE]


def save_evidently_report(reference: Features, current: Features, path: Path) -> None:
    """Save an Evidently DataDrift report on the per-image tabular features."""
    import pandas as pd
    from evidently import Report
    from evidently.presets import DataDriftPreset

    def frame(features: Features) -> pd.DataFrame:
        return pd.DataFrame({column: features[column] for column in TABULAR_COLUMNS})

    snapshot = Report([DataDriftPreset()]).run(
        current_data=frame(current), reference_data=frame(reference)
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    snapshot.save_html(str(path))


def push_gauges(url: str, result: dict[str, Any]) -> None:
    """Push the latest batch's drift gauges; the only label is ``detector``."""
    from prometheus_client import CollectorRegistry, Gauge, push_to_gateway

    registry = CollectorRegistry()
    score = Gauge("petbreed_drift_score", "Latest batch drift score.", ["detector"], registry=registry)
    alarm = Gauge("petbreed_drift_alarm", "1 when the score exceeds its threshold.", ["detector"], registry=registry)
    threshold = Gauge("petbreed_drift_threshold", "Alarm threshold per detector.", ["detector"], registry=registry)
    confidence = Gauge("petbreed_confidence_drift_ks", "KS statistic of calibrated max probability.", registry=registry)
    abstain = Gauge("petbreed_batch_abstain_rate", "Share of the latest batch below the abstain threshold.", registry=registry)
    for name in DETECTORS:
        score.labels(detector=name).set(result["scores"][name])
        alarm.labels(detector=name).set(int(result["alarms"][name]))
        threshold.labels(detector=name).set(result["thresholds"][name])
    confidence.set(result["scores"]["confidence"])
    abstain.set(result["abstain_rate"])
    push_to_gateway(url, job=PUSH_JOB, registry=registry)


def _relative(path: Path) -> str:
    resolved = path.resolve()
    if resolved.is_relative_to(cfg.PROJECT_ROOT):
        return resolved.relative_to(cfg.PROJECT_ROOT).as_posix()
    return resolved.as_posix()


def run_drift_check(
    paths: Sequence[Path],
    confidences: Sequence[float],
    *,
    source: str,
    extractor: FeatureExtractor | None = None,
) -> dict[str, Any] | None:
    """Check the scored images for drift; return the run summary, or None if too few."""
    slices = batch_slices(len(paths))
    if not slices:
        logger.warning("Drift check skipped: %d images is below one batch of %d", len(paths), MIN_BATCH_SIZE)
        return None
    thresholds = json.loads(THRESHOLDS_PATH.read_text(encoding="utf-8"))["thresholds"]
    extractor = extractor or FeatureExtractor()
    reference = load_reference(extractor)
    abstain = extractor.predictor.abstain_threshold
    if abstain is None:
        raise RuntimeError("The shipped model has no abstention threshold")
    features = extractor.extract(list(paths), with_confidence=False)
    features["max_confidence"] = np.asarray(confidences, dtype=np.float64)

    checked_at = datetime.now(UTC).isoformat()
    batches = []
    DRIFT_DIR.mkdir(parents=True, exist_ok=True)
    with HISTORY_PATH.open("a", encoding="utf-8") as history:
        for index, part in enumerate(slices):
            rows = np.arange(part.start, part.stop)
            result = check_batch(
                subset(features, rows),
                reference.train,
                reference.confidence,
                thresholds,
                float(abstain),
                cfg.SEED + index,
            )
            result.update({"batch": index, "checked_at": checked_at, "source": source})
            batches.append(result)
            history.write(json.dumps(result) + "\n")
            logger.info("Drift batch %d: scores=%s alarms=%s", index, result["scores"], result["alarms"])

    latest = batches[-1]
    save_evidently_report(reference.val, subset(features, np.arange(slices[-1].start, slices[-1].stop)), EVIDENTLY_PATH)
    summary = {
        "checked_at": checked_at,
        "source": source,
        "images": len(paths),
        "batch_size": BATCH_SIZE,
        "batches": batches,
        "latest": latest,
        "any_alarm": any(any(batch["alarms"].values()) for batch in batches),
        "evidently_report": _relative(EVIDENTLY_PATH),
    }
    url = os.environ.get("PETBREED_PUSHGATEWAY_URL")
    if url:
        push_gauges(url, latest)
        logger.info("Pushed drift gauges to %s", url)
    # An alarm may trigger a guarded fine-tune; its decision is part of the record.
    summary["retrain"] = on_drift_check(len(paths), summary["any_alarm"])
    LATEST_PATH.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary
