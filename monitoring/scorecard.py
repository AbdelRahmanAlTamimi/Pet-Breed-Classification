"""Score every drift detector against the labelled corruption suite.

``uv run --group monitor python -m monitoring.scorecard``

For each corruption x severity, and for clean test images as the control, the command
draws seeded disjoint batches and runs every detector with the frozen thresholds from
reports/drift_thresholds.json. On corrupted batches the alarm rate is the detection rate;
on clean batches it is the false-alarm rate. Test images are used only as unlabeled
detector inputs; nothing here changes a threshold. The post-hoc ``embedding_valref``
detector is appended as extra columns after the original ones.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from collections.abc import Sequence
from typing import Any

import numpy as np

from pet_breed_classification.config import cfg
from pet_breed_classification.evaluate_test import corruption_levels, load_test_records
from pet_breed_classification.logging_conf import setup_logging

from .detectors import DETECTORS, EMBEDDING_VALREF, check_batch, embedding_auc
from .features import FeatureExtractor, load_features, save_features, subset
from .reference import CACHE_DIR, load_reference
from .thresholds import BATCH_SIZE, THRESHOLDS_PATH

logger = logging.getLogger(__name__)

BATCHES_PER_LEVEL = 10
SCORECARD_PATH = cfg.REPORTS_DIR / "drift_scorecard.csv"


def level_features(
    extractor: FeatureExtractor, corruption: str | None, severity: int, count: int
) -> dict[str, np.ndarray]:
    """Return cached features for a seeded sample of one test level."""
    name = "clean" if corruption is None else f"{corruption}_s{severity}"
    cache = CACHE_DIR / "scorecard" / f"{name}_{count}.npz"
    if cache.is_file():
        return load_features(cache)
    records = load_test_records(corruption, severity)
    rows = np.random.default_rng(cfg.SEED).choice(len(records), size=count, replace=False)
    paths = [cfg.PROJECT_ROOT / str(records[index]["path"]) for index in rows]
    features = extractor.extract(paths)
    save_features(cache, features)
    return features


def score_level(
    features: dict[str, np.ndarray], reference: Any, thresholds: dict[str, float], abstain: float
) -> dict[str, Any]:
    results, valref = [], []
    for index in range(len(features["embedding"]) // BATCH_SIZE):
        rows = np.arange(index * BATCH_SIZE, (index + 1) * BATCH_SIZE)
        valref.append(embedding_auc(reference.embedding_valref, features["embedding"][rows], cfg.SEED + index))
        results.append(
            check_batch(
                subset(features, rows),
                reference.train,
                reference.confidence,
                thresholds,
                abstain,
                cfg.SEED + index,
            )
        )
    row: dict[str, Any] = {"batches": len(results)}
    for name in DETECTORS:
        row[f"{name}_alarm_rate"] = float(np.mean([result["alarms"][name] for result in results]))
        row[f"{name}_mean_score"] = float(np.mean([result["scores"][name] for result in results]))
    row["mean_abstain_rate"] = float(np.mean([result["abstain_rate"] for result in results]))
    row[f"{EMBEDDING_VALREF}_alarm_rate"] = float(np.mean(np.asarray(valref) > thresholds[EMBEDDING_VALREF]))
    row[f"{EMBEDDING_VALREF}_mean_score"] = float(np.mean(valref))
    return row


def markdown(rows: Sequence[dict[str, Any]]) -> str:
    lines = [
        (
            "| Corruption | Severity | Batches | Embedding | Pixel stats | Confidence | Abstain rate "
            "| Embedding, val ref (post-hoc) |"
        ),
        "|---|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        rates = " | ".join(f"{row[f'{name}_alarm_rate']:.0%}" for name in DETECTORS)
        lines.append(
            f"| {row['corruption']} | {row['severity']} | {row['batches']} | {rates} | "
            f"{row['mean_abstain_rate']:.3f} | {row[f'{EMBEDDING_VALREF}_alarm_rate']:.0%} |"
        )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    setup_logging()
    parser = argparse.ArgumentParser(description="Score drift detectors against corruptions.")
    parser.add_argument("--batches", type=int, default=BATCHES_PER_LEVEL)
    args = parser.parse_args(argv)
    if args.batches < 10:
        parser.error("use at least 10 batches per level")

    thresholds = json.loads(THRESHOLDS_PATH.read_text(encoding="utf-8"))["thresholds"]
    extractor = FeatureExtractor()
    reference = load_reference(extractor)
    abstain = float(extractor.predictor.abstain_threshold)
    count = args.batches * BATCH_SIZE

    rows = []
    for corruption, severity in [(None, 0), *corruption_levels()]:
        features = level_features(extractor, corruption, severity, count)
        row = {"corruption": corruption or "clean (control)", "severity": severity}
        row.update(score_level(features, reference, thresholds, abstain))
        logger.info("scorecard %s", row)
        rows.append(row)

    with SCORECARD_PATH.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(markdown(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
