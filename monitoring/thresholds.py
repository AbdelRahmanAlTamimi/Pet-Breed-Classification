"""Set drift thresholds from clean data only and write reports/drift_thresholds.json.

``uv run --group monitor python -m monitoring.thresholds``

Clean val batches are drawn repeatedly and scored against the clean references. Each
detector's threshold is the 99th percentile of its statistic over those batches. The
pixel detector keeps its fixed Bonferroni rule (min p < 0.01 / 3); its clean percentile is
recorded for comparison only. The test split is never read here.

The post-hoc ``embedding_valref`` threshold uses the val half that is the confidence
reference as its reference, and 50 batches of 200 drawn from the other val half with
its own generator, so the original three thresholds are unchanged.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from collections.abc import Sequence
from typing import Any

import numpy as np

from pet_breed_classification.config import cfg
from pet_breed_classification.logging_conf import setup_logging

from .detectors import (
    EMBEDDING_VALREF,
    PIXEL_ALPHA,
    confidence_ks,
    embedding_auc,
    pixel_p_values,
    pixel_score,
)
from .features import subset
from .reference import Reference, load_reference

logger = logging.getLogger(__name__)

BATCH_SIZE = 200
BOOTSTRAP_BATCHES = 50
PERCENTILE = 99.0
THRESHOLDS_PATH = cfg.REPORTS_DIR / "drift_thresholds.json"


def _summary(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "p50": float(np.percentile(array, 50)),
        "p99": float(np.percentile(array, PERCENTILE)),
        "max": float(array.max()),
    }


def embedding_valref_scores(reference: Reference, batches: int, batch_size: int) -> list[float]:
    """Score clean batches from the threshold val half against the confidence val half."""
    rng = np.random.default_rng(cfg.SEED)
    scores = []
    for index in range(batches):
        rows = rng.choice(reference.threshold_half, size=batch_size, replace=False)
        scores.append(embedding_auc(reference.embedding_valref, reference.val["embedding"][rows], cfg.SEED + index))
    return scores


def compute(batches: int = BOOTSTRAP_BATCHES, batch_size: int = BATCH_SIZE) -> dict[str, Any]:
    reference = load_reference()
    rng = np.random.default_rng(cfg.SEED)
    val_count = len(reference.val["max_confidence"])
    embedding, pixel, confidence = [], [], []
    for index in range(batches):
        rows = rng.choice(val_count, size=batch_size, replace=False)
        batch = subset(reference.val, rows)
        embedding.append(embedding_auc(reference.train["embedding"], batch["embedding"], cfg.SEED + index))
        pixel.append(pixel_score(pixel_p_values(reference.train, batch)))
        held_out = rng.choice(reference.threshold_half, size=batch_size, replace=False)
        confidence.append(confidence_ks(reference.confidence, reference.val["max_confidence"][held_out]))
        logger.info("bootstrap %d: auc=%.4f pixel=%.3f conf_ks=%.4f", index, embedding[-1], pixel[-1], confidence[-1])
    clean = {
        "embedding": _summary(embedding),
        "pixel": _summary(pixel),
        "confidence": _summary(confidence),
        EMBEDDING_VALREF: _summary(embedding_valref_scores(reference, batches, batch_size)),
    }
    return {
        "thresholds": {
            "embedding": clean["embedding"]["p99"],
            "pixel": -math.log10(PIXEL_ALPHA),
            "confidence": clean["confidence"]["p99"],
            EMBEDDING_VALREF: clean[EMBEDDING_VALREF]["p99"],
        },
        "rules": {
            "embedding": "alarm when 5-fold CV ROC-AUC > threshold (99th percentile on clean val)",
            "pixel": "alarm when -log10(min KS p) > threshold, i.e. min p < 0.01/3",
            "confidence": "alarm when KS statistic > threshold (99th percentile on clean val)",
            EMBEDDING_VALREF: (
                "post-hoc: alarm when 5-fold CV ROC-AUC vs the val reference half > threshold "
                "(99th percentile on the other val half)"
            ),
        },
        "clean_bootstrap": clean,
        "bootstrap_batches": batches,
        "batch_size": batch_size,
        "seed": cfg.SEED,
        "reference": {
            "embedding_and_pixel": f"clean train images (n={len(reference.train['embedding'])})",
            "confidence": f"seeded half of clean val (n={len(reference.confidence_half)})",
            "clean_batches": "embedding/pixel: clean val; confidence: the other val half",
            EMBEDDING_VALREF: (
                f"post-hoc: embeddings of the confidence val half (n={len(reference.confidence_half)}); "
                "clean batches from the other val half"
            ),
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    setup_logging()
    parser = argparse.ArgumentParser(description="Set drift thresholds from clean data.")
    parser.add_argument("--batches", type=int, default=BOOTSTRAP_BATCHES)
    args = parser.parse_args(argv)
    if args.batches < 20:
        parser.error("use at least 20 bootstrap batches")
    result = compute(args.batches)
    THRESHOLDS_PATH.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["thresholds"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
