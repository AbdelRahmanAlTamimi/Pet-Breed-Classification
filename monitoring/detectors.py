"""Batch drift detectors scored against a clean reference.

- ``embedding``: domain classifier (PCA-50 + logistic regression) separating reference
  from batch embeddings; the statistic is 5-fold cross-validated ROC-AUC (0.5 = no drift).
- ``pixel``: two-sample KS test per pixel statistic; the score is -log10 of the smallest
  p-value, and the alarm rule is min p < 0.01 / 3 (Bonferroni over three statistics).
- ``confidence``: KS statistic between batch and reference calibrated max probability,
  tracked separately from input drift together with the abstain rate.

Thresholds for ``embedding`` and ``confidence`` come only from clean data (see
``monitoring.thresholds``).

``embedding_valref`` is a post-hoc variant of ``embedding``, added after the clean-test
scorecard: the same domain classifier, but with a seeded val half as the reference
instead of train images. It is scored in the scorecard only, not in the live check.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.stats import ks_2samp
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .features import PIXEL_STATS, Features

DETECTORS = ("embedding", "pixel", "confidence")
EMBEDDING_VALREF = "embedding_valref"
PCA_COMPONENTS = 50
CV_FOLDS = 5
PIXEL_ALPHA = 0.01 / len(PIXEL_STATS)
# KS p-values underflow to 0 for strong shifts; cap the -log10 score.
MIN_P_VALUE = 1e-300


def embedding_auc(
    reference: NDArray[np.floating], batch: NDArray[np.floating], seed: int
) -> float:
    """Return the cross-validated ROC-AUC of a reference-vs-batch domain classifier."""
    features = np.concatenate([reference, batch])
    labels = np.concatenate([np.zeros(len(reference)), np.ones(len(batch))])
    components = min(PCA_COMPONENTS, len(features) - len(features) // CV_FOLDS - 1)
    classifier = make_pipeline(
        StandardScaler(),
        PCA(n_components=components, random_state=seed),
        LogisticRegression(max_iter=2000, class_weight="balanced"),
    )
    folds = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=seed)
    return float(cross_val_score(classifier, features, labels, cv=folds, scoring="roc_auc").mean())


def pixel_p_values(reference: Features, batch: Features) -> dict[str, float]:
    """Return the two-sample KS p-value of each pixel statistic."""
    return {name: float(ks_2samp(reference[name], batch[name]).pvalue) for name in PIXEL_STATS}


def pixel_score(p_values: dict[str, float]) -> float:
    return -math.log10(max(min(p_values.values()), MIN_P_VALUE))


def confidence_ks(reference: NDArray[np.floating], batch: NDArray[np.floating]) -> float:
    return float(ks_2samp(reference, batch).statistic)


def check_batch(
    batch: Features,
    reference: Features,
    confidence_reference: NDArray[np.floating],
    thresholds: dict[str, float],
    abstain_threshold: float,
    seed: int,
) -> dict[str, Any]:
    """Score one batch with every detector and compare each score with its threshold."""
    p_values = pixel_p_values(reference, batch)
    scores = {
        "embedding": embedding_auc(reference["embedding"], batch["embedding"], seed),
        "pixel": pixel_score(p_values),
        "confidence": confidence_ks(confidence_reference, batch["max_confidence"]),
    }
    return {
        "n": len(batch["embedding"]),
        "scores": scores,
        "thresholds": dict(thresholds),
        "alarms": {name: bool(scores[name] > thresholds[name]) for name in DETECTORS},
        "pixel_p_values": p_values,
        "abstain_rate": float(np.mean(batch["max_confidence"] < abstain_threshold)),
    }
