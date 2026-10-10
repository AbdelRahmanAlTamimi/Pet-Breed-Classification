"""Temperature scaling, calibration metrics, and abstention thresholds.

Every function here is pure: arrays and numbers in, values out. Nothing reads images,
manifests, or model files, and the module depends on NumPy only, so the API can import it.
The calibration command fits these values on the validation split only.

Conventions
- Probabilities are softmax(logits / T). Temperature scaling never changes the argmax.
- Confidence is the top-1 probability. A prediction is accepted ("confident") when its
  confidence is greater than or equal to the threshold, and abstained otherwise.
- ECE uses left-closed bins [k / n, (k + 1) / n); the top bin also contains 1.0.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]

TEMPERATURE_MIN = 0.05
TEMPERATURE_MAX = 20.0
_GRID_SIZE = 401
_REFINE_TOLERANCE = 1e-9
_REFINE_MAX_ITERATIONS = 200
_INVERSE_PHI = (math.sqrt(5.0) - 1.0) / 2.0


@dataclass(frozen=True)
class ReliabilityBins:
    """Per-bin counts, mean confidence and accuracy. Empty bins hold NaN."""

    edges: FloatArray
    counts: IntArray
    confidence: FloatArray
    accuracy: FloatArray


@dataclass(frozen=True)
class ThresholdSelection:
    """Chosen threshold, or None with the reason no threshold qualifies."""

    threshold: float | None
    coverage: float | None
    selective_accuracy: float | None
    reason: str


def _check_temperature(temperature: float) -> float:
    value = float(temperature)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(
            f"temperature must be a finite positive number, got {temperature!r}"
        )
    return value


def _check_score_matrix(scores: FloatArray, name: str) -> FloatArray:
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] < 1:
        raise ValueError(f"{name} must have shape (N, C) with C >= 1")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must be finite")
    return values


def _check_labels(labels: IntArray, sample_count: int, class_count: int) -> IntArray:
    values = np.asarray(labels)
    if values.ndim != 1 or len(values) != sample_count:
        raise ValueError("labels must have shape (N,) matching the score rows")
    if len(values) == 0:
        return values.astype(np.int64, copy=False)
    if not np.issubdtype(values.dtype, np.integer):
        raise ValueError("labels must contain integer class indices")
    labels64 = values.astype(np.int64, copy=False)
    if len(labels64) and (
        int(labels64.min()) < 0 or int(labels64.max()) >= class_count
    ):
        raise ValueError("labels must be class indices within the score width")
    return labels64


def _check_logits(logits: FloatArray, labels: IntArray) -> tuple[FloatArray, IntArray]:
    values = _check_score_matrix(logits, "logits")
    labels64 = _check_labels(labels, len(values), values.shape[1])
    if len(labels64) == 0:
        raise ValueError("at least one sample is required")
    return values, labels64


def _check_probabilities(
    probabilities: FloatArray, labels: IntArray
) -> tuple[FloatArray, IntArray]:
    values = _check_score_matrix(probabilities, "probabilities")
    labels64 = _check_labels(labels, len(values), values.shape[1])
    if np.any(values < -1e-12) or np.any(values > 1.0 + 1e-12):
        raise ValueError("probabilities must be between 0 and 1")
    if not np.allclose(values.sum(axis=1), 1.0, rtol=1e-6, atol=1e-8):
        raise ValueError("probability rows must sum to 1")
    return values, labels64


def softmax(logits: FloatArray, temperature: float = 1.0) -> FloatArray:
    """Return row-wise softmax(logits / temperature), computed in float64."""
    values = _check_score_matrix(logits, "logits")
    temperature_value = _check_temperature(temperature)
    if len(values) == 0:
        return values.copy()
    scaled = values / temperature_value
    scaled = scaled - scaled.max(axis=1, keepdims=True)
    exponentials = np.exp(scaled)
    return exponentials / exponentials.sum(axis=1, keepdims=True)


def negative_log_likelihood(
    logits: FloatArray, labels: IntArray, temperature: float = 1.0
) -> float:
    """Return the mean negative log-likelihood of the labels under softmax(logits / T)."""
    logits64, labels64 = _check_logits(logits, labels)
    scaled = logits64 / _check_temperature(temperature)
    shifted = scaled - scaled.max(axis=1, keepdims=True)
    log_normalizer = np.log(np.exp(shifted).sum(axis=1))
    rows = np.arange(len(labels64))
    log_probability = shifted[rows, labels64] - log_normalizer
    return float(-log_probability.mean())


def _golden_section_minimize(
    objective: Callable[[float], float], lower: float, upper: float
) -> float:
    """Minimize a unimodal function on [lower, upper]; returns the midpoint of the bracket."""
    a, b = lower, upper
    c = b - _INVERSE_PHI * (b - a)
    d = a + _INVERSE_PHI * (b - a)
    value_c = objective(c)
    value_d = objective(d)
    for _ in range(_REFINE_MAX_ITERATIONS):
        if b - a < _REFINE_TOLERANCE:
            break
        if value_c <= value_d:
            b, d, value_d = d, c, value_c
            c = b - _INVERSE_PHI * (b - a)
            value_c = objective(c)
        else:
            a, c, value_c = c, d, value_d
            d = a + _INVERSE_PHI * (b - a)
            value_d = objective(d)
    return (a + b) / 2.0


def fit_temperature(logits: FloatArray, labels: IntArray) -> float:
    """Return the temperature T > 0 that minimizes validation NLL of softmax(logits / T).

    The search runs over log(T). A coarse grid over [TEMPERATURE_MIN, TEMPERATURE_MAX]
    locates the best grid point, and golden-section search refines it within the grid
    cells on either side. The result is deterministic for the same inputs.
    """
    logits64, labels64 = _check_logits(logits, labels)

    def objective(log_temperature: float) -> float:
        return negative_log_likelihood(logits64, labels64, math.exp(log_temperature))

    grid = np.linspace(math.log(TEMPERATURE_MIN), math.log(TEMPERATURE_MAX), _GRID_SIZE)
    values = np.array([objective(float(point)) for point in grid])
    best = int(np.argmin(values))
    if best == 0 or best == len(grid) - 1:
        return math.exp(float(grid[best]))
    lower = float(grid[max(best - 1, 0)])
    upper = float(grid[min(best + 1, len(grid) - 1)])
    refined = _golden_section_minimize(objective, lower, upper)
    if objective(refined) < values[best]:
        return math.exp(refined)
    return math.exp(float(grid[best]))


def _bin_indices(confidences: FloatArray, n_bins: int) -> IntArray:
    indices = np.floor(confidences * n_bins).astype(np.int64)
    return np.clip(indices, 0, n_bins - 1)


def reliability_bins(
    probabilities: FloatArray, labels: IntArray, n_bins: int = 15
) -> ReliabilityBins:
    """Bin predictions by top-1 confidence and report count, mean confidence, accuracy."""
    probabilities64, labels64 = _check_probabilities(probabilities, labels)
    if not isinstance(n_bins, int) or isinstance(n_bins, bool) or n_bins < 1:
        raise ValueError("n_bins must be at least 1")
    confidences = probabilities64.max(axis=1)
    correct = (probabilities64.argmax(axis=1) == labels64).astype(np.float64)
    indices = _bin_indices(confidences, n_bins)
    counts = np.bincount(indices, minlength=n_bins).astype(np.int64)
    confidence_sums = np.bincount(indices, weights=confidences, minlength=n_bins)
    correct_sums = np.bincount(indices, weights=correct, minlength=n_bins)
    populated = counts > 0
    confidence = np.full(n_bins, np.nan)
    accuracy = np.full(n_bins, np.nan)
    confidence[populated] = confidence_sums[populated] / counts[populated]
    accuracy[populated] = correct_sums[populated] / counts[populated]
    return ReliabilityBins(
        edges=np.linspace(0.0, 1.0, n_bins + 1),
        counts=counts,
        confidence=confidence,
        accuracy=accuracy,
    )


def expected_calibration_error(
    probabilities: FloatArray, labels: IntArray, n_bins: int = 15
) -> float:
    """Return top-label ECE: the count-weighted mean of |accuracy - confidence| over bins."""
    bins = reliability_bins(probabilities, labels, n_bins)
    populated = bins.counts > 0
    total = int(bins.counts.sum())
    if total == 0:
        return 0.0
    gaps = np.abs(bins.accuracy[populated] - bins.confidence[populated])
    return float(np.sum(bins.counts[populated] / total * gaps))


def _check_threshold(threshold: float) -> float:
    value = float(threshold)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"threshold must be finite and in [0, 1], got {threshold!r}")
    return value


def accepted_counts(
    probabilities: FloatArray, labels: IntArray, threshold: float
) -> tuple[int, int]:
    """Return (accepted, correct among accepted) for confidence >= threshold."""
    probabilities64, labels64 = _check_probabilities(probabilities, labels)
    threshold_value = _check_threshold(threshold)
    confidences = probabilities64.max(axis=1)
    correct = probabilities64.argmax(axis=1) == labels64
    accepted = confidences >= threshold_value
    return int(accepted.sum()), int(np.count_nonzero(correct & accepted))


def coverage_and_selective_accuracy(
    probabilities: FloatArray, labels: IntArray, threshold: float
) -> tuple[float, float | None]:
    """Return (coverage, selective accuracy) at a threshold.

    Coverage is the fraction of samples accepted. Selective accuracy is the accuracy on
    the accepted samples, or None when nothing is accepted.
    """
    probabilities64, labels64 = _check_probabilities(probabilities, labels)
    threshold_value = _check_threshold(threshold)
    if len(labels64) == 0:
        return 0.0, None
    confidences = probabilities64.max(axis=1)
    accepted_mask = confidences >= threshold_value
    accepted = int(accepted_mask.sum())
    correct = int(
        np.count_nonzero((probabilities64.argmax(axis=1) == labels64) & accepted_mask)
    )
    coverage = accepted / len(labels64)
    selective_accuracy = None if accepted == 0 else correct / accepted
    return coverage, selective_accuracy


def select_threshold(
    probabilities: FloatArray, labels: IntArray, target_selective_accuracy: float
) -> ThresholdSelection:
    """Return the lowest threshold whose selective accuracy meets the target.

    A lower threshold accepts more samples, so the lowest qualifying threshold gives the
    maximum coverage among qualifying thresholds. Candidates are the distinct observed
    confidences, sorted in ascending order, so every chosen threshold is a score the model
    actually produced. Samples tied at a candidate are accepted together, because acceptance
    is confidence >= threshold. Selective accuracy is not monotone in the threshold, so every
    candidate is checked. If no candidate qualifies, the threshold is None and the reason
    gives the best value.
    """
    target = float(target_selective_accuracy)
    if not math.isfinite(target) or not 0.0 < target <= 1.0:
        raise ValueError("target_selective_accuracy must be in (0, 1]")
    probabilities64, labels64 = _check_probabilities(probabilities, labels)
    if len(labels64) == 0:
        return ThresholdSelection(None, None, None, "no validation samples")

    confidences = probabilities64.max(axis=1)
    correct = probabilities64.argmax(axis=1) == labels64
    best_seen: float | None = None
    for candidate in np.unique(confidences):
        threshold = float(candidate)
        accepted_mask = confidences >= threshold
        accepted = int(accepted_mask.sum())
        correct_count = int(np.count_nonzero(correct & accepted_mask))
        selective = correct_count / accepted
        best_seen = selective if best_seen is None else max(best_seen, selective)
        if selective >= target:
            return ThresholdSelection(
                threshold=threshold,
                coverage=accepted / len(labels64),
                selective_accuracy=selective,
                reason="lowest threshold whose selective accuracy meets the target",
            )
    return ThresholdSelection(
        threshold=None,
        coverage=None,
        selective_accuracy=None,
        reason=(
            f"target {target} is unattainable: no threshold reaches it; "
            f"the best selective accuracy over all thresholds is {best_seen:.6f}"
        ),
    )
