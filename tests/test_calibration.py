"""Hermetic tests for the pure calibration functions. No data, no files, no network."""

from __future__ import annotations

import math

import numpy as np
import pytest

from pet_breed_classification.calibration import (
    ThresholdSelection,
    accepted_counts,
    coverage_and_selective_accuracy,
    expected_calibration_error,
    fit_temperature,
    negative_log_likelihood,
    reliability_bins,
    select_threshold,
    softmax,
)


def _rows_from_confidence(
    confidences: list[float], correct: list[bool]
) -> tuple[np.ndarray, np.ndarray]:
    """Two-class rows whose top-1 probability is the given confidence (above 0.5)."""
    probabilities = np.array([[value, 1.0 - value] for value in confidences])
    labels = np.array([0 if right else 1 for right in correct], dtype=np.int64)
    return probabilities, labels


def test_softmax_rows_sum_to_one_and_temperature_flattens() -> None:
    logits = np.array([[2.0, 0.0, -1.0], [0.5, 0.5, 0.5]])

    sharp = softmax(logits, 0.5)
    flat = softmax(logits, 4.0)

    assert np.allclose(sharp.sum(axis=1), 1.0)
    assert np.allclose(flat.sum(axis=1), 1.0)
    assert sharp[0].max() > flat[0].max()
    assert np.array_equal(sharp.argmax(axis=1), logits.argmax(axis=1))


@pytest.mark.parametrize("temperature", [0.0, -1.0, math.inf, math.nan])
def test_non_positive_temperature_is_rejected(temperature: float) -> None:
    with pytest.raises(ValueError, match="temperature"):
        softmax(np.zeros((1, 2)), temperature)


def test_negative_log_likelihood_matches_hand_value() -> None:
    logits = np.array([[0.0, 0.0]])
    labels = np.array([0])

    assert negative_log_likelihood(logits, labels) == pytest.approx(math.log(2.0))


def test_fit_temperature_sharpens_overconfident_logits() -> None:
    rng = np.random.default_rng(0)
    base = rng.normal(size=(4000, 5))
    cumulative = softmax(base).cumsum(axis=1)
    labels = (cumulative > rng.random((4000, 1))).argmax(axis=1).astype(np.int64)
    logits = 3.0 * base  # the model is three times too confident

    temperature = fit_temperature(logits, labels)

    assert temperature > 1.0
    assert abs(temperature - 3.0) < 0.5
    assert negative_log_likelihood(
        logits, labels, temperature
    ) < negative_log_likelihood(logits, labels, 1.0)
    assert np.array_equal(logits.argmax(axis=1), (logits / temperature).argmax(axis=1))


def test_fit_temperature_is_near_one_for_calibrated_logits() -> None:
    rng = np.random.default_rng(1)
    logits = rng.normal(size=(6000, 5))
    cumulative = softmax(logits).cumsum(axis=1)
    labels = (cumulative > rng.random((6000, 1))).argmax(axis=1).astype(np.int64)

    assert abs(fit_temperature(logits, labels) - 1.0) < 0.1


def test_fit_temperature_is_deterministic() -> None:
    rng = np.random.default_rng(2)
    logits = rng.normal(size=(300, 4)) * 2.0
    labels = rng.integers(0, 4, size=300)

    assert fit_temperature(logits, labels) == fit_temperature(logits, labels)


def test_fit_temperature_rejects_bad_shapes() -> None:
    with pytest.raises(ValueError):
        fit_temperature(np.zeros((3, 2)), np.zeros(2, dtype=np.int64))
    with pytest.raises(ValueError):
        fit_temperature(np.zeros((0, 2)), np.zeros(0, dtype=np.int64))


def test_expected_calibration_error_matches_hand_computation() -> None:
    probabilities = np.array([[0.95, 0.05], [0.95, 0.05], [0.35, 0.65], [0.35, 0.65]])
    labels = np.array([0, 1, 1, 1])

    # Bin 9 ([0.9, 1.0)): 2 samples, accuracy 0.5, confidence 0.95 -> gap 0.45, weight 0.5.
    # Bin 6 ([0.6, 0.7)): 2 samples, accuracy 1.0, confidence 0.65 -> gap 0.35, weight 0.5.
    assert expected_calibration_error(
        probabilities, labels, n_bins=10
    ) == pytest.approx(0.4)


def test_expected_calibration_error_is_zero_for_calibrated_toy_data() -> None:
    coin = np.full((4, 2), 0.5)
    coin_labels = np.array([0, 1, 0, 1])
    one_hot = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
    one_hot_labels = np.array([0, 1, 0])

    assert expected_calibration_error(coin, coin_labels) == pytest.approx(0.0)
    assert expected_calibration_error(one_hot, one_hot_labels) == pytest.approx(0.0)


def test_reliability_bins_report_empty_bins_as_nan() -> None:
    probabilities, labels = _rows_from_confidence(
        [0.95, 0.95, 0.65], [True, False, True]
    )

    bins = reliability_bins(probabilities, labels, n_bins=10)

    assert bins.counts.sum() == 3
    assert bins.counts[9] == 2
    assert bins.counts[6] == 1
    assert bins.accuracy[9] == pytest.approx(0.5)
    assert math.isnan(bins.accuracy[0])
    assert bins.edges[0] == 0.0 and bins.edges[-1] == 1.0


def test_empty_probability_metrics_are_defined() -> None:
    probabilities = np.empty((0, 3), dtype=np.float64)
    labels = np.empty(0, dtype=np.int64)

    bins = reliability_bins(probabilities, labels, n_bins=4)

    assert np.array_equal(bins.counts, np.zeros(4, dtype=np.int64))
    assert np.all(np.isnan(bins.confidence))
    assert np.all(np.isnan(bins.accuracy))
    assert expected_calibration_error(probabilities, labels, n_bins=4) == 0.0
    assert accepted_counts(probabilities, labels, 0.9) == (0, 0)
    assert coverage_and_selective_accuracy(probabilities, labels, 0.9) == (0.0, None)


def test_coverage_and_selective_accuracy_hand_example() -> None:
    probabilities, labels = _rows_from_confidence(
        [0.9, 0.8, 0.7, 0.6], [True, False, True, False]
    )

    assert coverage_and_selective_accuracy(probabilities, labels, 0.7) == (
        pytest.approx(0.75),
        pytest.approx(2 / 3),
    )
    assert coverage_and_selective_accuracy(probabilities, labels, 0.85) == (
        pytest.approx(0.25),
        pytest.approx(1.0),
    )
    assert coverage_and_selective_accuracy(probabilities, labels, 0.95) == (0.0, None)


def test_threshold_is_the_lowest_one_that_meets_the_target() -> None:
    probabilities, labels = _rows_from_confidence(
        [0.9, 0.8, 0.7, 0.6], [True, False, True, False]
    )

    at_half = select_threshold(probabilities, labels, 0.5)
    at_two_thirds = select_threshold(probabilities, labels, 0.6)
    at_one = select_threshold(probabilities, labels, 1.0)

    assert at_half.threshold == pytest.approx(0.6)
    assert at_half.coverage == pytest.approx(1.0)
    assert at_two_thirds.threshold == pytest.approx(0.7)
    assert at_two_thirds.coverage == pytest.approx(0.75)
    assert at_two_thirds.selective_accuracy == pytest.approx(2 / 3)
    assert at_one.threshold == pytest.approx(0.9)
    assert at_one.coverage == pytest.approx(0.25)


def test_unattainable_target_returns_none_with_reason() -> None:
    probabilities, labels = _rows_from_confidence([0.9, 0.8], [False, False])

    selection = select_threshold(probabilities, labels, 0.5)

    assert selection == ThresholdSelection(
        threshold=None,
        coverage=None,
        selective_accuracy=None,
        reason=selection.reason,
    )
    assert "unattainable" in selection.reason
    assert "0.000000" in selection.reason


def test_ties_are_accepted_together() -> None:
    probabilities, labels = _rows_from_confidence(
        [0.7, 0.7, 0.7, 0.6], [True, False, True, False]
    )

    selection = select_threshold(probabilities, labels, 0.6)

    assert selection.threshold == pytest.approx(0.7)
    assert selection.coverage == pytest.approx(0.75)
    assert selection.selective_accuracy == pytest.approx(2 / 3)


def test_empty_input_has_no_threshold() -> None:
    selection = select_threshold(np.zeros((0, 3)), np.zeros(0, dtype=np.int64), 0.9)

    assert selection.threshold is None
    assert selection.reason == "no validation samples"


@pytest.mark.parametrize("target", [0.0, -0.1, 1.5])
def test_target_outside_unit_interval_is_rejected(target: float) -> None:
    probabilities, labels = _rows_from_confidence([0.9], [True])

    with pytest.raises(ValueError, match="target"):
        select_threshold(probabilities, labels, target)


def test_threshold_rises_monotonically_with_the_target() -> None:
    rng = np.random.default_rng(3)
    logits = rng.normal(size=(400, 4)) * 2.0
    probabilities = softmax(logits, 1.0)
    labels = (probabilities.cumsum(axis=1) > rng.random((400, 1))).argmax(axis=1)

    targets = np.linspace(0.3, 1.0, 36)
    thresholds = [
        select_threshold(probabilities, labels, float(target)).threshold
        for target in targets
    ]

    seen_none = False
    previous: float | None = None
    for threshold in thresholds:
        if threshold is None:
            seen_none = True
            continue
        assert not seen_none, "a higher target cannot become attainable again"
        if previous is not None:
            assert threshold >= previous
        previous = threshold
