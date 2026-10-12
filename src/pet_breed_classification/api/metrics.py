"""Prometheus metrics exposed by the FastAPI service at ``/metrics``."""

from __future__ import annotations

from prometheus_client import Counter, Histogram

from ..predict import Prediction

REQUEST_DURATION = Histogram(
    "petbreed_request_duration_seconds",
    "HTTP request duration in seconds.",
    ["endpoint", "status_class"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0),
)
PREDICTION_CONFIDENCE = Histogram(
    "petbreed_prediction_confidence",
    "Calibrated top-1 probability of served predictions.",
    buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99, 1.0),
)
PREDICTIONS = Counter(
    "petbreed_predictions",
    "Served predictions by abstention decision.",
    ["decision"],
)


def record_prediction(prediction: Prediction) -> None:
    """Count one served prediction and observe its calibrated confidence."""
    PREDICTION_CONFIDENCE.observe(prediction.confidence)
    PREDICTIONS.labels(decision=prediction.decision).inc()


def status_class(status_code: int) -> str:
    """Return the status class label, for example ``2xx``."""
    return f"{status_code // 100}xx"
