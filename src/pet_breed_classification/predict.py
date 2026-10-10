"""ONNX-backed single and batch prediction."""

from __future__ import annotations

import functools
import hmac
import json
import logging
import math
import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ParamSpec, TypeVar

import numpy as np
import onnxruntime as ort
from PIL import Image

from .artifacts import _sha256
from .calibration import FloatArray, softmax
from .config import cfg
from .transforms import build_eval_transform

logger = logging.getLogger(__name__)
P = ParamSpec("P")
R = TypeVar("R")


def timed(function: Callable[P, R]) -> Callable[P, R]:
    """Log a wrapped function's elapsed time in milliseconds."""

    @functools.wraps(function)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        started_at = time.perf_counter()
        result = function(*args, **kwargs)
        elapsed_ms = (time.perf_counter() - started_at) * 1000
        logger.debug("%s took %.3f ms", function.__name__, elapsed_ms)
        return result

    return wrapper


UNCALIBRATED_MESSAGE = (
    "abstain_threshold is null in model_meta.json: the model has no calibrated "
    "abstention threshold. Run `uv run python -m pet_breed_classification.calibrate fit` "
    "to fit the temperature and threshold on the validation split."
)


class UncalibratedModelError(RuntimeError):
    """Raised when the served model has no calibrated abstention threshold."""


def _check_temperature(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"temperature must be a finite positive number, got {value!r}")
    temperature = float(value)
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError(f"temperature must be a finite positive number, got {value!r}")
    return temperature


def _check_threshold(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(
            f"abstain_threshold must be a finite number in [0, 1], got {value!r}"
        )
    threshold = float(value)
    if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError(
            f"abstain_threshold must be a finite number in [0, 1], got {value!r}"
        )
    return threshold


def _validate_metadata(
    metadata: object,
    *,
    require_calibration: bool,
) -> dict[str, Any]:
    """Validate the metadata required to serve one ONNX artifact."""
    if not isinstance(metadata, dict):
        raise TypeError("model metadata must be a JSON object")

    required_fields = {
        "model_version",
        "backbone",
        "framework",
        "opset",
        "classes",
        "eval_transform",
        "temperature",
        "abstain_threshold",
        "onnx_sha256",
        "trained_at",
        "exported_at",
    }
    missing_fields = sorted(required_fields.difference(metadata))
    if missing_fields:
        raise ValueError(f"model metadata is missing fields: {missing_fields}")

    for field in ("model_version", "backbone", "framework", "trained_at", "exported_at"):
        value = metadata[field]
        if not isinstance(value, str) or not value:
            raise ValueError(f"metadata {field} must be a non-empty string")

    opset = metadata["opset"]
    if isinstance(opset, bool) or not isinstance(opset, int) or opset <= 0:
        raise ValueError(f"metadata opset must be a positive integer, got {opset!r}")

    expected_sha = metadata["onnx_sha256"]
    if not isinstance(expected_sha, str) or re.fullmatch(r"[0-9a-fA-F]{64}", expected_sha) is None:
        raise ValueError("metadata onnx_sha256 must be a 64-character hexadecimal digest")

    classes = metadata["classes"]
    if not isinstance(classes, list) or len(classes) < 3:
        raise ValueError("metadata classes must contain at least three entries")
    indices: set[int] = set()
    for entry in classes:
        if not isinstance(entry, dict):
            raise TypeError("metadata class entries must be JSON objects")
        index = entry.get("index")
        if isinstance(index, bool) or not isinstance(index, int) or index in indices:
            raise ValueError("metadata class indices must be unique integers")
        if not isinstance(entry.get("breed"), str) or not entry["breed"]:
            raise ValueError("metadata class breed must be a non-empty string")
        if not isinstance(entry.get("species"), str) or not entry["species"]:
            raise ValueError("metadata class species must be a non-empty string")
        indices.add(index)
    if indices != set(range(len(classes))):
        raise ValueError("metadata class indices must be contiguous from zero")

    if not isinstance(metadata["eval_transform"], dict):
        raise TypeError("metadata eval_transform must be a JSON object")
    _check_temperature(metadata["temperature"])

    threshold = metadata["abstain_threshold"]
    if threshold is None:
        if require_calibration:
            raise UncalibratedModelError(UNCALIBRATED_MESSAGE)
    else:
        _check_threshold(threshold)
    return metadata


@dataclass(frozen=True)
class Prediction:
    """Prediction result returned by the predictor."""

    breed: str
    species: str
    confidence: float
    top_3: list[tuple[str, float]]
    decision: str


class PetBreedPredictor:
    """Load ONNX Runtime once and expose one inference path."""

    def __init__(
        self,
        session: ort.InferenceSession,
        metadata: dict[str, Any],
        *,
        require_calibration: bool = True,
    ) -> None:
        metadata = _validate_metadata(
            metadata,
            require_calibration=require_calibration,
        )
        self.session = session
        self._metadata = metadata
        self.transform = build_eval_transform(metadata["eval_transform"])
        self.input_name = session.get_inputs()[0].name
        self.classes = metadata["classes"]
        self.classes_by_index = {
            int(entry["index"]): entry for entry in self.classes
        }
        self.temperature = _check_temperature(metadata["temperature"])
        threshold = metadata["abstain_threshold"]
        if threshold is None and require_calibration:
            raise UncalibratedModelError(UNCALIBRATED_MESSAGE)
        self.abstain_threshold = None if threshold is None else _check_threshold(threshold)

    @property
    def metadata(self) -> dict[str, Any]:
        """Return the metadata used to serve this model."""
        return self._metadata

    @classmethod
    def load(
        cls,
        onnx_path: Path | None = None,
        meta_path: Path | None = None,
        *,
        require_calibration: bool = True,
    ) -> PetBreedPredictor:
        """Load and validate the ONNX model and its metadata.

        With ``require_calibration`` (the default, used for serving), a null
        ``abstain_threshold`` raises ``UncalibratedModelError``. The calibration command
        passes False so it can read uncalibrated logits.
        """
        model_path = onnx_path or cfg.ONNX_PATH
        metadata_path = meta_path or cfg.MODEL_META_PATH
        try:
            raw_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exception:
            raise ValueError(f"invalid model metadata JSON: {metadata_path}") from exception
        metadata = _validate_metadata(
            raw_metadata,
            require_calibration=require_calibration,
        )
        expected_sha = metadata["onnx_sha256"]
        actual_sha = _sha256(model_path)
        if not hmac.compare_digest(actual_sha, expected_sha.lower()):
            raise ValueError(
                f"ONNX SHA-256 mismatch: expected {expected_sha}, got {actual_sha}"
            )
        session = ort.InferenceSession(
            str(model_path), providers=["CPUExecutionProvider"]
        )
        return cls(session, metadata, require_calibration=require_calibration)

    def raw_logits(self, images: list[Image.Image]) -> FloatArray:
        """Return uncalibrated logits, shape (N, C), from one ONNX Runtime invocation.

        This is the single inference path: serving and calibration both call it.
        """
        if not images:
            raise ValueError("images must not be empty")
        tensors = [self.transform(image) for image in images]
        batch = np.stack([tensor.numpy() for tensor in tensors]).astype(
            np.float32, copy=False
        )
        logger.debug("Input tensor prepared", extra={"input_tensor_shape": list(batch.shape)})
        logits = self.session.run(["logits"], {self.input_name: batch})[0]
        logits_array = np.asarray(logits, dtype=np.float64)
        expected_shape = (len(images), len(self.classes))
        if logits_array.shape != expected_shape:
            raise ValueError(
                f"ONNX logits shape must be {expected_shape}, got {logits_array.shape}"
            )
        if not np.all(np.isfinite(logits_array)):
            raise ValueError("ONNX logits must be finite")
        return logits_array

    def probabilities(self, images: list[Image.Image]) -> FloatArray:
        """Return calibrated probabilities softmax(logits / temperature), shape (N, C)."""
        return softmax(self.raw_logits(images), self.temperature)

    @timed
    def predict_one(self, image: Image.Image) -> Prediction:
        """Predict one image through the batch inference path."""
        return self.predict_batch([image])[0]

    def predict_batch(self, images: list[Image.Image]) -> list[Prediction]:
        """Predict a batch of images with one ONNX Runtime invocation.

        ``decision`` is "confident" when the top-1 calibrated probability is greater than
        or equal to the abstention threshold, and "uncertain" otherwise.
        """
        threshold = self.abstain_threshold
        if threshold is None:
            raise UncalibratedModelError(UNCALIBRATED_MESSAGE)
        probabilities = self.probabilities(images)
        top_indices = np.argsort(-probabilities, axis=1, kind="stable")[:, :3]

        predictions: list[Prediction] = []
        for row_probabilities, row_indices in zip(
            probabilities, top_indices, strict=True
        ):
            top_3 = [
                (
                    str(self.classes_by_index[int(index)]["breed"]),
                    float(row_probabilities[int(index)]),
                )
                for index in row_indices
            ]
            top_index = int(row_indices[0])
            confidence = float(row_probabilities[top_index])
            decision = "confident" if confidence >= threshold else "uncertain"
            top_class = self.classes_by_index[top_index]
            predictions.append(
                Prediction(
                    breed=str(top_class["breed"]),
                    species=str(top_class["species"]),
                    confidence=confidence,
                    top_3=top_3,
                    decision=decision,
                )
            )
        return predictions
