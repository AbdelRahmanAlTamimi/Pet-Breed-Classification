"""ONNX-backed single and batch prediction."""

from __future__ import annotations

import functools
import json
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ParamSpec, TypeVar

import numpy as np
import onnxruntime as ort
from PIL import Image

from .artifacts import _sha256
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
    ) -> None:
        self.session = session
        self._metadata = metadata
        self.transform = build_eval_transform(metadata["eval_transform"])
        self.input_name = session.get_inputs()[0].name
        self.classes = metadata["classes"]
        self.classes_by_index = {
            int(entry["index"]): entry for entry in self.classes
        }
        self.temperature = float(metadata["temperature"])
        self.abstain_threshold = metadata["abstain_threshold"]

    @property
    def metadata(self) -> dict[str, Any]:
        """Return the metadata used to serve this model."""
        return self._metadata

    @classmethod
    def load(
        cls,
        onnx_path: Path | None = None,
        meta_path: Path | None = None,
    ) -> PetBreedPredictor:
        """Load and validate the ONNX model and its metadata."""
        model_path = onnx_path or cfg.ONNX_PATH
        metadata_path = meta_path or cfg.MODEL_META_PATH
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        expected_sha = str(metadata["onnx_sha256"])
        actual_sha = _sha256(model_path)
        if actual_sha != expected_sha:
            raise ValueError(
                f"ONNX SHA-256 mismatch: expected {expected_sha}, got {actual_sha}"
            )
        session = ort.InferenceSession(
            str(model_path), providers=["CPUExecutionProvider"]
        )
        return cls(session, metadata)

    @timed
    def predict_one(self, image: Image.Image) -> Prediction:
        """Predict one image through the batch inference path."""
        return self.predict_batch([image])[0]

    def predict_batch(self, images: list[Image.Image]) -> list[Prediction]:
        """Predict a batch of images with one ONNX Runtime invocation."""
        if not images:
            raise ValueError("images must not be empty")

        tensors = [self.transform(image) for image in images]
        batch = np.stack([tensor.numpy() for tensor in tensors]).astype(
            np.float32, copy=False
        )
        logger.debug("Input tensor prepared", extra={"input_tensor_shape": list(batch.shape)})
        logits = self.session.run(["logits"], {self.input_name: batch})[0]
        scaled = logits / self.temperature
        scaled -= np.max(scaled, axis=1, keepdims=True)
        probabilities = np.exp(scaled)
        probabilities /= np.sum(probabilities, axis=1, keepdims=True)
        top_indices = np.argsort(-probabilities, axis=1)[:, :3]

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
            decision = (
                "uncertain"
                if self.abstain_threshold is not None
                and confidence < float(self.abstain_threshold)
                else "confident"
            )
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
