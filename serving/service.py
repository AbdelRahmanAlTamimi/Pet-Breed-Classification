"""BentoML legacy-Runner service for the calibrated ONNX pet-breed classifier."""

from __future__ import annotations

import io
import logging
import time
import uuid
import warnings
from pathlib import Path
from typing import Any, BinaryIO

from bentoml import legacy
from bentoml.exceptions import BadInput, ServiceUnavailable
from bentoml.io import JSON, File
from fastapi import FastAPI, HTTPException
from PIL import Image, UnidentifiedImageError

from pet_breed_classification.config import cfg
from pet_breed_classification.predict import PetBreedPredictor, Prediction

logger = logging.getLogger(__name__)
warnings.filterwarnings("error", category=Image.DecompressionBombWarning)

MAX_BATCH_SIZE = 8
# Above the per-batch latency under load; at 50 ms the dispatcher shed 8.6% of
# requests as "Service Busy" in the first Locust run (reports/locust_bentoml_run1_*).
MAX_LATENCY_MS = 500
_health_predictor: PetBreedPredictor | None = None


def _bad_input(message: str) -> BadInput:
    """Build a BentoML 400-level input error."""
    return BadInput(message)


def _decode(upload: Path | BinaryIO | bytes | bytearray) -> Image.Image:
    """Read and validate one upload, preserving the FastAPI safety rules."""
    if isinstance(upload, (bytes, bytearray)):
        contents = bytes(upload)
    elif isinstance(upload, Path):
        if upload.stat().st_size > cfg.MAX_UPLOAD_BYTES:
            raise _bad_input(f"The uploaded file exceeds {cfg.MAX_UPLOAD_BYTES} bytes")
        contents = upload.read_bytes()
    elif hasattr(upload, "read"):
        contents = upload.read(cfg.MAX_UPLOAD_BYTES + 1)
    else:
        raise _bad_input("The uploaded file is not readable")

    if len(contents) > cfg.MAX_UPLOAD_BYTES:
        raise _bad_input(f"The uploaded file exceeds {cfg.MAX_UPLOAD_BYTES} bytes")
    if not contents:
        raise _bad_input("The uploaded file is empty")

    try:
        with Image.open(io.BytesIO(contents)) as image:
            image.load()
            decoded = image.convert("RGB")
    except (Image.DecompressionBombError, Image.DecompressionBombWarning):
        raise _bad_input("The image was rejected by Pillow's decompression-bomb guard")
    except (UnidentifiedImageError, OSError, ValueError) as exception:
        raise _bad_input(f"The uploaded bytes are not a valid image: {exception}") from exception
    return decoded


def _response(
    prediction: Prediction,
    model_version: str,
    latency_ms: float,
) -> dict[str, Any]:
    """Return the same prediction fields as the FastAPI endpoint."""
    return {
        "breed": prediction.breed,
        "species": prediction.species,
        "confidence": prediction.confidence,
        "top_3": [
            {"breed": breed, "probability": probability}
            for breed, probability in prediction.top_3
        ],
        "decision": prediction.decision,
        "model_version": model_version,
        "correlation_id": str(uuid.uuid4()),
        "latency_ms": latency_ms,
    }


def _predict_many(
    predictor: PetBreedPredictor | None,
    uploads: list[Path | BinaryIO | bytes | bytearray],
) -> list[dict[str, Any]]:
    """Decode uploads and score them with one shared-predictor batch call."""
    if predictor is None:
        raise ServiceUnavailable("Model is unavailable")
    images = [_decode(upload) for upload in uploads]
    started_at = time.perf_counter()
    predictions = predictor.predict_batch(images)
    latency_ms = (time.perf_counter() - started_at) * 1000
    model_version = str(predictor.metadata["model_version"])
    return [_response(prediction, model_version, latency_ms) for prediction in predictions]


class PetBreedRunnable(legacy.Runnable):
    """Load one predictor per Bento worker and batch its image requests."""

    SUPPORTED_RESOURCES = ("cpu",)
    SUPPORTS_CPU_MULTI_THREADING = True

    def __init__(self) -> None:
        global _health_predictor
        try:
            self.predictor: PetBreedPredictor | None = PetBreedPredictor.load()
            _health_predictor = self.predictor
            logger.info(
                "Bento predictor loaded",
                extra={
                    "model_version": self.predictor.metadata["model_version"],
                    "num_classes": len(self.predictor.classes),
                },
            )
        except Exception:
            self.predictor = None
            _health_predictor = None
            logger.exception("Bento predictor could not be loaded")

    @legacy.Runnable.method(batchable=True)
    def predict(
        self,
        uploads: list[Path | BinaryIO | bytes | bytearray],
    ) -> list[dict[str, Any]]:
        """Score one adaptive batch in a single ONNX Runtime call."""
        return _predict_many(self.predictor, uploads)

    @legacy.Runnable.method
    def metadata(self) -> dict[str, Any]:
        """Return health metadata from the same loaded Runner predictor."""
        if self.predictor is None:
            raise ServiceUnavailable("Model is unavailable")
        return {
            "status": "healthy",
            "model_version": str(self.predictor.metadata["model_version"]),
            "num_classes": len(self.predictor.classes),
        }


runner = legacy.Runner(
    PetBreedRunnable,
    name="pet-breed-runner",
    max_batch_size=MAX_BATCH_SIZE,
    max_latency_ms=MAX_LATENCY_MS,
)
svc = legacy.Service("pet-breed-classifier", runners=[runner])


@svc.api(input=File(), output=JSON(), route="/predict")
async def predict(file: BinaryIO) -> dict[str, Any]:
    """Predict one uploaded image through the batchable Runner."""
    contents = file.read(cfg.MAX_UPLOAD_BYTES + 1)
    _decode(contents)
    responses = await runner.predict.async_run([contents])
    return responses[0]


health_app = FastAPI()


@health_app.get("/health")
async def health() -> dict[str, Any]:
    """Return healthy only after the Runner has loaded the predictor."""
    try:
        return await runner.metadata.async_run()
    except Exception as exception:
        logger.exception("Bento health check failed")
        raise HTTPException(status_code=503, detail="Model is unavailable") from exception


svc.mount_asgi_app(health_app, path="/")
