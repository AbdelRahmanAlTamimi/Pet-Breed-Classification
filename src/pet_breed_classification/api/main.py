"""FastAPI application for ONNX pet breed prediction."""

from __future__ import annotations

import io
import logging
import time
import uuid
import warnings
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any, NoReturn

from fastapi import FastAPI, File, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from PIL import Image, UnidentifiedImageError
from starlette.concurrency import run_in_threadpool
from starlette.exceptions import HTTPException
from starlette.responses import Response

from ..config import cfg
from ..logging_conf import correlation_id_var, setup_logging
from ..predict import PetBreedPredictor, Prediction
from .schemas import (
    BatchPredictionResponse,
    BreedPrediction,
    ClassProbability,
    HealthResponse,
    MetadataResponse,
    PredictionResponse,
)

logger = logging.getLogger(__name__)
setup_logging()
warnings.filterwarnings("error", category=Image.DecompressionBombWarning)
REQUIRED_FILE = File(...)
REQUIRED_FILES = File(...)


@asynccontextmanager
async def lifespan(application: FastAPI) -> AsyncGenerator[None, None]:
    """Load the predictor once during application startup."""
    try:
        application.state.predictor = PetBreedPredictor.load()
        logger.info("Predictor loaded", extra={"model_version": cfg.MODEL_VERSION})
    except Exception:
        application.state.predictor = None
        logger.exception("Model load failure")
    yield


app = FastAPI(
    title="Pet Breed Classifier",
    version=cfg.MODEL_VERSION,
    lifespan=lifespan,
)


def _correlation_id(request: Request) -> str:
    """Return the request correlation ID."""
    return str(getattr(request.state, "correlation_id", correlation_id_var.get()))


def _error_response(
    request: Request,
    status_code: int,
    detail: object,
) -> JSONResponse:
    """Build an error response with the request ID header."""
    correlation_id = _correlation_id(request)
    return JSONResponse(
        status_code=status_code,
        content={"detail": detail, "correlation_id": correlation_id},
        headers={"X-Request-ID": correlation_id},
    )


@app.middleware("http")
async def request_context(request: Request, call_next: Any) -> Response:
    """Attach a correlation ID and log every completed request."""
    correlation_id = str(uuid.uuid4())
    request.state.correlation_id = correlation_id
    token = correlation_id_var.set(correlation_id)
    started_at = time.perf_counter()
    response: Any = None
    try:
        response = await call_next(request)
        response.headers["X-Request-ID"] = correlation_id
        return response
    except Exception:
        duration_ms = (time.perf_counter() - started_at) * 1000
        logger.info(
            "Request completed",
            extra={
                "method": request.method,
                "path": request.url.path,
                "status": 500,
                "duration_ms": duration_ms,
            },
        )
        raise
    finally:
        if response is not None:
            duration_ms = (time.perf_counter() - started_at) * 1000
            logger.info(
                "Request completed",
                extra={
                    "method": request.method,
                    "path": request.url.path,
                    "status": response.status_code,
                    "duration_ms": duration_ms,
                },
            )
        correlation_id_var.reset(token)


@app.exception_handler(RequestValidationError)
async def request_validation_handler(
    request: Request,
    exception: RequestValidationError,
) -> JSONResponse:
    """Return readable 422 validation errors."""
    detail = []
    for error in exception.errors():
        location = error.get("loc", ())
        field = str(location[-1]) if location else "request"
        detail.append({"field": field, "message": str(error.get("msg", "Invalid value"))})
    logger.error("Validation rejection", extra={"fields": detail})
    return _error_response(request, 422, detail)


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exception: HTTPException) -> JSONResponse:
    """Return HTTP errors with a correlation ID."""
    detail = exception.detail
    level = logging.ERROR if exception.status_code >= 500 else logging.WARNING
    logger.log(level, "HTTP error", extra={"status": exception.status_code})
    return _error_response(request, exception.status_code, detail)


@app.exception_handler(Exception)
async def unexpected_exception_handler(request: Request, exception: Exception) -> JSONResponse:
    """Return a safe 500 response and log the traceback."""
    correlation_id = _correlation_id(request)
    token = correlation_id_var.set(correlation_id)
    try:
        logger.error(
            "Unexpected request error",
            exc_info=(type(exception), exception, exception.__traceback__),
        )
    finally:
        correlation_id_var.reset(token)
    return _error_response(request, 500, "Internal server error")


def _validation_error(field: str, message: str) -> NoReturn:
    """Raise a FastAPI validation error for an uploaded file."""
    raise RequestValidationError(
        [{"type": "value_error", "loc": (field,), "msg": message, "input": None}]
    )


def _decode(contents: bytes, field: str) -> Image.Image:
    """Decode one upload and enforce Pillow's safety checks."""
    if not contents:
        _validation_error(field, "The uploaded file is empty")
    try:
        with Image.open(io.BytesIO(contents)) as image:
            image.load()
            decoded = image.copy()
    except (Image.DecompressionBombError, Image.DecompressionBombWarning):
        _validation_error(field, "The image was rejected by Pillow's decompression-bomb guard")
    except (UnidentifiedImageError, OSError, ValueError) as exception:
        _validation_error(field, f"The uploaded bytes are not a valid image: {exception}")

    logger.debug(
        "Image decoded",
        extra={"mode": decoded.mode, "width": decoded.width, "height": decoded.height},
    )
    if min(decoded.size) < cfg.LOW_RES_WARN_SIDE:
        logger.warning(
            "Input image is below the expected resolution",
            extra={"width": decoded.width, "height": decoded.height},
        )
    return decoded


async def _read_upload(file: UploadFile, field: str) -> bytes:
    """Read at most the configured upload limit plus one byte."""
    contents = await file.read(cfg.MAX_UPLOAD_BYTES + 1)
    if len(contents) > cfg.MAX_UPLOAD_BYTES:
        _validation_error(field, f"The uploaded file exceeds {cfg.MAX_UPLOAD_BYTES} bytes")
    return contents


def _to_breed_prediction(prediction: Prediction) -> BreedPrediction:
    """Convert the internal prediction to the API schema."""
    return BreedPrediction(
        breed=prediction.breed,
        species=prediction.species,
        confidence=prediction.confidence,
        top_3=[
            ClassProbability(breed=breed, probability=probability)
            for breed, probability in prediction.top_3
        ],
        decision=prediction.decision,
    )


def _predict_one_bytes(
    predictor: PetBreedPredictor,
    contents: bytes,
    field: str,
) -> tuple[Prediction, float]:
    """Decode and predict one upload in a worker thread."""
    image = _decode(contents, field)
    started_at = time.perf_counter()
    prediction = predictor.predict_one(image)
    latency_ms = (time.perf_counter() - started_at) * 1000
    logger.info(
        "Prediction served",
        extra={
            "breed": prediction.breed,
            "confidence": prediction.confidence,
            "decision": prediction.decision,
            "latency_ms": latency_ms,
        },
    )
    return prediction, latency_ms


def _require_predictor(request: Request) -> PetBreedPredictor:
    """Return the loaded predictor or raise service unavailable."""
    predictor = getattr(request.app.state, "predictor", None)
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model is unavailable")
    return predictor


@app.get("/health", response_model=HealthResponse)
async def health(request: Request) -> HealthResponse:
    """Report whether the model is loaded."""
    predictor = _require_predictor(request)
    return HealthResponse(
        status="healthy",
        model_version=str(predictor.metadata["model_version"]),
        num_classes=len(predictor.metadata["classes"]),
    )


@app.get("/metadata", response_model=MetadataResponse)
async def metadata(request: Request) -> MetadataResponse:
    """Return model metadata without the class list."""
    predictor = _require_predictor(request)
    values = predictor.metadata
    return MetadataResponse(
        model_version=values["model_version"],
        backbone=values["backbone"],
        framework=values["framework"],
        opset=values["opset"],
        num_classes=len(values["classes"]),
        eval_transform=values["eval_transform"],
        temperature=values["temperature"],
        abstain_threshold=values["abstain_threshold"],
        onnx_sha256=values["onnx_sha256"],
        trained_at=values["trained_at"],
        exported_at=values["exported_at"],
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(
    request: Request,
    file: UploadFile = REQUIRED_FILE,
) -> PredictionResponse:
    """Predict the breed in one uploaded image."""
    predictor = _require_predictor(request)
    contents = await _read_upload(file, "file")
    prediction, latency_ms = await run_in_threadpool(
        _predict_one_bytes, predictor, contents, "file"
    )
    return PredictionResponse(
        **_to_breed_prediction(prediction).model_dump(),
        model_version=str(predictor.metadata["model_version"]),
        correlation_id=_correlation_id(request),
        latency_ms=latency_ms,
    )


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(
    request: Request,
    files: list[UploadFile] = REQUIRED_FILES,
) -> BatchPredictionResponse:
    """Predict the breeds in a batch of uploaded images."""
    predictor = _require_predictor(request)
    if len(files) > cfg.MAX_BATCH_FILES:
        _validation_error(
            "files",
            f"The batch contains {len(files)} files; the maximum is {cfg.MAX_BATCH_FILES}",
        )
    contents = [
        await _read_upload(file, f"files[{index}]")
        for index, file in enumerate(files)
    ]
    started_at = time.perf_counter()
    images = await run_in_threadpool(
        lambda: [_decode(contents_item, f"files[{index}]") for index, contents_item in enumerate(contents)]
    )
    predictions = await run_in_threadpool(predictor.predict_batch, images)
    latency_ms = (time.perf_counter() - started_at) * 1000
    logger.info(
        "Batch prediction served",
        extra={"batch_size": len(predictions), "latency_ms": latency_ms},
    )
    for prediction in predictions:
        logger.info(
            "Prediction served",
            extra={
                "breed": prediction.breed,
                "confidence": prediction.confidence,
                "decision": prediction.decision,
            },
        )
    return BatchPredictionResponse(
        predictions=[_to_breed_prediction(prediction) for prediction in predictions],
        model_version=str(predictor.metadata["model_version"]),
        correlation_id=_correlation_id(request),
        latency_ms=latency_ms,
    )
