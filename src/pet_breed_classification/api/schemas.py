"""Pydantic response schemas for the prediction API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ClassProbability(BaseModel):
    """One breed and its probability."""

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [{"breed": "Abyssinian", "probability": 0.91}]
        }
    )

    breed: str
    probability: float = Field(ge=0, le=1)


class BreedPrediction(BaseModel):
    """A single breed prediction."""

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "breed": "Abyssinian",
                    "species": "cat",
                    "confidence": 0.91,
                    "top_3": [
                        {"breed": "Abyssinian", "probability": 0.91},
                        {"breed": "Bengal", "probability": 0.04},
                        {"breed": "Sphynx", "probability": 0.02},
                    ],
                    "decision": "confident",
                }
            ]
        }
    )

    breed: str
    species: str
    confidence: float = Field(ge=0, le=1)
    top_3: list[ClassProbability]
    decision: Literal["confident", "uncertain"]


class PredictionResponse(BreedPrediction):
    """Single prediction response with request metadata."""

    model_version: str
    correlation_id: str
    latency_ms: float = Field(ge=0)


class BatchPredictionResponse(BaseModel):
    """Batch prediction response with request metadata."""

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "predictions": [],
                    "model_version": "v1",
                    "correlation_id": "00000000-0000-0000-0000-000000000000",
                    "latency_ms": 12.3,
                }
            ]
        }
    )

    predictions: list[BreedPrediction]
    model_version: str
    correlation_id: str
    latency_ms: float = Field(ge=0)


class HealthResponse(BaseModel):
    """Health response for a loaded model."""

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {"status": "healthy", "model_version": "v1", "num_classes": 37}
            ]
        }
    )

    status: str
    model_version: str
    num_classes: int


class MetadataResponse(BaseModel):
    """Serving metadata without the full classes list."""

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "model_version": "v1",
                    "backbone": "resnet50",
                    "framework": "onnxruntime",
                    "opset": 17,
                    "num_classes": 37,
                    "eval_transform": {"resize_size": 256, "crop_size": 224},
                    "temperature": 1.5,
                    "abstain_threshold": 0.9,
                    "onnx_sha256": "...",
                    "trained_at": "2026-10-03T12:00:00Z",
                    "exported_at": "2026-10-03T12:01:00Z",
                }
            ]
        }
    )

    model_version: str
    backbone: str
    framework: str
    opset: int
    num_classes: int
    eval_transform: dict[str, object]
    temperature: float = Field(gt=0)
    abstain_threshold: float = Field(ge=0, le=1)
    onnx_sha256: str
    trained_at: str
    exported_at: str
