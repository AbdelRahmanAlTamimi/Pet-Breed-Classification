"""Small MLflow model-registry helpers for completed training runs."""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Any

from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

from .artifacts import _sha256
from .config import cfg

logger = logging.getLogger(__name__)
MODEL_NAME = "PetBreedClassifier"
PRODUCTION_ALIAS = "production"


def _client(tracking_uri: str | None = None) -> MlflowClient:
    return MlflowClient(tracking_uri or cfg.MLFLOW_TRACKING_URI)


def register_run(
    run_id: str,
    name: str = MODEL_NAME,
    *,
    tracking_uri: str | None = None,
) -> Any:
    """Register a run's ONNX artifact and tag its companion artifacts and digest."""
    client = _client(tracking_uri)
    artifact_names = {info.path for info in client.list_artifacts(run_id)}
    required = {"model.onnx", "model_meta.json", "calibration.json"}
    missing = sorted(required - artifact_names)
    if missing:
        raise FileNotFoundError(f"run {run_id} is missing artifacts: {missing}")

    with tempfile.TemporaryDirectory() as directory:
        onnx_path = Path(client.download_artifacts(run_id, "model.onnx", directory))
        digest = _sha256(onnx_path)

    try:
        client.get_registered_model(name)
    except MlflowException:
        client.create_registered_model(name)

    version = client.create_model_version(
        name=name,
        source=f"runs:/{run_id}/model.onnx",
        run_id=run_id,
        tags={
            "run_id": run_id,
            "onnx_sha256": digest,
            "model_meta_artifact": "model_meta.json",
            "calibration_artifact": "calibration.json",
        },
    )
    logger.info("Registered %s version %s from run %s", name, version.version, run_id)
    return version


def set_production(
    version: Any,
    name: str = MODEL_NAME,
    *,
    tracking_uri: str | None = None,
) -> None:
    """Set the MLflow 3 alias used for the production version."""
    _client(tracking_uri).set_registered_model_alias(
        name, PRODUCTION_ALIAS, str(version.version)
    )
    logger.info("Set %s@%s to version %s", name, PRODUCTION_ALIAS, version.version)


def promote_if_better(
    candidate_run_id: str,
    metric: str = "top1",
    margin: float = 0.01,
    name: str = MODEL_NAME,
    *,
    tracking_uri: str | None = None,
) -> bool:
    """Promote a registered candidate when it clears the production margin."""
    if margin < 0:
        raise ValueError("margin must be non-negative")
    client = _client(tracking_uri)
    versions = client.search_model_versions(f"name='{name}'")
    candidate = next(
        (version for version in versions if version.run_id == candidate_run_id), None
    )
    if candidate is None:
        raise ValueError(f"no registered version for candidate run {candidate_run_id}")

    candidate_value = float(client.get_run(candidate_run_id).data.metrics[metric])
    try:
        production = client.get_model_version_by_alias(name, PRODUCTION_ALIAS)
    except MlflowException:
        production = None

    if production is None:
        set_production(candidate, name, tracking_uri=tracking_uri)
        return True

    production_run = client.get_run(production.run_id)
    production_value = float(production_run.data.metrics[metric])
    if candidate_value < production_value - margin:
        logger.info(
            "Rejected run %s: %s=%s is below production %s - margin %s",
            candidate_run_id,
            metric,
            candidate_value,
            production_value,
            margin,
        )
        return False

    set_production(candidate, name, tracking_uri=tracking_uri)
    return True
