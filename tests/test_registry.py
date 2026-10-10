from __future__ import annotations

import tempfile
from pathlib import Path

import mlflow
import pytest
from mlflow.tracking import MlflowClient

from pet_breed_classification import registry


@pytest.fixture
def registry_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")
    monkeypatch.setenv("MLFLOW_DISABLE_TELEMETRY", "true")
    mlflow.set_tracking_uri((tmp_path / "mlruns").as_uri())
    return (tmp_path / "mlruns").as_uri()


def _run(registry_store: str, name: str, top1: float) -> str:
    with mlflow.start_run(run_name=name) as run:
        mlflow.log_metric("top1", top1)
        with tempfile.TemporaryDirectory() as directory:
            for artifact in ("model.onnx", "model_meta.json", "calibration.json"):
                path = Path(directory) / artifact
                path.write_text(artifact, encoding="utf-8")
                mlflow.log_artifact(path)
        return run.info.run_id


def test_register_and_promote_alias(registry_store: str) -> None:
    run_id = _run(registry_store, "candidate", 0.9)

    version = registry.register_run(run_id, tracking_uri=registry_store)
    assert version.tags["run_id"] == run_id
    assert version.tags["model_meta_artifact"] == "model_meta.json"

    assert registry.promote_if_better(run_id, tracking_uri=registry_store)
    production = MlflowClient(registry_store).get_model_version_by_alias(
        registry.MODEL_NAME, registry.PRODUCTION_ALIAS
    )
    assert production.run_id == run_id


def test_promote_if_better_rejects_below_margin(registry_store: str) -> None:
    first = _run(registry_store, "first", 0.9)
    second = _run(registry_store, "second", 0.7)
    first_version = registry.register_run(first, tracking_uri=registry_store)
    registry.register_run(second, tracking_uri=registry_store)
    registry.set_production(first_version, tracking_uri=registry_store)

    assert not registry.promote_if_better(
        second, margin=0.01, tracking_uri=registry_store
    )


def test_register_requires_companion_artifacts(registry_store: str) -> None:
    with mlflow.start_run(run_name="incomplete") as run:
        run_id = run.info.run_id
        path = Path("incomplete.onnx")
        path.write_bytes(b"onnx")
        mlflow.log_artifact(path, "model.onnx")
        path.unlink()

    with pytest.raises(FileNotFoundError, match="missing artifacts"):
        registry.register_run(run_id, tracking_uri=registry_store)
