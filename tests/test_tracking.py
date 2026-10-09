"""Hermetic tests for optional MLflow tracking: a local file store under tmp_path only."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any

import pytest
from mlflow.tracking import MlflowClient

from pet_breed_classification import tracking
from pet_breed_classification.config import Config, cfg
from pet_breed_classification.tracking import tracking_run

HISTORY: list[dict[str, Any]] = [
    {
        "epoch": 1,
        "learning_rate": 0.1,
        "train_loss": 2.0,
        "train_accuracy": 0.30,
        "train_macro_f1": 0.28,
        "val_loss": 1.5,
        "val_accuracy": 0.50,
        "val_macro_f1": 0.40,
        "elapsed_seconds": 10.0,
    },
    {
        "epoch": 2,
        "learning_rate": 0.05,
        "train_loss": 1.0,
        "train_accuracy": 0.60,
        "train_macro_f1": 0.58,
        "val_loss": 0.6,
        "val_accuracy": 0.90,
        "val_macro_f1": 0.88,
        "elapsed_seconds": 10.0,
    },
    {
        "epoch": 3,
        "learning_rate": 0.0,
        "train_loss": 0.5,
        "train_accuracy": 0.80,
        "train_macro_f1": 0.79,
        "val_loss": 0.6,
        "val_accuracy": 0.90,
        "val_macro_f1": 0.87,
        "elapsed_seconds": 10.0,
    },
]
DVC_LOCK = textwrap.dedent(
    """\
    schema: '2.0'
    stages:
      build_manifest:
        cmd: uv run python -m pet_breed_classification.data.build_manifest
        outs:
        - path: data/processed/manifest.json
          hash: md5
          md5: 0123456789abcdef0123456789abcdef
    """
)
MIB = 1024 * 1024


@pytest.fixture(autouse=True)
def offline_mlflow(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep MLflow offline: no telemetry, no hints, and the file store opted in for tests.

    MLflow 3.17 refuses the file:// store unless MLFLOW_ALLOW_FILE_STORE is set, because the
    file backend is in maintenance mode. Production uses PostgreSQL, never this store.
    """
    monkeypatch.setenv("MLFLOW_DISABLE_TELEMETRY", "true")
    monkeypatch.setenv("MLFLOW_DISABLE_AGENT_HINT", "1")
    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")


@pytest.fixture
def tracking_cfg(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Config:
    """Point tracking at tmp_path files and a file:// store; keep the real repo root for git."""
    data_root = tmp_path / "data"
    artifacts = tmp_path / "artifacts"
    data_root.mkdir()
    artifacts.mkdir()
    (data_root / "label_map.json").write_text('{"classes": []}\n', encoding="utf-8")
    (data_root / "split_index.json").write_text('{"seed": 42}\n', encoding="utf-8")
    (artifacts / "resnet50_training_history.json").write_text(
        json.dumps(HISTORY), encoding="utf-8"
    )
    (tmp_path / "dvc.lock").write_text(DVC_LOCK, encoding="utf-8")

    test_cfg = cfg.model_copy(
        update={
            "LABEL_MAP_PATH": data_root / "label_map.json",
            "SPLIT_INDEX_PATH": data_root / "split_index.json",
            "HISTORY_PATH": artifacts / "resnet50_training_history.json",
            "ONNX_PATH": artifacts / "model.onnx",
            "MODEL_META_PATH": artifacts / "model_meta.json",
            "DVC_LOCK_PATH": tmp_path / "dvc.lock",
            "MLFLOW_ENABLED": True,
            "MLFLOW_TRACKING_URI": (tmp_path / "mlruns").as_uri(),
            "MLFLOW_EXPERIMENT": "tracking-tests",
        }
    )
    monkeypatch.setattr(tracking, "cfg", test_cfg)
    return test_cfg


def _client(test_cfg: Config) -> MlflowClient:
    return MlflowClient(tracking_uri=test_cfg.MLFLOW_TRACKING_URI)


def _artifact_names(client: MlflowClient, run_id: str) -> set[str]:
    return {info.path for info in client.list_artifacts(run_id)}


def _git_output(*args: str) -> str | None:
    """Return git output for the repository root, or None outside a git checkout."""
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=cfg.PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def test_disabled_tracking_is_a_noop(
    tracking_cfg: Config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        tracking, "cfg", tracking_cfg.model_copy(update={"MLFLOW_ENABLED": False})
    )

    def must_not_be_called(*args: object, **kwargs: object) -> None:
        raise AssertionError(
            "MLflow or a subprocess was called while tracking is disabled"
        )

    monkeypatch.setattr(tracking.mlflow, "start_run", must_not_be_called)
    monkeypatch.setattr(tracking.subprocess, "run", must_not_be_called)
    artifact = tmp_path / "artifact.txt"
    artifact.write_text("payload", encoding="utf-8")

    with tracking_run("disabled") as tracker:
        tracker.log_params({"lr": 0.1})
        tracker.log_metrics({"val_accuracy": 0.5}, step=1)
        tracker.set_tags({"key": "value"})
        tracker.log_artifact(artifact)
        tracker.log_requirements()

    assert tracker.enabled is False
    assert tracker.run_id is None
    assert not (tmp_path / "mlruns").exists()


def test_enabled_run_logs_params_metrics_tags_and_artifacts(
    tracking_cfg: Config, tmp_path: Path
) -> None:
    artifact = tmp_path / "note.txt"
    artifact.write_text("hello", encoding="utf-8")

    with tracking_run("unit-run") as tracker:
        tracker.log_params({"lr": 0.0001, "amp": False})
        tracker.log_metrics({"val_accuracy": 0.5}, step=1)
        tracker.log_metrics({"val_accuracy": 0.75}, step=2)
        tracker.log_artifact(artifact)
        tracker.log_requirements()
        run_id = tracker.run_id

    assert run_id is not None
    client = _client(tracking_cfg)
    run = client.get_run(run_id)
    assert run.info.status == "FINISHED"
    assert run.data.params == {"lr": "0.0001", "amp": "False"}
    history = client.get_metric_history(run_id, "val_accuracy")
    assert [(metric.step, metric.value) for metric in history] == [(1, 0.5), (2, 0.75)]
    assert run.data.tags["data_version"] == "0123456789abcdef0123456789abcdef"
    assert (
        run.data.tags["data_version_source"] == "dvc.lock:data/processed/manifest.json"
    )
    assert run.data.tags["framework"] == "pytorch"
    assert run.data.tags["backfilled"] == "false"
    assert run.data.tags["author"]
    assert _artifact_names(client, run_id) == {"note.txt", "requirements.txt"}

    download_dir = tmp_path / "download"
    requirements = Path(
        client.download_artifacts(run_id, "requirements.txt", str(download_dir))
    )
    contents = requirements.read_text(encoding="utf-8")
    assert "mlflow-skinny==" in contents
    assert "--hash" not in contents


def test_git_tags_match_the_repository(tracking_cfg: Config) -> None:
    with tracking_run("git-tags") as tracker:
        run_id = tracker.run_id

    assert run_id is not None
    tags = _client(tracking_cfg).get_run(run_id).data.tags
    head = _git_output("rev-parse", "HEAD")
    if head is None:
        # Not a git checkout (for example an exported tarball): tags must say so.
        assert tags["git_commit"] == "unknown"
        assert tags["git_dirty"] == "unknown"
        return
    assert tags["git_commit"] == head
    dirty = _git_output("status", "--porcelain", "--untracked-files=no")
    assert tags["git_dirty"] == ("true" if dirty else "false")
    assert tags["author"]


def test_git_tags_fall_back_to_unknown_without_git(
    tracking_cfg: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    def git_missing(*args: object, **kwargs: object) -> None:
        raise FileNotFoundError("git")

    monkeypatch.setattr(tracking.subprocess, "run", git_missing)

    tags = tracking.run_tags()

    assert tags["git_commit"] == "unknown"
    assert tags["git_dirty"] == "unknown"
    assert tags["author"] == "unknown"
    assert tags["data_version"] == "0123456789abcdef0123456789abcdef"


def test_data_version_prefers_manifest_md5_from_dvc_lock(tmp_path: Path) -> None:
    lock = tmp_path / "dvc.lock"
    lock.write_text(DVC_LOCK, encoding="utf-8")

    version, source = tracking.read_data_version(lock, tmp_path / "split_index.json")

    assert version == "0123456789abcdef0123456789abcdef"
    assert source == "dvc.lock:data/processed/manifest.json"


def test_data_version_falls_back_to_split_index_sha256(tmp_path: Path) -> None:
    split_index = tmp_path / "split_index.json"
    split_index.write_bytes(b'{"seed": 42}\n')

    version, source = tracking.read_data_version(tmp_path / "missing.lock", split_index)

    assert version == hashlib.sha256(b'{"seed": 42}\n').hexdigest()
    assert source == "sha256:split_index.json"


def test_data_version_is_unknown_when_nothing_exists(tmp_path: Path) -> None:
    version, source = tracking.read_data_version(
        tmp_path / "missing.lock", tmp_path / "missing.json"
    )

    assert (version, source) == ("unknown", "unknown")


def test_model_size_mb_is_file_size_in_mib(tmp_path: Path) -> None:
    model = tmp_path / "model.onnx"
    model.write_bytes(b"\0" * (2 * MIB))

    assert tracking.model_size_mb(model) == 2.0


def test_log_history_logs_epoch_series_and_summary(tracking_cfg: Config) -> None:
    with tracking_run("history-run") as tracker:
        unlogged = tracking.log_history(tracker, HISTORY)
        run_id = tracker.run_id

    assert unlogged == []
    assert run_id is not None
    client = _client(tracking_cfg)
    steps = [
        metric.step for metric in client.get_metric_history(run_id, "val_accuracy")
    ]
    assert steps == [1, 2, 3]
    metrics = client.get_run(run_id).data.metrics
    assert metrics["final_val_accuracy"] == 0.90
    assert metrics["final_val_macro_f1"] == 0.87
    assert metrics["best_val_accuracy"] == 0.90
    assert metrics["best_epoch"] == 2.0
    assert not any(name.startswith("learning_rate") for name in metrics)


def test_log_history_reports_values_it_cannot_log(tracking_cfg: Config) -> None:
    rows = [dict(row) for row in HISTORY]
    del rows[0]["val_macro_f1"]

    with tracking_run("gap-run") as tracker:
        unlogged = tracking.log_history(tracker, rows)
        run_id = tracker.run_id

    assert unlogged == ["val_macro_f1"]
    assert run_id is not None
    client = _client(tracking_cfg)
    steps = [
        metric.step for metric in client.get_metric_history(run_id, "val_macro_f1")
    ]
    assert steps == [2, 3]


def test_log_history_rejects_empty_history(tracking_cfg: Config) -> None:
    with tracking_run("empty-run") as tracker, pytest.raises(ValueError, match="empty"):
        tracking.log_history(tracker, [])


def test_backfill_logs_only_values_recorded_on_disk(
    tracking_cfg: Config, onnx_artifacts: dict[str, Path]
) -> None:
    shutil.copy(onnx_artifacts["onnx"], tracking_cfg.ONNX_PATH)
    shutil.copy(onnx_artifacts["meta"], tracking_cfg.MODEL_META_PATH)

    run_id = tracking.backfill()

    client = _client(tracking_cfg)
    run = client.get_run(run_id)
    assert run.info.run_name == tracking.BACKFILL_RUN_NAME
    assert run.data.params == {"backbone": "resnet50", "epochs": "3"}
    assert run.data.metrics["final_val_accuracy"] == HISTORY[-1]["val_accuracy"]
    assert run.data.metrics["final_val_macro_f1"] == HISTORY[-1]["val_macro_f1"]
    assert run.data.metrics["model_size_mb"] == pytest.approx(
        tracking_cfg.ONNX_PATH.stat().st_size / MIB
    )
    assert "learning_rate" not in run.data.metrics
    assert "train_duration_seconds" not in run.data.metrics
    assert "temperature" not in run.data.metrics
    assert "temperature" not in run.data.params
    assert run.data.tags["backfilled"] == "true"
    assert set(run.data.tags["backfill_missing"].split(",")) == {
        "lr",
        "batch_size",
        "split_seed",
        "train_seed",
        "amp",
        "optimizer",
        "scheduler",
        "train_duration_seconds",
    }
    assert _artifact_names(client, run_id) == {
        "label_map.json",
        "split_index.json",
        "resnet50_training_history.json",
        "model_meta.json",
        "model.onnx",
        "requirements.txt",
    }


def test_backfill_refuses_to_invent_missing_values(
    tracking_cfg: Config, onnx_artifacts: dict[str, Path]
) -> None:
    rows = [
        {key: value for key, value in row.items() if key != "val_macro_f1"}
        for row in HISTORY
    ]
    tracking_cfg.HISTORY_PATH.write_text(json.dumps(rows), encoding="utf-8")
    shutil.copy(onnx_artifacts["onnx"], tracking_cfg.ONNX_PATH)
    meta = json.loads(onnx_artifacts["meta"].read_text(encoding="utf-8"))
    del meta["backbone"]
    meta["onnx_sha256"] = "0" * 64
    tracking_cfg.MODEL_META_PATH.write_text(json.dumps(meta), encoding="utf-8")

    run_id = tracking.backfill()

    client = _client(tracking_cfg)
    run = client.get_run(run_id)
    assert "backbone" not in run.data.params
    assert "val_macro_f1" not in run.data.metrics
    assert "final_val_macro_f1" not in run.data.metrics
    assert "model_size_mb" not in run.data.metrics
    assert "model.onnx" not in _artifact_names(client, run_id)
    assert {"backbone", "val_macro_f1", "model_size_mb"} <= set(
        run.data.tags["backfill_missing"].split(",")
    )


def test_backfill_requires_history_and_creates_no_run(tracking_cfg: Config) -> None:
    tracking_cfg.HISTORY_PATH.unlink()

    with pytest.raises(FileNotFoundError, match="resnet50_training_history.json"):
        tracking.backfill()

    assert (
        _client(tracking_cfg).get_experiment_by_name(tracking_cfg.MLFLOW_EXPERIMENT)
        is None
    )


def test_attach_exported_model_adds_onnx_meta_and_size(
    tracking_cfg: Config, onnx_artifacts: dict[str, Path]
) -> None:
    with tracking_run("train-then-export") as tracker:
        run_id = tracker.run_id
    assert run_id is not None

    tracking.attach_exported_model(
        run_id, onnx_artifacts["onnx"], onnx_artifacts["meta"]
    )

    client = _client(tracking_cfg)
    run = client.get_run(run_id)
    assert {"model.onnx", "model_meta.json"} <= _artifact_names(client, run_id)
    assert run.data.metrics["model_size_mb"] == pytest.approx(
        onnx_artifacts["onnx"].stat().st_size / MIB
    )


def test_attach_is_a_noop_when_disabled(
    tracking_cfg: Config,
    onnx_artifacts: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        tracking, "cfg", tracking_cfg.model_copy(update={"MLFLOW_ENABLED": False})
    )

    tracking.attach_exported_model(
        "run-that-does-not-exist", onnx_artifacts["onnx"], onnx_artifacts["meta"]
    )


def test_cli_backfill_calls_backfill(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr(tracking, "setup_logging", lambda: None)
    monkeypatch.setattr(tracking, "backfill", lambda: calls.append("called") or "run-1")
    monkeypatch.setattr(sys, "argv", ["tracking", "backfill"])

    tracking.main()

    assert calls == ["called"]
