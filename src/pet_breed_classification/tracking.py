"""Optional MLflow tracking for training runs, export attachment, and the existing-run backfill.

Tracking is off unless ``PETBREED_MLFLOW_ENABLED`` is true. While it is off, every
``Tracker`` method returns without touching MLflow, so callers never branch on the flag.

Backfill the existing ResNet-50 run from files on disk with:
``uv run python -m pet_breed_classification.tracking backfill``
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import tempfile
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import mlflow
import yaml
from mlflow.tracking import MlflowClient

from .artifacts import _sha256
from .config import cfg
from .logging_conf import setup_logging

logger = logging.getLogger(__name__)

FRAMEWORK = "pytorch"
UNKNOWN = "unknown"
MANIFEST_DVC_PATH = "data/processed/manifest.json"
BYTES_PER_MB = 1024 * 1024
EPOCH_METRICS = (
    "train_loss",
    "train_accuracy",
    "train_macro_f1",
    "val_loss",
    "val_accuracy",
    "val_macro_f1",
)
# Parameters the existing run did not record in any file on disk. The backfill never
# guesses them; they are listed in the backfill_missing tag instead.
UNRECORDED_PARAMS = (
    "lr",
    "batch_size",
    "split_seed",
    "train_seed",
    "amp",
    "optimizer",
    "scheduler",
)
BACKFILL_RUN_NAME = "backfill-resnet50-existing-run"
BACKFILL_NOTE = (
    "git_commit, git_dirty, data_version and requirements.txt describe the checkout "
    "at backfill time, not at training time"
)


class Tracker:
    """Logging facade for one run. Every method is a no-op when tracking is disabled."""

    def __init__(self, enabled: bool, run_id: str | None = None) -> None:
        self.enabled = enabled
        self.run_id = run_id

    def log_params(self, params: Mapping[str, Any]) -> None:
        if self.enabled and params:
            mlflow.log_params(dict(params))

    def set_tags(self, tags: Mapping[str, str]) -> None:
        if self.enabled and tags:
            mlflow.set_tags(dict(tags))

    def log_metrics(
        self, metrics: Mapping[str, float], step: int | None = None
    ) -> None:
        if self.enabled and metrics:
            mlflow.log_metrics(dict(metrics), step=step)

    def log_artifact(self, path: Path) -> None:
        if self.enabled:
            mlflow.log_artifact(str(path))

    def log_requirements(self) -> None:
        """Attach a pinned dependency snapshot from uv.lock as requirements.txt."""
        if not self.enabled:
            return
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "requirements.txt"
            subprocess.run(
                [
                    "uv",
                    "export",
                    "--frozen",
                    "--no-hashes",
                    "--no-emit-project",
                    "--all-groups",
                    "--output-file",
                    str(output),
                ],
                cwd=cfg.PROJECT_ROOT,
                stdout=subprocess.DEVNULL,  # uv echoes the whole snapshot to stdout as well
                check=True,
                timeout=120,
            )
            self.log_artifact(output)


def _is_enabled(enabled: bool | None) -> bool:
    return cfg.MLFLOW_ENABLED if enabled is None else enabled


def _git(*args: str) -> str | None:
    """Return stripped git output, or None when git or the repository is unavailable."""
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=cfg.PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip()


def _git_dirty() -> str:
    """Return 'true' when tracked files differ from HEAD. Untracked files are ignored."""
    status = _git("status", "--porcelain", "--untracked-files=no")
    if status is None:
        return UNKNOWN
    return "true" if status else "false"


def read_data_version(dvc_lock_path: Path, split_index_path: Path) -> tuple[str, str]:
    """Return (version, source). Prefers the manifest md5 recorded in dvc.lock."""
    if dvc_lock_path.is_file():
        lock = yaml.safe_load(dvc_lock_path.read_text(encoding="utf-8")) or {}
        for stage in (lock.get("stages") or {}).values():
            for output in stage.get("outs") or []:
                if output.get("path") == MANIFEST_DVC_PATH and output.get("md5"):
                    return output["md5"], f"dvc.lock:{MANIFEST_DVC_PATH}"
    if split_index_path.is_file():
        return _sha256(split_index_path), f"sha256:{split_index_path.name}"
    logger.warning("No dataset version: neither dvc.lock nor the split index exists")
    return UNKNOWN, UNKNOWN


def run_tags() -> dict[str, str]:
    """Return the tags every run carries."""
    data_version, data_version_source = read_data_version(
        cfg.DVC_LOCK_PATH, cfg.SPLIT_INDEX_PATH
    )
    return {
        "git_commit": _git("rev-parse", "HEAD") or UNKNOWN,
        "git_dirty": _git_dirty(),
        "data_version": data_version,
        "data_version_source": data_version_source,
        "framework": FRAMEWORK,
        "author": _git("config", "user.name") or UNKNOWN,
    }


@contextmanager
def tracking_run(
    run_name: str,
    *,
    tags: Mapping[str, str] | None = None,
    enabled: bool | None = None,
    tracking_uri: str | None = None,
    experiment: str | None = None,
) -> Iterator[Tracker]:
    """Start an MLflow run when tracking is enabled; otherwise yield a no-op tracker."""
    if not _is_enabled(enabled):
        yield Tracker(enabled=False)
        return

    mlflow.set_tracking_uri(tracking_uri or cfg.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(experiment or cfg.MLFLOW_EXPERIMENT)
    with mlflow.start_run(run_name=run_name) as run:
        run_id = run.info.run_id
        logger.info("MLflow run started: %s (%s)", run_id, run_name)
        tracker = Tracker(enabled=True, run_id=run_id)
        tracker.set_tags({"backfilled": "false", **run_tags(), **(tags or {})})
        yield tracker


def model_size_mb(onnx_path: Path) -> float:
    """Return the ONNX file size in MiB, read from disk."""
    return onnx_path.stat().st_size / BYTES_PER_MB


def log_history(tracker: Tracker, history: Sequence[Mapping[str, Any]]) -> list[str]:
    """Log per-epoch metrics (step = epoch) and the final/best validation summary.

    A value absent from the history is not logged; its name is returned instead.
    """
    if not history:
        raise ValueError("Training history is empty")

    missing: set[str] = set()
    for row in history:
        present = {name: float(row[name]) for name in EPOCH_METRICS if name in row}
        missing.update(name for name in EPOCH_METRICS if name not in row)
        tracker.log_metrics(present, step=int(row["epoch"]))

    last = history[-1]
    summary: dict[str, float] = {}
    for name in ("val_accuracy", "val_macro_f1"):
        if name in last:
            summary[f"final_{name}"] = float(last[name])
        else:
            missing.add(name)

    scored = [row for row in history if "val_accuracy" in row]
    if scored:
        # max() returns the first maximum, matching train.py's strict ">" checkpoint rule.
        best = max(scored, key=lambda row: float(row["val_accuracy"]))
        summary["best_val_accuracy"] = float(best["val_accuracy"])
        summary["best_epoch"] = float(best["epoch"])
    tracker.log_metrics(summary)
    return sorted(missing)


def attach_exported_model(
    run_id: str,
    onnx_path: Path,
    meta_path: Path,
    *,
    enabled: bool | None = None,
    tracking_uri: str | None = None,
) -> None:
    """Attach the freshly exported ONNX model, its metadata, and its size to a run."""
    if not _is_enabled(enabled):
        logger.info("MLflow tracking is disabled; export not attached to %s", run_id)
        return
    client = MlflowClient(tracking_uri=tracking_uri or cfg.MLFLOW_TRACKING_URI)
    client.log_artifact(run_id, str(onnx_path))
    client.log_artifact(run_id, str(meta_path))
    client.log_metric(run_id, "model_size_mb", model_size_mb(onnx_path))


def _read_json(path: Path) -> Any | None:
    """Return parsed JSON, or None when the file does not exist."""
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def backfill() -> str:
    """Log the existing ResNet-50 run from files on disk and return its MLflow run id.

    Always logs, regardless of PETBREED_MLFLOW_ENABLED, because it is an explicit command.
    Only values recorded in the history or metadata files are logged; every other value
    is left out and listed in the ``backfill_missing`` tag.
    """
    required = (cfg.HISTORY_PATH, cfg.LABEL_MAP_PATH, cfg.SPLIT_INDEX_PATH)
    absent = [str(path) for path in required if not path.is_file()]
    if absent:
        raise FileNotFoundError(f"Cannot backfill, missing files: {', '.join(absent)}")

    history = _read_json(cfg.HISTORY_PATH)
    if not isinstance(history, list) or not history:
        raise ValueError(f"Training history is empty or not a list: {cfg.HISTORY_PATH}")
    meta = _read_json(cfg.MODEL_META_PATH) or {}

    params: dict[str, Any] = {"epochs": len(history)}
    missing: list[str] = list(UNRECORDED_PARAMS)
    if "backbone" in meta:
        params["backbone"] = meta["backbone"]
    else:
        missing.append("backbone")

    # The ONNX file is only trusted when its hash matches the one recorded at export.
    onnx_verified = (
        cfg.ONNX_PATH.is_file()
        and "onnx_sha256" in meta
        and _sha256(cfg.ONNX_PATH) == meta["onnx_sha256"]
    )
    missing.append("train_duration_seconds")
    if not onnx_verified:
        missing.append("model_size_mb")

    artifacts = [cfg.LABEL_MAP_PATH, cfg.SPLIT_INDEX_PATH, cfg.HISTORY_PATH]
    if cfg.MODEL_META_PATH.is_file():
        artifacts.append(cfg.MODEL_META_PATH)
    if onnx_verified:
        artifacts.append(cfg.ONNX_PATH)

    with tracking_run(
        BACKFILL_RUN_NAME,
        enabled=True,
        tags={"backfilled": "true", "backfill_note": BACKFILL_NOTE},
    ) as tracker:
        tracker.log_params(params)
        missing.extend(log_history(tracker, history))
        if onnx_verified:
            tracker.log_metrics({"model_size_mb": model_size_mb(cfg.ONNX_PATH)})
        tracker.set_tags({"backfill_missing": ",".join(sorted(set(missing))) or "none"})
        for path in artifacts:
            tracker.log_artifact(path)
        tracker.log_requirements()

    if tracker.run_id is None:
        raise RuntimeError("Backfill did not start an MLflow run")
    return tracker.run_id


def main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser(description="MLflow tracking utilities.")
    parser.add_argument("command", choices=["backfill"])
    args = parser.parse_args()
    if args.command == "backfill":
        run_id = backfill()
        logger.info("Backfilled existing run as MLflow run %s", run_id)


if __name__ == "__main__":
    main()
