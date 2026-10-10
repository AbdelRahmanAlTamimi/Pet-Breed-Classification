"""Helpers for model artifact files and isolated training runs."""

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

RUN_NAME_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\Z")


@dataclass(frozen=True)
class RunArtifacts:
    """Paths belonging to one immutable training run directory."""

    directory: Path
    checkpoint: Path
    onnx: Path
    metadata: Path
    history: Path
    reports: Path


def create_run_artifacts(run_name: str, artifacts_dir: Path) -> RunArtifacts:
    """Create a fresh ``artifacts/runs/<run_name>`` layout without reusing it."""
    if not RUN_NAME_PATTERN.fullmatch(run_name):
        raise ValueError(
            "run_name must contain only letters, numbers, '.', '_' or '-' and not be empty"
        )
    directory = artifacts_dir / "runs" / run_name
    directory.mkdir(parents=True, exist_ok=False)
    reports = directory / "reports"
    reports.mkdir()
    return RunArtifacts(
        directory=directory,
        checkpoint=directory / "resnet50_best.pt",
        onnx=directory / "model.onnx",
        metadata=directory / "model_meta.json",
        history=directory / "resnet50_training_history.json",
        reports=reports,
    )


def _sha256(path: Path) -> str:
    """Return a file's SHA-256 digest."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
