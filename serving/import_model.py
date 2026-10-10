"""Import the ONNX serving artifacts into the BentoML model store."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import bentoml

from pet_breed_classification.artifacts import _sha256
from pet_breed_classification.config import cfg

MODEL_NAME = "pet_breed_classifier"


def _calibration_path(artifacts_dir: Path, requested: Path | None) -> Path | None:
    """Find the optional calibration report without inventing one."""
    if requested is not None:
        return requested if requested.is_file() else None
    for candidate in (
        artifacts_dir / "calibration.json",
        artifacts_dir / "reports" / "calibration.json",
        artifacts_dir.parent / "reports" / "calibration.json",
    ):
        if candidate.is_file():
            return candidate
    return None


def import_model(
    *,
    onnx_path: Path = cfg.ONNX_PATH,
    metadata_path: Path = cfg.MODEL_META_PATH,
    calibration_path: Path | None = None,
) -> bentoml.Model:
    """Create one model version, or reuse the latest matching version."""
    if not onnx_path.is_file():
        raise FileNotFoundError(onnx_path)
    if not metadata_path.is_file():
        raise FileNotFoundError(metadata_path)

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    onnx_sha = _sha256(onnx_path)
    if metadata.get("onnx_sha256", "").lower() != onnx_sha:
        raise ValueError("model_meta.json onnx_sha256 does not match model.onnx")
    calibration = _calibration_path(onnx_path.parent, calibration_path)
    labels = {
        "onnx_sha256": onnx_sha,
        "model_version": str(metadata["model_version"]),
        "temperature": str(metadata["temperature"]),
        "abstain_threshold": str(metadata["abstain_threshold"]),
    }

    try:
        latest = bentoml.models.get(f"{MODEL_NAME}:latest")
    except bentoml.exceptions.NotFound:
        latest = None
    has_calibration = latest is not None and (
        calibration is None or Path(latest.path).joinpath("calibration.json").is_file()
    )
    if latest is not None and latest.info.labels == labels and has_calibration:
        return latest

    with bentoml.models.create(
        MODEL_NAME,
        labels=labels,
        metadata={
            "model_version": metadata["model_version"],
            "temperature": metadata["temperature"],
            "abstain_threshold": metadata["abstain_threshold"],
            "onnx_sha256": onnx_sha,
        },
        module="pet_breed_classification.predict",
    ) as model:
        model_dir = Path(model.path)
        shutil.copy2(onnx_path, model_dir / "model.onnx")
        shutil.copy2(metadata_path, model_dir / "model_meta.json")
        if calibration is not None:
            shutil.copy2(calibration, model_dir / "calibration.json")
    return bentoml.models.get(str(model.tag))


def main() -> None:
    """Parse paths and import the model."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", type=Path, default=cfg.ONNX_PATH)
    parser.add_argument("--metadata", type=Path, default=cfg.MODEL_META_PATH)
    parser.add_argument("--calibration", type=Path)
    args = parser.parse_args()
    model = import_model(
        onnx_path=args.onnx,
        metadata_path=args.metadata,
        calibration_path=args.calibration,
    )
    print(f"Imported {model.tag} from {args.onnx}")
    print(f"labels: {model.info.labels}")


if __name__ == "__main__":
    main()
