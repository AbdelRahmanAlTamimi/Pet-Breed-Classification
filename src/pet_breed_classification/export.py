"""Export the trained checkpoint to ONNX and write serving metadata."""

from __future__ import annotations

import argparse
import json
import logging
import warnings
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from PIL import Image

from . import tracking
from .artifacts import _sha256
from .config import cfg
from .logging_conf import setup_logging
from .model import load_checkpoint
from .transforms import build_eval_transform

logger = logging.getLogger(__name__)
OPSET = 17


def _utc_iso(timestamp: float) -> str:
    """Format a POSIX timestamp as UTC ISO 8601."""
    return datetime.fromtimestamp(timestamp, UTC).isoformat().replace("+00:00", "Z")


def _prove_dynamic_batch(path: Path) -> None:
    """Run ONNX Runtime with the required dynamic batch sizes."""
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    for batch_size in (1, 2, 32):
        inputs = torch.randn(batch_size, 3, 224, 224).numpy()
        output = session.run(None, {input_name: inputs})[0]
        if output.shape[0] != batch_size:
            raise RuntimeError(
                f"ONNX batch axis is not dynamic: requested {batch_size}, got {output.shape[0]}"
            )


def export(checkpoint_path: Path, onnx_path: Path, meta_path: Path) -> None:
    """Export a checkpoint, validate it, and write its serving metadata."""
    model, checkpoint = load_checkpoint(checkpoint_path, torch.device("cpu"))
    model.eval()
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    dummy_input = torch.randn(1, 3, 224, 224)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            input_names=["images"],
            output_names=["logits"],
            dynamic_axes={"images": {0: "batch"}, "logits": {0: "batch"}},
            opset_version=OPSET,
            dynamo=False,
        )

    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    _prove_dynamic_batch(onnx_path)

    exported_at = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    # The checkpoint mtime approximates when training completed.
    trained_at = _utc_iso(checkpoint_path.stat().st_mtime)
    metadata = {
        "model_version": cfg.MODEL_VERSION,
        "backbone": checkpoint["backbone"],
        "framework": "onnxruntime",
        "opset": OPSET,
        "classes": checkpoint["classes"],
        "eval_transform": checkpoint["eval_transform"],
        "temperature": checkpoint["temperature"],
        "abstain_threshold": checkpoint["abstain_threshold"],
        "onnx_sha256": _sha256(onnx_path),
        "trained_at": trained_at,
        "exported_at": exported_at,
    }
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    logger.info("Exported ONNX model to %s", onnx_path)
    logger.info("Wrote model metadata to %s", meta_path)


def validate_onnx_parity(
    checkpoint_path: Path,
    onnx_path: Path,
    records: list[dict[str, object]],
    *,
    max_images: int = 200,
    tolerance: float = 1e-4,
) -> float:
    """Compare PyTorch and ONNX CPU logits for validation images."""
    selected = records[:max_images]
    if not selected:
        raise ValueError("cannot check ONNX parity without validation images")

    model, checkpoint = load_checkpoint(checkpoint_path, torch.device("cpu"))
    transform = build_eval_transform(checkpoint["eval_transform"])
    batches: list[np.ndarray] = []
    for record in selected:
        path = cfg.PROJECT_ROOT / str(record["path"])
        with Image.open(path) as image:
            batches.append(transform(image).numpy())
    inputs = np.stack(batches).astype(np.float32, copy=False)
    with torch.inference_mode():
        pytorch_logits = model(torch.from_numpy(inputs)).numpy()

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    onnx_logits = session.run(["logits"], {input_name: inputs})[0]
    maximum_difference = float(np.max(np.abs(pytorch_logits - onnx_logits)))
    if maximum_difference > tolerance:
        raise RuntimeError(
            f"PyTorch-vs-ONNX logit parity failed: {maximum_difference:.8g} > {tolerance}"
        )
    return maximum_difference


def main() -> None:
    """Export the configured production checkpoint."""
    setup_logging()
    parser = argparse.ArgumentParser(
        description="Export the production checkpoint to ONNX."
    )
    parser.add_argument(
        "--mlflow-run-id",
        help="Attach the exported model to this existing MLflow run (needs tracking enabled).",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=cfg.ARTIFACTS_DIR,
        help="Directory containing the checkpoint and receiving ONNX artifacts.",
    )
    args = parser.parse_args()
    checkpoint_path = args.artifacts_dir / cfg.MODEL_PATH.name
    onnx_path = args.artifacts_dir / "model.onnx"
    meta_path = args.artifacts_dir / "model_meta.json"
    export(checkpoint_path, onnx_path, meta_path)
    if args.mlflow_run_id:
        tracking.attach_exported_model(
            args.mlflow_run_id, onnx_path, meta_path
        )


if __name__ == "__main__":
    main()
