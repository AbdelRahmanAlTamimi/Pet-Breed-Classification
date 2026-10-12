from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from pet_breed_classification import predict as predict_module
from pet_breed_classification.config import cfg
from pet_breed_classification.predict import PetBreedPredictor


def _images() -> list[Image.Image]:
    rng = np.random.default_rng(7)
    return [
        Image.fromarray(rng.integers(0, 256, size=(40 + 8 * index, 48, 3), dtype=np.uint8))
        for index in range(4)
    ]


def test_openvino_runtime_matches_onnxruntime(
    onnx_artifacts: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("openvino")
    reference = PetBreedPredictor.load(onnx_artifacts["onnx"], onnx_artifacts["meta"])
    monkeypatch.setattr(predict_module, "cfg", cfg.model_copy(update={"RUNTIME": "openvino"}))

    candidate = PetBreedPredictor.load(onnx_artifacts["onnx"], onnx_artifacts["meta"])

    assert type(candidate.session).__name__ == "OpenVINOSession"
    images = _images()
    assert np.allclose(candidate.raw_logits(images), reference.raw_logits(images), atol=1e-4)
    assert [p.decision for p in candidate.predict_batch(images)] == [
        p.decision for p in reference.predict_batch(images)
    ]


def test_unknown_runtime_is_rejected(
    onnx_artifacts: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(predict_module, "cfg", cfg.model_copy(update={"RUNTIME": "tensorrt"}))

    with pytest.raises(ValueError, match="PETBREED_RUNTIME"):
        PetBreedPredictor.load(onnx_artifacts["onnx"], onnx_artifacts["meta"])
