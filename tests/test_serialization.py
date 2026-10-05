import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pytest
import torch
from PIL import Image

from pet_breed_classification.config import cfg
from pet_breed_classification.data import load_records
from pet_breed_classification.export import export
from pet_breed_classification.model import (
    PetBreedClassifier,
    load_checkpoint,
    save_checkpoint,
)
from pet_breed_classification.transforms import build_eval_transform


def classes() -> list[dict[str, object]]:
    return [
        {"index": index, "breed": f"breed_{index}", "species": "dog"}
        for index in range(3)
    ]


def test_random_checkpoint_pytorch_onnx_parity(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "model.pt"
    onnx_path = tmp_path / "model.onnx"
    meta_path = tmp_path / "model_meta.json"
    model = PetBreedClassifier(num_classes=3, pretrained=False)
    save_checkpoint(model, checkpoint_path, classes(), 42, 1, {})
    export(checkpoint_path, onnx_path, meta_path)
    loaded_model, _ = load_checkpoint(checkpoint_path, torch.device("cpu"))

    inputs = np.random.default_rng(42).standard_normal((2, 3, 224, 224)).astype(
        np.float32
    )
    with torch.inference_mode():
        torch_logits = loaded_model(torch.from_numpy(inputs)).numpy()
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    onnx_logits = session.run(["logits"], {"images": inputs})[0]

    assert np.allclose(torch_logits, onnx_logits, atol=1e-4)
    assert np.array_equal(torch_logits.argmax(axis=1), onnx_logits.argmax(axis=1))


@pytest.mark.skipif(
    not cfg.MODEL_PATH.is_file()
    or not cfg.ONNX_PATH.is_file()
    or not cfg.MODEL_META_PATH.is_file()
    or not cfg.MANIFEST_PATH.is_file(),
    reason="real checkpoint, ONNX artifacts, or validation manifest unavailable",
)
def test_real_validation_parity() -> None:
    records = load_records("val")[:200]
    transform = build_eval_transform()
    batches = []
    for record in records:
        with Image.open(cfg.PROJECT_ROOT / str(record["path"])) as image:
            batches.append(transform(image).numpy())
    inputs = np.stack(batches).astype(np.float32, copy=False)

    model, _ = load_checkpoint(cfg.MODEL_PATH, torch.device("cpu"))
    with torch.inference_mode():
        torch_logits = model(torch.from_numpy(inputs)).numpy()
    session = ort.InferenceSession(str(cfg.ONNX_PATH), providers=["CPUExecutionProvider"])
    onnx_logits = session.run(["logits"], {"images": inputs})[0]

    metadata = json.loads(cfg.MODEL_META_PATH.read_text(encoding="utf-8"))
    assert metadata["framework"] == "onnxruntime"
    assert np.allclose(torch_logits, onnx_logits, atol=1e-4)
    assert np.array_equal(torch_logits.argmax(axis=1), onnx_logits.argmax(axis=1))
