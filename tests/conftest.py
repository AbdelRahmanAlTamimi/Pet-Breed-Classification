from __future__ import annotations

from collections.abc import Callable, Iterator
from io import BytesIO
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

import pet_breed_classification.data as data_module
from pet_breed_classification.config import Config, cfg
from pet_breed_classification.data import build_manifest, corruptions, split_data
from pet_breed_classification.export import export
from pet_breed_classification.model import PetBreedClassifier, save_checkpoint
from pet_breed_classification.predict import PetBreedPredictor

# Test-only abstention threshold for the synthetic random model. It is not a calibrated value.
TEST_ABSTAIN_THRESHOLD = 0.5


def _classes() -> list[dict[str, object]]:
    return [
        {"index": index, "breed": f"breed_{index}", "species": "dog"}
        for index in range(3)
    ]


@pytest.fixture
def sample_image() -> Image.Image:
    """Return a small non-square RGB image."""
    return Image.new("RGB", (100, 80), (20, 80, 140))


@pytest.fixture
def image_bytes() -> Callable[[Image.Image, str], bytes]:
    """Return a helper that encodes a PIL image for multipart requests."""

    def encode(image: Image.Image, image_format: str = "PNG") -> bytes:
        buffer = BytesIO()
        image.save(buffer, format=image_format)
        return buffer.getvalue()

    return encode


@pytest.fixture
def tiny_pet_dataset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Config:
    """Build a tiny Oxford-Pet-shaped dataset and patch data-module settings."""
    data_root = tmp_path / "data"
    images_dir = data_root / "raw" / "images"
    annotations_dir = data_root / "raw" / "annotations"
    processed_dir = data_root / "processed"
    images_dir.mkdir(parents=True)
    annotations_dir.mkdir(parents=True)
    processed_dir.mkdir(parents=True)

    trainval_lines: list[str] = []
    test_lines: list[str] = []
    for breed_index in range(37):
        breed = f"breed_{breed_index:02d}"
        class_id = breed_index + 1
        species_id = 1 if breed_index < 12 else 2
        for image_index in range(7):
            image_id = f"{breed}_{image_index}"
            image = Image.new(
                "RGB",
                (48, 40),
                (
                    (breed_index * 7 + image_index) % 256,
                    (breed_index * 11 + image_index * 3) % 256,
                    (breed_index * 13 + image_index * 5) % 256,
                ),
            )
            image.save(images_dir / f"{image_id}.jpg", format="JPEG", quality=90)
            line = f"{image_id} {class_id} {species_id} {class_id}"
            if image_index < 5:
                trainval_lines.append(line)
            else:
                test_lines.append(line)

    (annotations_dir / "trainval.txt").write_text(
        "\n".join(trainval_lines) + "\n", encoding="utf-8"
    )
    (annotations_dir / "test.txt").write_text(
        "\n".join(test_lines) + "\n", encoding="utf-8"
    )

    tiny_cfg = cfg.model_copy(
        update={
            "PROJECT_ROOT": tmp_path,
            "IMAGES_DIR": images_dir,
            "ANNOTATIONS_DIR": annotations_dir,
            "LABEL_MAP_PATH": data_root / "label_map.json",
            "SPLIT_INDEX_PATH": data_root / "split_index.json",
            "CORRUPTED_DIR": data_root / "corrupted",
            "MANIFEST_PATH": processed_dir / "manifest.json",
            "CORRUPTION_SAMPLE_PER_CLASS": 1,
            "NUM_WORKERS": 1,
        }
    )
    for module in (data_module, split_data, build_manifest, corruptions):
        monkeypatch.setattr(module, "cfg", tiny_cfg)

    for module in (split_data, build_manifest, corruptions):
        monkeypatch.setattr(module, "setup_logging", lambda: None)

    return tiny_cfg


@pytest.fixture(scope="session")
def onnx_artifacts(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    """Create one session-scoped random checkpoint and ONNX export."""
    directory = tmp_path_factory.mktemp("onnx-artifacts")
    checkpoint_path = directory / "model.pt"
    onnx_path = directory / "model.onnx"
    meta_path = directory / "model_meta.json"
    model = PetBreedClassifier(num_classes=3, pretrained=False)
    save_checkpoint(
        model,
        checkpoint_path,
        _classes(),
        42,
        1,
        {},
        temperature=1.0,
        abstain_threshold=TEST_ABSTAIN_THRESHOLD,
    )
    export(checkpoint_path, onnx_path, meta_path)
    return {"checkpoint": checkpoint_path, "onnx": onnx_path, "meta": meta_path}


@pytest.fixture
def predictor(onnx_artifacts: dict[str, Path]) -> PetBreedPredictor:
    """Load the session-scoped ONNX artifacts."""
    return PetBreedPredictor.load(onnx_artifacts["onnx"], onnx_artifacts["meta"])


@pytest.fixture
def client(predictor: PetBreedPredictor, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    """Create a TestClient whose lifespan loads the fixture predictor."""
    from pet_breed_classification.api import main as api_main

    monkeypatch.setattr(
        api_main.PetBreedPredictor,
        "load",
        classmethod(lambda cls, *args, **kwargs: predictor),
    )
    with TestClient(api_main.app) as test_client:
        yield test_client
