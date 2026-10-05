import json
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
from PIL import Image

from pet_breed_classification.config import Config
from pet_breed_classification.data import corruptions, split_data
from pet_breed_classification.data.corruptions import (
    CORRUPTIONS,
    DOWNSCALE_UPSCALE_SIZE,
    apply_downscale_upscale,
    apply_gaussian_blur,
)


def synthetic_images() -> list[Image.Image]:
    rgb = Image.new("RGB", (300, 200), (80, 140, 210))
    grayscale = Image.new("L", (300, 200), 120).convert("RGB")
    rgba = Image.new("RGBA", (300, 200), (180, 90, 40, 128)).convert("RGB")
    return [rgb, grayscale, rgba]


@pytest.mark.parametrize("corruption", CORRUPTIONS.values())
@pytest.mark.parametrize("severity", (1, 2, 3))
def test_corruption_output_is_rgb_and_same_size(corruption, severity) -> None:
    for image in synthetic_images():
        output = corruption(image, severity)
        assert output.mode == "RGB"
        assert output.size == image.size


@pytest.mark.parametrize("corruption", CORRUPTIONS.values())
@pytest.mark.parametrize("severity", (1, 2, 3))
def test_corruption_is_deterministic(corruption, severity) -> None:
    for image in synthetic_images():
        first = corruption(image, severity)
        second = corruption(image, severity)
        assert first.tobytes() == second.tobytes()


def test_gaussian_blur_difference_grows_with_severity() -> None:
    values = np.indices((200, 300)).sum(axis=0) % 2 * 255
    image = Image.fromarray(values.astype(np.uint8), mode="L").convert("RGB")
    source = np.asarray(image, dtype=np.float32)

    differences = [
        np.abs(np.asarray(apply_gaussian_blur(image, severity), dtype=np.float32) - source).mean()
        for severity in (1, 2, 3)
    ]

    assert differences[0] < differences[1] < differences[2]


def test_downscale_upscale_preserves_aspect_ratio(monkeypatch) -> None:
    image = Image.new("RGB", (300, 200), (80, 140, 210))
    resized_sizes = []
    original_resize = Image.Image.resize

    def capture_resize(self, size, *args, **kwargs):
        resized_sizes.append(size)
        return original_resize(self, size, *args, **kwargs)

    monkeypatch.setattr(Image.Image, "resize", capture_resize)
    output = apply_downscale_upscale(image, 1)

    assert output.size == image.size
    target = DOWNSCALE_UPSCALE_SIZE[0]
    assert resized_sizes[0] == (round(image.width * target / image.height), target)


def test_process_record_writes_all_corruptions(tiny_pet_dataset: Config) -> None:
    record = {
        "image_id": "breed_00_0",
        "path": tiny_pet_dataset.IMAGES_DIR / "breed_00_0.jpg",
        "breed": "breed_00",
        "species": "cat",
        "class_index": 0,
        "split": "test",
    }

    produced = corruptions.process_record(record)

    assert len(produced) == 18
    assert all(
        set(item)
        == {
            "image_id",
            "path",
            "breed",
            "species",
            "class_index",
            "split",
            "corruption",
            "severity",
            "width",
            "height",
        }
        for item in produced
    )
    assert len(list(tiny_pet_dataset.CORRUPTED_DIR.rglob("*.jpg"))) == 18


def test_sample_per_class_is_order_independent() -> None:
    records = [
        {"breed": "breed_00", "image_id": f"breed_00_{index}"}
        for index in range(4)
    ] + [
        {"breed": "breed_01", "image_id": f"breed_01_{index}"}
        for index in range(4)
    ]

    first = corruptions.sample_per_class(records, n=2, seed=42)
    second = corruptions.sample_per_class(list(reversed(records)), n=2, seed=42)

    assert first == second
    assert corruptions.sample_per_class(records, n=2, seed=42) == first


def test_resolve_num_workers(tiny_pet_dataset: Config, monkeypatch: pytest.MonkeyPatch) -> None:
    assert corruptions.resolve_num_workers() == 1

    monkeypatch.setattr(
        corruptions,
        "cfg",
        tiny_pet_dataset.model_copy(update={"NUM_WORKERS": None}),
    )
    monkeypatch.setattr(corruptions.os, "cpu_count", lambda: 4)
    assert corruptions.resolve_num_workers() == 3

    monkeypatch.setattr(
        corruptions,
        "cfg",
        tiny_pet_dataset.model_copy(update={"NUM_WORKERS": 0}),
    )
    assert corruptions.resolve_num_workers() == 1


def test_load_test_records_reports_missing_label(
    tiny_pet_dataset: Config,
) -> None:
    split_data.main()
    label_map = json.loads(tiny_pet_dataset.LABEL_MAP_PATH.read_text(encoding="utf-8"))
    label_map["classes"][0]["breed"] = "not_breed"
    tiny_pet_dataset.LABEL_MAP_PATH.write_text(json.dumps(label_map), encoding="utf-8")

    with pytest.raises(ValueError, match="missing from the label map"):
        corruptions.load_test_records()


def test_corruption_main_end_to_end(
    tiny_pet_dataset: Config,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    split_data.main()
    monkeypatch.setattr(corruptions, "ProcessPoolExecutor", ThreadPoolExecutor)

    corruptions.main()

    assert len(list(tiny_pet_dataset.CORRUPTED_DIR.rglob("*.jpg"))) == 37 * 6 * 3
