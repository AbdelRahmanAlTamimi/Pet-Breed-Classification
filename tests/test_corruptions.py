import numpy as np
import pytest
from PIL import Image

from pet_breed_classification.corruptions import (
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
