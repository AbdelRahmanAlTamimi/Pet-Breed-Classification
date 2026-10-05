import pytest
import torch
from PIL import Image

from pet_breed_classification.transforms import build_eval_transform


@pytest.mark.parametrize("mode", ["RGB", "L", "CMYK", "RGBA", "P"])
@pytest.mark.parametrize("size", [(301, 197), (1, 1), (1000, 10), (10, 1000)])
def test_eval_transform_normalizes_supported_modes(
    mode: str,
    size: tuple[int, int],
) -> None:
    image = Image.new(mode, size)
    transform = build_eval_transform()

    first = transform(image)
    second = transform(image)

    assert first.shape == (3, 224, 224)
    assert first.dtype == torch.float32
    assert torch.equal(first, second)
