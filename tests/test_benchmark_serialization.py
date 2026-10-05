import numpy as np

from pet_breed_classification.benchmark_serialization import (
    _markdown_table,
    _measure,
    _percentiles,
)


class FakeModel:
    def __call__(self, values):
        return values


class FakeSession:
    def run(self, output_names, inputs):
        del output_names
        batch = next(iter(inputs.values()))
        return [np.zeros((batch.shape[0], 3), dtype=np.float32)]


def test_benchmark_helpers_and_measure() -> None:
    values = [np.zeros((3, 224, 224), dtype=np.float32) for _ in range(2)]
    torch_mean, torch_p95, onnx_mean, onnx_p95 = _measure(
        values, FakeModel(), FakeSession(), "images"
    )

    assert torch_mean >= 0
    assert torch_p95 >= torch_mean
    assert onnx_mean >= 0
    assert onnx_p95 >= onnx_mean
    assert _percentiles([1.0, 2.0]) == (1.5, 1.95)
    table = _markdown_table(
        {
            "cpu_model": "test CPU",
            "threads": 1,
            "torch_mean_ms": 1.0,
            "torch_p95_ms": 2.0,
            "onnx_mean_ms": 0.5,
            "onnx_p95_ms": 1.0,
        }
    )
    assert "ONNX Runtime" in table
