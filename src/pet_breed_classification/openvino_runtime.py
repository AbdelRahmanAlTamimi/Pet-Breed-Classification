"""OpenVINO CPU runtime behind the ONNX Runtime session interface the predictor uses.

The predictor keeps one inference path: the same eval transform, the same temperature and
the same abstention threshold. Only the engine that turns the input tensor into logits
changes. The verified ONNX file is converted to an OpenVINO IR model in memory and
compiled for the CPU in FP32.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class _Input:
    name: str


class OpenVINOSession:
    """Expose ``get_inputs`` and ``run`` like ``onnxruntime.InferenceSession``."""

    def __init__(self, onnx_path: Path, threads: int | None = None) -> None:
        import openvino as ov

        config: dict[str, Any] = {"INFERENCE_PRECISION_HINT": "f32"}
        if threads is not None:
            config["INFERENCE_NUM_THREADS"] = threads
        model = ov.convert_model(str(onnx_path))
        self.compiled = ov.Core().compile_model(model, "CPU", config)
        self._inputs = [_Input(self.compiled.input(0).get_any_name())]

    def get_inputs(self) -> list[_Input]:
        return self._inputs

    def run(self, output_names: list[str], feeds: dict[str, np.ndarray]) -> list[np.ndarray]:
        results = self.compiled(feeds)
        return [np.asarray(results[self.compiled.output(name)]) for name in output_names]


def save_ir(onnx_path: Path, output_path: Path) -> Path:
    """Convert the ONNX model to OpenVINO IR (.xml + .bin) without FP16 compression."""
    import openvino as ov

    output_path.parent.mkdir(parents=True, exist_ok=True)
    ov.save_model(ov.convert_model(str(onnx_path)), str(output_path), compress_to_fp16=False)
    return output_path
