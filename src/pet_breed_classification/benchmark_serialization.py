"""Benchmark CPU PyTorch eager and ONNX Runtime inference."""

from __future__ import annotations

import logging
import platform
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from PIL import Image

from .config import cfg
from .data import load_records
from .logging_conf import setup_logging
from .model import load_checkpoint
from .predict import PetBreedPredictor

logger = logging.getLogger(__name__)
SAMPLE_SIZE = 200
WARMUP_RUNS = 20
THREADS = 1


def _cpu_model_name() -> str:
    """Return the Linux CPU model name when available."""
    try:
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def _percentiles(values: list[float]) -> tuple[float, float]:
    """Return mean and p95 milliseconds."""
    return float(np.mean(values)), float(np.percentile(values, 95))


def _measure(
    inputs: list[np.ndarray],
    torch_model: torch.nn.Module,
    onnx_session: ort.InferenceSession,
    input_name: str,
) -> tuple[float, float, float, float]:
    """Measure one-image inference after warmup."""
    torch_input = torch.from_numpy(inputs[0][None, ...])
    for _ in range(WARMUP_RUNS):
        with torch.inference_mode():
            torch_model(torch_input)
        onnx_session.run(["logits"], {input_name: inputs[0][None, ...]})

    torch_times: list[float] = []
    for image in inputs:
        started = time.perf_counter()
        with torch.inference_mode():
            torch_model(torch.from_numpy(image[None, ...]))
        torch_times.append((time.perf_counter() - started) * 1000)
    torch_mean, torch_p95 = _percentiles(torch_times)

    onnx_times: list[float] = []
    for image in inputs:
        started = time.perf_counter()
        onnx_session.run(["logits"], {input_name: image[None, ...]})
        onnx_times.append((time.perf_counter() - started) * 1000)
    onnx_mean, onnx_p95 = _percentiles(onnx_times)
    return torch_mean, torch_p95, onnx_mean, onnx_p95


def benchmark() -> dict[str, float | str]:
    """Measure PyTorch and ONNX Runtime on 200 validation images."""
    torch.set_num_threads(THREADS)
    torch.set_num_interop_threads(THREADS)
    records = load_records("val")[:SAMPLE_SIZE]
    predictor = PetBreedPredictor.load()
    inputs = []
    for record in records:
        with Image.open(cfg.PROJECT_ROOT / str(record["path"])) as image:
            inputs.append(predictor.transform(image).numpy().astype(np.float32))

    torch_model, _ = load_checkpoint(cfg.MODEL_PATH, torch.device("cpu"))
    session_options = ort.SessionOptions()
    session_options.intra_op_num_threads = THREADS
    session_options.inter_op_num_threads = THREADS
    session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    onnx_session = ort.InferenceSession(
        str(cfg.ONNX_PATH),
        sess_options=session_options,
        providers=["CPUExecutionProvider"],
    )
    torch_mean, torch_p95, onnx_mean, onnx_p95 = _measure(
        inputs, torch_model, onnx_session, onnx_session.get_inputs()[0].name
    )
    return {
        "cpu_model": _cpu_model_name(),
        "threads": THREADS,
        "torch_mean_ms": torch_mean,
        "torch_p95_ms": torch_p95,
        "onnx_mean_ms": onnx_mean,
        "onnx_p95_ms": onnx_p95,
    }


def _markdown_table(results: dict[str, float | str]) -> str:
    """Format benchmark results as a Markdown table."""
    return "\n".join(
        [
            "| Runtime | Mean latency (ms) | P95 latency (ms) | CPU | Threads |",
            "|---|---:|---:|---|---:|",
            f"| PyTorch eager | {results['torch_mean_ms']:.3f} | {results['torch_p95_ms']:.3f} | {results['cpu_model']} | {results['threads']} |",
            f"| ONNX Runtime | {results['onnx_mean_ms']:.3f} | {results['onnx_p95_ms']:.3f} | {results['cpu_model']} | {results['threads']} |",
        ]
    )


def main() -> None:
    """Run and log the serialization benchmark."""
    setup_logging()
    results = benchmark()
    logger.info("Serialization benchmark\n%s", _markdown_table(results))


if __name__ == "__main__":
    main()
