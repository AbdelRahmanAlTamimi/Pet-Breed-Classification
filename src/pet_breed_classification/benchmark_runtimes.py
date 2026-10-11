"""Compare ONNX Runtime and OpenVINO on CPU with one protocol.

``uv run --group openvino python -m pet_breed_classification.benchmark_runtimes``

Both engines run the shipped ONNX model behind the same ``PetBreedPredictor`` (same eval
transform, temperature and threshold) with the same thread count. Latency covers only
the engine call on an already preprocessed batch of one image: 50 warm-up runs are
discarded, then at least 500 timed runs cycle through 200 val images. Agreement is
measured on the calibrated probabilities of those 200 images.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from collections.abc import Sequence
from typing import Any

import numpy as np
import onnxruntime as ort
from PIL import Image

from .config import cfg
from .data import load_records
from .openvino_runtime import OpenVINOSession, save_ir
from .predict import PetBreedPredictor

IMAGE_COUNT = 200
WARMUP_RUNS = 50
TIMED_RUNS = 500
THREADS = 6
REPORT_PATH = cfg.REPORTS_DIR / "runtime_benchmark.json"
IR_PATH = cfg.ARTIFACTS_DIR / "openvino" / "model.xml"


def cpu_model() -> str:
    with open("/proc/cpuinfo", encoding="utf-8") as cpuinfo:
        for line in cpuinfo:
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or "unknown"


def latency_ms(predictor: PetBreedPredictor, inputs: list[np.ndarray], runs: int) -> dict[str, float]:
    """Return p50/p95/mean latency of the engine call for batch-1 inputs."""
    for index in range(WARMUP_RUNS):
        predictor.session.run(["logits"], {predictor.input_name: inputs[index % len(inputs)]})
    timings = []
    for index in range(runs):
        feed = {predictor.input_name: inputs[index % len(inputs)]}
        started_at = time.perf_counter()
        predictor.session.run(["logits"], feed)
        timings.append((time.perf_counter() - started_at) * 1000)
    array = np.asarray(timings)
    return {
        "p50_ms": float(np.percentile(array, 50)),
        "p95_ms": float(np.percentile(array, 95)),
        "mean_ms": float(array.mean()),
        "timed_runs": runs,
    }


def run(runs: int = TIMED_RUNS, threads: int = THREADS) -> dict[str, Any]:
    metadata = PetBreedPredictor.load().metadata
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    predictors = {
        "onnxruntime": PetBreedPredictor(
            ort.InferenceSession(str(cfg.ONNX_PATH), options, providers=["CPUExecutionProvider"]),
            metadata,
        ),
        "openvino": PetBreedPredictor(OpenVINOSession(cfg.ONNX_PATH, threads=threads), metadata),
    }
    images = []
    for record in load_records("val")[:IMAGE_COUNT]:
        with Image.open(cfg.PROJECT_ROOT / str(record["path"])) as image:
            images.append(image.convert("RGB"))
    reference = predictors["onnxruntime"]
    inputs = [reference.transform(image).numpy()[None].astype(np.float32) for image in images]

    probabilities = {name: predictor.probabilities(images) for name, predictor in predictors.items()}
    ort_top1 = probabilities["onnxruntime"].argmax(axis=1)
    ov_top1 = probabilities["openvino"].argmax(axis=1)
    import openvino

    return {
        "images": len(images),
        "batch_size": 1,
        "warmup_runs": WARMUP_RUNS,
        "threads": threads,
        "cpu": cpu_model(),
        "onnx_sha256": metadata["onnx_sha256"],
        "versions": {"onnxruntime": ort.__version__, "openvino": openvino.__version__},
        "latency": {name: latency_ms(predictor, inputs, runs) for name, predictor in predictors.items()},
        "top1_agreement": float(np.mean(ort_top1 == ov_top1)),
        "max_abs_probability_diff": float(
            np.abs(probabilities["onnxruntime"] - probabilities["openvino"]).max()
        ),
        "openvino_ir": IR_PATH.relative_to(cfg.PROJECT_ROOT).as_posix(),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Benchmark ONNX Runtime against OpenVINO.")
    parser.add_argument("--runs", type=int, default=TIMED_RUNS)
    parser.add_argument("--threads", type=int, default=THREADS)
    args = parser.parse_args(argv)
    if args.runs < TIMED_RUNS:
        parser.error(f"use at least {TIMED_RUNS} timed runs")
    save_ir(cfg.ONNX_PATH, IR_PATH)
    result = run(args.runs, args.threads)
    REPORT_PATH.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
