# Module 1

## Serialization Comparison

| Format | Human-readable | Cross-language | Schema-enforced | Safe to load from an untrusted source |
|---|---|---|---|---|
| JSON | Yes | Yes | No | Yes, when parsed as data |
| Protobuf | No | Yes | Yes | Yes, when parsed with a trusted schema |
| Pickle / `.pt` | No | No | No | No |
| ONNX | No | Yes | Yes, through the graph model | No, unless the file is trusted and verified |

The service serves ONNX because ONNX Runtime provides a language-neutral CPU inference format with a verified dynamic batch axis and lower measured latency than PyTorch eager on the benchmark CPU.

Pickle executes arbitrary code on load. A `.pkl` or `.pt` file not produced by this project must never be loaded.

## Serialization Parity

The random-weight parity test and the real checkpoint parity test on 200 validation images passed with `np.allclose(atol=1e-4)` and matching argmax outputs. The exported graph uses raw logits; temperature scaling remains in the predictor.

## CPU Latency

Measured on 200 validation images with batch size 1, 20 warmup runs, CPU-only execution, and one thread for both runtimes.

| Runtime | Mean latency (ms) | P95 latency (ms) | CPU | Threads |
|---|---:|---:|---|---:|
| PyTorch eager | 91.189 | 92.969 | 12th Gen Intel(R) Core(TM) i5-12500 | 1 |
| ONNX Runtime | 67.491 | 68.598 | 12th Gen Intel(R) Core(TM) i5-12500 | 1 |

## Containerization

The multi-stage image was built as `pet-breed:0.1.0` and `pet-breed:latest` with Python 3.11. The final image contains CPU-only `torch==2.14.1+cpu` and `torchvision==0.29.1+cpu`; `uv pip list --python /opt/venv/bin/python | grep -i nvidia` returned no matches. The final image is 1,590.785 MB, below the approximately 2.5 GB stop threshold.

| Build | Image size |
|---|---:|
| Multi-stage runtime image | 1,590.785 MB |
| Throwaway single-stage image | 1,701.647 MB |

The multi-stage image is 110.862 MB smaller. The `.dockerignore` build context measured 94.29 MB in a no-cache build. Without `.dockerignore`, the unfiltered checkout context upper bound is 15,548,820,191 bytes (15.549 GB), dominated by local environment and dataset files.

Compose intentionally uses `image: pet-breed:latest` and does not mount `artifacts/`; the model and metadata are baked into the image so an empty host artifacts directory cannot shadow them. A reviewer without the ignored artifacts can pull the tagged image after it is pushed, rather than rebuilding from a context that lacks the model.

## Test Coverage

| File | Coverage |
|---|---:|
| `__init__.py` | 80% |
| `api/main.py` | 98% |
| `api/schemas.py` | 100% |
| `benchmark_serialization.py` | 63% |
| `config.py` | 100% |
| `data/__init__.py` | 67% |
| `data/build_manifest.py` | 0% |
| `data/corruptions.py` | 48% |
| `data/split_data.py` | 85% |
| `export.py` | 93% |
| `logging_conf.py` | 100% |
| `model.py` | 89% |
| `predict.py` | 99% |
| `transforms.py` | 92% |
| **Total** | **76.76%** |

`train.py` is excluded from coverage because its full GPU/data-loader behavior is verified by the separate training acceptance run.
