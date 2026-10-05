# Build prerequisite: artifacts/model.onnx and artifacts/model_meta.json must
# exist locally. Create them with `uv run python -m pet_breed_classification.export`,
# then build from the repository root with:
# docker build -f docker/Dockerfile -t pet-breed:0.1.0 .
FROM python:3.11-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:0.9.5 /uv /uvx /bin/

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN set -eux; \
    uv export --frozen --no-dev --no-group train --no-emit-project \
        --format requirements.txt --no-hashes --no-header \
        > /tmp/requirements.lock.txt; \
    awk 'tolower($0) ~ /^(torch==|torchvision==|nvidia-|cuda-|triton==)/ { next } { print }' \
        /tmp/requirements.lock.txt > /tmp/requirements.runtime.txt; \
    awk -F '==' '$1 == "torch" { print $2; exit }' \
        /tmp/requirements.lock.txt > /tmp/torch.version; \
    awk -F '==' '$1 == "torchvision" { print $2; exit }' \
        /tmp/requirements.lock.txt > /tmp/torchvision.version; \
    test -s /tmp/torch.version; \
    test -s /tmp/torchvision.version
RUN uv venv /opt/venv \
    && uv pip install --python /opt/venv/bin/python -r /tmp/requirements.runtime.txt \
    && uv pip install --python /opt/venv/bin/python \
        --no-deps \
        --index-url https://download.pytorch.org/whl/cpu \
        "torch==$(cat /tmp/torch.version)+cpu" \
        "torchvision==$(cat /tmp/torchvision.version)+cpu"

COPY src ./src
COPY README.md ./README.md
RUN uv pip install --python /opt/venv/bin/python --no-deps .

FROM python:3.11-slim AS runtime

WORKDIR /app
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PETBREED_PROJECT_ROOT=/app \
    PETBREED_ARTIFACTS_DIR=/app/artifacts \
    PETBREED_ONNX_PATH=/app/artifacts/model.onnx \
    PETBREED_MODEL_META_PATH=/app/artifacts/model_meta.json

COPY --from=builder /opt/venv /opt/venv
COPY --chown=1000:1000 artifacts/model.onnx artifacts/model_meta.json /app/artifacts/

RUN useradd --create-home --uid 1000 appuser
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)"

CMD ["uvicorn", "pet_breed_classification.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--no-access-log"]
