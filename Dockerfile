# syntax=docker/dockerfile:1.7

FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app/src \
    HF_HOME=/root/.cache/huggingface

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
COPY docs ./docs
COPY scripts ./scripts

RUN python -m pip install --upgrade pip \
    && python -m pip install .

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2).read()"

CMD ["uvicorn", "solort.api.server:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]


FROM base AS dev

COPY tests ./tests
COPY benchmarks ./benchmarks

RUN python -m pip install -e ".[dev]"

CMD ["uvicorn", "solort.api.server:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000", "--reload"]


FROM dev AS llm

ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu126

RUN python -m pip install torch==2.10.0 --index-url "${TORCH_INDEX_URL}" \
    && python -m pip install -e ".[dev,model]"

ENV SOLORT_EXECUTOR=paged \
    SOLORT_MODEL_ID=Qwen/Qwen3-4B \
    SOLORT_SPECULATIVE_DRAFT_MODEL_ID=Qwen/Qwen3-0.6B \
    SOLORT_SPECULATIVE_TOKENS=4 \
    SOLORT_ATTENTION_BACKEND=flashinfer \
    SOLORT_DEVICE_MAP=auto \
    SOLORT_TORCH_DTYPE=auto \
    SOLORT_ENABLE_THINKING=0 \
    HF_HOME=/root/.cache/huggingface

CMD ["uvicorn", "solort.api.server:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]


FROM nvcr.io/nvidia/pytorch:24.07-py3 AS ngc

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app/src \
    SOLORT_EXECUTOR=paged \
    SOLORT_MODEL_ID=Qwen/Qwen3-4B \
    SOLORT_SPECULATIVE_DRAFT_MODEL_ID=Qwen/Qwen3-0.6B \
    SOLORT_SPECULATIVE_TOKENS=4 \
    SOLORT_ATTENTION_BACKEND=flashinfer \
    SOLORT_DEVICE_MAP=auto \
    SOLORT_TORCH_DTYPE=auto \
    SOLORT_ENABLE_THINKING=0 \
    HF_HOME=/root/.cache/huggingface

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
COPY docs ./docs
COPY scripts ./scripts
COPY tests ./tests
COPY benchmarks ./benchmarks

RUN python -m pip install --upgrade pip \
    && python -m pip install -e ".[dev]" \
    && python -m pip install "accelerate>=0.34" "transformers>=4.51.0,<5" "hf_transfer>=0.1.8" "flashinfer-python>=0.2"

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2).read()"

CMD ["uvicorn", "solort.api.server:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
