IMAGE ?= solort:dev
LLM_IMAGE ?= solort:qwen3-4b-spec
NGC_IMAGE ?= solort:qwen3-4b-spec-ngc
CONTAINER ?= solort-api
PORT ?= 8000
BASELINE_PORT ?= 8002
CPU_PORT ?= 8001
HF_HOME ?= $(HOME)/.cache/huggingface
QWEN06B_MODEL ?= Qwen/Qwen3-0.6B
QWEN4B_MODEL ?= Qwen/Qwen3-4B
KV_TENSOR_STORAGE ?= 0
KV_NUM_PAGES ?= 1024
KV_PAGE_SIZE ?= 16

# Generic single-machine model knobs (default to the Qwen3 pairing). Override to serve any
# HF causal LM, e.g.:
#   make docker-ngc-up-model MODEL=meta-llama/Llama-3.2-3B-Instruct \
#       DRAFT_MODEL=meta-llama/Llama-3.2-1B-Instruct
#   make docker-ngc-up-model MODEL=google/gemma-2-2b-it SPEC_TOKENS=0 ATTENTION_BACKEND=sdpa
# Speculative decoding needs DRAFT_MODEL to share the target's tokenizer/vocab; set SPEC_TOKENS=0
# (and DRAFT_MODEL=) to disable it. Use ATTENTION_BACKEND=sdpa for architectures the FlashInfer
# bridge does not model exactly (e.g. Gemma2 logit soft-capping, MLA).
MODEL ?= $(QWEN4B_MODEL)
DRAFT_MODEL ?= $(QWEN06B_MODEL)
SPEC_TOKENS ?= 4
ATTENTION_BACKEND ?= flashinfer
TRUST_REMOTE_CODE ?= 0

.PHONY: docker-build docker-down docker-test docker-lint docker-shell docker-llm-build docker-llm-up docker-llm-shell docker-ngc-build docker-ngc-up docker-ngc-up-qwen4b docker-ngc-up-nospec docker-ngc-up-cpu docker-ngc-up-model docker-ngc-probe docker-ngc-shell docker-hf-prefetch docker-serving-bench docker-spec-bench compose-test compose-lint

docker-build:
	docker build --target dev -t $(IMAGE) .

docker-llm-build:
	docker build --target llm -t $(LLM_IMAGE) .

docker-ngc-build:
	docker build -f Dockerfile.ngc -t $(NGC_IMAGE) .

docker-down:
	-docker rm -f $(CONTAINER)

docker-test:
	docker run --rm \
		-v $(PWD)/src:/app/src \
		-v $(PWD)/tests:/app/tests \
		-v $(PWD)/benchmarks:/app/benchmarks \
		-v $(PWD)/README.md:/app/README.md \
		-v $(PWD)/docs:/app/docs \
		-v $(PWD)/pyproject.toml:/app/pyproject.toml \
		$(IMAGE) pytest

docker-lint:
	docker run --rm \
		-v $(PWD)/src:/app/src \
		-v $(PWD)/tests:/app/tests \
		-v $(PWD)/benchmarks:/app/benchmarks \
		-v $(PWD)/README.md:/app/README.md \
		-v $(PWD)/docs:/app/docs \
		-v $(PWD)/pyproject.toml:/app/pyproject.toml \
		$(IMAGE) ruff check .

docker-shell:
	docker run --rm -it \
		-v $(PWD)/src:/app/src \
		-v $(PWD)/tests:/app/tests \
		-v $(PWD)/benchmarks:/app/benchmarks \
		-v $(PWD)/README.md:/app/README.md \
		-v $(PWD)/docs:/app/docs \
		-v $(PWD)/pyproject.toml:/app/pyproject.toml \
		$(IMAGE) /bin/bash

docker-llm-up:
	mkdir -p $(HF_HOME)
	docker run --rm --gpus all --name $(CONTAINER)-qwen -p $(PORT):8000 \
		-e SOLORT_EXECUTOR=paged \
		-e SOLORT_MODEL_ID=$(QWEN4B_MODEL) \
		-e SOLORT_SPECULATIVE_DRAFT_MODEL_ID=$(QWEN06B_MODEL) \
		-e SOLORT_SPECULATIVE_TOKENS=4 \
		-e SOLORT_ATTENTION_BACKEND=flashinfer \
		-e SOLORT_KV_TENSOR_STORAGE=$(KV_TENSOR_STORAGE) \
		-e SOLORT_KV_NUM_PAGES=$(KV_NUM_PAGES) \
		-e SOLORT_KV_PAGE_SIZE=$(KV_PAGE_SIZE) \
		-e SOLORT_ENABLE_THINKING=0 \
		-e HF_HOME=/root/.cache/huggingface \
		-v $(HF_HOME):/root/.cache/huggingface \
		-v $(PWD)/src:/app/src \
		-v $(PWD)/docs:/app/docs \
		-v $(PWD)/scripts:/app/scripts \
		$(LLM_IMAGE)

docker-llm-shell:
	mkdir -p $(HF_HOME)
	docker run --rm -it --gpus all \
		-e SOLORT_EXECUTOR=paged \
		-e SOLORT_MODEL_ID=$(QWEN4B_MODEL) \
		-e SOLORT_SPECULATIVE_DRAFT_MODEL_ID=$(QWEN06B_MODEL) \
		-e SOLORT_SPECULATIVE_TOKENS=4 \
		-e SOLORT_ATTENTION_BACKEND=flashinfer \
		-e SOLORT_KV_TENSOR_STORAGE=$(KV_TENSOR_STORAGE) \
		-e SOLORT_KV_NUM_PAGES=$(KV_NUM_PAGES) \
		-e SOLORT_KV_PAGE_SIZE=$(KV_PAGE_SIZE) \
		-e HF_HOME=/root/.cache/huggingface \
		-v $(HF_HOME):/root/.cache/huggingface \
		-v $(PWD)/src:/app/src \
		-v $(PWD)/docs:/app/docs \
		-v $(PWD)/scripts:/app/scripts \
		$(LLM_IMAGE) /bin/bash

docker-ngc-up:
	mkdir -p $(HF_HOME)
	docker run --rm --gpus all --ipc=host \
		--ulimit memlock=-1 --ulimit stack=67108864 \
		--name $(CONTAINER)-ngc -p $(PORT):8000 \
		-e SOLORT_EXECUTOR=paged \
		-e SOLORT_MODEL_ID=$(QWEN4B_MODEL) \
		-e SOLORT_SPECULATIVE_DRAFT_MODEL_ID=$(QWEN06B_MODEL) \
		-e SOLORT_SPECULATIVE_TOKENS=4 \
		-e SOLORT_ATTENTION_BACKEND=flashinfer \
		-e SOLORT_KV_TENSOR_STORAGE=$(KV_TENSOR_STORAGE) \
		-e SOLORT_KV_NUM_PAGES=$(KV_NUM_PAGES) \
		-e SOLORT_KV_PAGE_SIZE=$(KV_PAGE_SIZE) \
		-e SOLORT_ENABLE_THINKING=0 \
		-e HF_HOME=/root/.cache/huggingface \
		-v $(HF_HOME):/root/.cache/huggingface \
		-v $(PWD)/src:/app/src \
		-v $(PWD)/docs:/app/docs \
		-v $(PWD)/scripts:/app/scripts \
		$(NGC_IMAGE)

docker-ngc-up-qwen4b:
	mkdir -p $(HF_HOME)
	docker run --rm --gpus all --ipc=host \
		--ulimit memlock=-1 --ulimit stack=67108864 \
		--name $(CONTAINER)-ngc -p $(PORT):8000 \
		-e SOLORT_EXECUTOR=paged \
		-e SOLORT_MODEL_ID=$(QWEN4B_MODEL) \
		-e SOLORT_SPECULATIVE_DRAFT_MODEL_ID=$(QWEN06B_MODEL) \
		-e SOLORT_SPECULATIVE_TOKENS=4 \
		-e SOLORT_ATTENTION_BACKEND=flashinfer \
		-e SOLORT_KV_TENSOR_STORAGE=$(KV_TENSOR_STORAGE) \
		-e SOLORT_KV_NUM_PAGES=$(KV_NUM_PAGES) \
		-e SOLORT_KV_PAGE_SIZE=$(KV_PAGE_SIZE) \
		-e SOLORT_ENABLE_THINKING=0 \
		-e HF_HOME=/root/.cache/huggingface \
		-v $(HF_HOME):/root/.cache/huggingface \
		-v $(PWD)/src:/app/src \
		-v $(PWD)/docs:/app/docs \
		-v $(PWD)/scripts:/app/scripts \
		$(NGC_IMAGE)

docker-ngc-up-nospec:
	mkdir -p $(HF_HOME)
	docker run --rm --gpus all --ipc=host \
		--ulimit memlock=-1 --ulimit stack=67108864 \
		--name $(CONTAINER)-ngc-nospec -p $(BASELINE_PORT):8000 \
		-e SOLORT_EXECUTOR=paged \
		-e SOLORT_MODEL_ID=$(QWEN4B_MODEL) \
		-e SOLORT_SPECULATIVE_DRAFT_MODEL_ID= \
		-e SOLORT_SPECULATIVE_TOKENS=0 \
		-e SOLORT_ATTENTION_BACKEND=flashinfer \
		-e SOLORT_KV_TENSOR_STORAGE=$(KV_TENSOR_STORAGE) \
		-e SOLORT_KV_NUM_PAGES=$(KV_NUM_PAGES) \
		-e SOLORT_KV_PAGE_SIZE=$(KV_PAGE_SIZE) \
		-e SOLORT_ENABLE_THINKING=0 \
		-e HF_HOME=/root/.cache/huggingface \
		-v $(HF_HOME):/root/.cache/huggingface \
		-v $(PWD)/src:/app/src \
		-v $(PWD)/docs:/app/docs \
		-v $(PWD)/scripts:/app/scripts \
		$(NGC_IMAGE)

docker-ngc-up-cpu:
	mkdir -p $(HF_HOME)
	docker run --rm \
		--name $(CONTAINER)-ngc-cpu -p $(CPU_PORT):8000 \
		-e SOLORT_EXECUTOR=transformers \
		-e SOLORT_MODEL_ID=$(QWEN06B_MODEL) \
		-e SOLORT_DEVICE_MAP=cpu \
		-e SOLORT_TORCH_DTYPE=float32 \
		-e SOLORT_KV_TENSOR_STORAGE=$(KV_TENSOR_STORAGE) \
		-e SOLORT_KV_NUM_PAGES=$(KV_NUM_PAGES) \
		-e SOLORT_KV_PAGE_SIZE=$(KV_PAGE_SIZE) \
		-e SOLORT_ENABLE_THINKING=0 \
		-e HF_HOME=/root/.cache/huggingface \
		-v $(HF_HOME):/root/.cache/huggingface \
		-v $(PWD)/src:/app/src \
		-v $(PWD)/docs:/app/docs \
		-v $(PWD)/scripts:/app/scripts \
		$(NGC_IMAGE)

docker-ngc-up-model:
	mkdir -p $(HF_HOME)
	docker run --rm --gpus all --ipc=host \
		--ulimit memlock=-1 --ulimit stack=67108864 \
		--name $(CONTAINER)-ngc -p $(PORT):8000 \
		-e SOLORT_EXECUTOR=paged \
		-e SOLORT_MODEL_ID=$(MODEL) \
		-e SOLORT_SPECULATIVE_DRAFT_MODEL_ID=$(DRAFT_MODEL) \
		-e SOLORT_SPECULATIVE_TOKENS=$(SPEC_TOKENS) \
		-e SOLORT_ATTENTION_BACKEND=$(ATTENTION_BACKEND) \
		-e SOLORT_TRUST_REMOTE_CODE=$(TRUST_REMOTE_CODE) \
		-e SOLORT_KV_TENSOR_STORAGE=$(KV_TENSOR_STORAGE) \
		-e SOLORT_KV_NUM_PAGES=$(KV_NUM_PAGES) \
		-e SOLORT_KV_PAGE_SIZE=$(KV_PAGE_SIZE) \
		-e SOLORT_ENABLE_THINKING=0 \
		-e HF_HOME=/root/.cache/huggingface \
		-v $(HF_HOME):/root/.cache/huggingface \
		-v $(PWD)/src:/app/src \
		-v $(PWD)/docs:/app/docs \
		-v $(PWD)/scripts:/app/scripts \
		$(NGC_IMAGE)

docker-serving-bench:
	python benchmarks/bench_serving.py \
		--case gpu=http://127.0.0.1:$(PORT) \
		--case cpu=http://127.0.0.1:$(CPU_PORT)

docker-spec-bench:
	python benchmarks/bench_serving.py \
		--case spec=http://127.0.0.1:$(PORT) \
		--case nospec=http://127.0.0.1:$(BASELINE_PORT) \
		--model $(QWEN4B_MODEL) \
		--temperature 0 \
		--max-tokens 128 \
		--warmup 1 \
		--runs 3

docker-ngc-probe:
	docker run --rm --gpus all --ipc=host \
		--ulimit memlock=-1 --ulimit stack=67108864 \
		$(NGC_IMAGE) python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"

docker-ngc-shell:
	mkdir -p $(HF_HOME)
	docker run --rm -it --gpus all --ipc=host \
		--ulimit memlock=-1 --ulimit stack=67108864 \
		-e SOLORT_EXECUTOR=paged \
		-e SOLORT_MODEL_ID=$(QWEN4B_MODEL) \
		-e SOLORT_SPECULATIVE_DRAFT_MODEL_ID=$(QWEN06B_MODEL) \
		-e SOLORT_SPECULATIVE_TOKENS=4 \
		-e SOLORT_ATTENTION_BACKEND=flashinfer \
		-e SOLORT_KV_TENSOR_STORAGE=$(KV_TENSOR_STORAGE) \
		-e SOLORT_KV_NUM_PAGES=$(KV_NUM_PAGES) \
		-e SOLORT_KV_PAGE_SIZE=$(KV_PAGE_SIZE) \
		-e HF_HOME=/root/.cache/huggingface \
		-v $(HF_HOME):/root/.cache/huggingface \
		-v $(PWD)/src:/app/src \
		-v $(PWD)/docs:/app/docs \
		-v $(PWD)/scripts:/app/scripts \
		$(NGC_IMAGE) /bin/bash

docker-hf-prefetch:
	mkdir -p $(HF_HOME)
	docker run --rm --ipc=host \
		--ulimit memlock=-1 --ulimit stack=67108864 \
		-e HF_HOME=/root/.cache/huggingface \
		-e HF_HUB_ENABLE_HF_TRANSFER=1 \
		-e QWEN4B_MODEL=$(QWEN4B_MODEL) \
		-e QWEN06B_MODEL=$(QWEN06B_MODEL) \
		-e MODEL=$(MODEL) \
		-e DRAFT_MODEL=$(DRAFT_MODEL) \
		-v $(HF_HOME):/root/.cache/huggingface \
		-v $(PWD)/scripts:/app/scripts \
		$(NGC_IMAGE) python /app/scripts/prefetch_hf_models.py

compose-test:
	docker compose --profile test run --rm test

compose-lint:
	docker compose --profile lint run --rm lint
