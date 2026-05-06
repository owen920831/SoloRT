IMAGE ?= solort:dev
LLM_IMAGE ?= solort:qwen3-0.6b
NGC_IMAGE ?= solort:qwen3-0.6b-ngc
CONTAINER ?= solort-api
PORT ?= 8000
CPU_PORT ?= 8001
HF_HOME ?= /tmp/solort-hf-cache
QWEN06B_MODEL ?= Qwen/Qwen3-0.6B
QWEN4B_MODEL ?= Qwen/Qwen3-4B

.PHONY: docker-build docker-up docker-down docker-test docker-lint docker-shell docker-llm-build docker-llm-up docker-llm-shell docker-ngc-build docker-ngc-up docker-ngc-up-qwen4b docker-ngc-up-cpu docker-ngc-probe docker-ngc-shell docker-serving-bench compose-up compose-test compose-lint

docker-build:
	docker build --target dev -t $(IMAGE) .

docker-llm-build:
	docker build --target llm -t $(LLM_IMAGE) .

docker-ngc-build:
	docker build -f Dockerfile.ngc -t $(NGC_IMAGE) .

docker-up:
	docker run --rm --name $(CONTAINER) -p $(PORT):8000 \
		-v $(PWD)/src:/app/src \
		-v $(PWD)/tests:/app/tests \
		-v $(PWD)/benchmarks:/app/benchmarks \
		-v $(PWD)/README.md:/app/README.md \
		-v $(PWD)/pyproject.toml:/app/pyproject.toml \
		$(IMAGE)

docker-down:
	-docker rm -f $(CONTAINER)

docker-test:
	docker run --rm \
		-v $(PWD)/src:/app/src \
		-v $(PWD)/tests:/app/tests \
		-v $(PWD)/benchmarks:/app/benchmarks \
		-v $(PWD)/README.md:/app/README.md \
		-v $(PWD)/pyproject.toml:/app/pyproject.toml \
		$(IMAGE) pytest

docker-lint:
	docker run --rm \
		-v $(PWD)/src:/app/src \
		-v $(PWD)/tests:/app/tests \
		-v $(PWD)/benchmarks:/app/benchmarks \
		-v $(PWD)/README.md:/app/README.md \
		-v $(PWD)/pyproject.toml:/app/pyproject.toml \
		$(IMAGE) ruff check .

docker-shell:
	docker run --rm -it \
		-v $(PWD)/src:/app/src \
		-v $(PWD)/tests:/app/tests \
		-v $(PWD)/benchmarks:/app/benchmarks \
		-v $(PWD)/README.md:/app/README.md \
		-v $(PWD)/pyproject.toml:/app/pyproject.toml \
		$(IMAGE) /bin/bash

docker-llm-up:
	docker run --rm --gpus all --name $(CONTAINER)-qwen -p $(PORT):8000 \
		-e SOLORT_EXECUTOR=transformers \
		-e SOLORT_MODEL_ID=$(QWEN06B_MODEL) \
		-e SOLORT_ENABLE_THINKING=0 \
		-v $(HF_HOME):/root/.cache/huggingface \
		$(LLM_IMAGE)

docker-llm-shell:
	docker run --rm -it --gpus all \
		-e SOLORT_EXECUTOR=transformers \
		-e SOLORT_MODEL_ID=$(QWEN06B_MODEL) \
		-v $(HF_HOME):/root/.cache/huggingface \
		$(LLM_IMAGE) /bin/bash

docker-ngc-up:
	mkdir -p $(HF_HOME)
	docker run --rm --gpus all --ipc=host \
		--ulimit memlock=-1 --ulimit stack=67108864 \
		--name $(CONTAINER)-ngc -p $(PORT):8000 \
		-e SOLORT_EXECUTOR=transformers \
		-e SOLORT_MODEL_ID=$(QWEN06B_MODEL) \
		-e SOLORT_ENABLE_THINKING=0 \
		-v $(HF_HOME):/root/.cache/huggingface \
		$(NGC_IMAGE)

docker-ngc-up-qwen4b:
	mkdir -p $(HF_HOME)
	docker run --rm --gpus all --ipc=host \
		--ulimit memlock=-1 --ulimit stack=67108864 \
		--name $(CONTAINER)-ngc -p $(PORT):8000 \
		-e SOLORT_EXECUTOR=transformers \
		-e SOLORT_MODEL_ID=$(QWEN4B_MODEL) \
		-e SOLORT_ENABLE_THINKING=0 \
		-v $(HF_HOME):/root/.cache/huggingface \
		$(NGC_IMAGE)

docker-ngc-up-cpu:
	mkdir -p $(HF_HOME)
	docker run --rm \
		--name $(CONTAINER)-ngc-cpu -p $(CPU_PORT):8000 \
		-e SOLORT_EXECUTOR=transformers \
		-e SOLORT_MODEL_ID=$(QWEN06B_MODEL) \
		-e SOLORT_DEVICE_MAP=cpu \
		-e SOLORT_TORCH_DTYPE=float32 \
		-e SOLORT_ENABLE_THINKING=0 \
		-v $(HF_HOME):/root/.cache/huggingface \
		$(NGC_IMAGE)

docker-serving-bench:
	python benchmarks/bench_serving.py \
		--case gpu=http://127.0.0.1:$(PORT) \
		--case cpu=http://127.0.0.1:$(CPU_PORT)

docker-ngc-probe:
	docker run --rm --gpus all --ipc=host \
		--ulimit memlock=-1 --ulimit stack=67108864 \
		$(NGC_IMAGE) python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"

docker-ngc-shell:
	mkdir -p $(HF_HOME)
	docker run --rm -it --gpus all --ipc=host \
		--ulimit memlock=-1 --ulimit stack=67108864 \
		-e SOLORT_EXECUTOR=transformers \
		-e SOLORT_MODEL_ID=$(QWEN06B_MODEL) \
		-v $(HF_HOME):/root/.cache/huggingface \
		$(NGC_IMAGE) /bin/bash

compose-up:
	docker compose up --build solort

compose-test:
	docker compose --profile test run --rm test

compose-lint:
	docker compose --profile lint run --rm lint
