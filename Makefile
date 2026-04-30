# Makefile — Autonomous Oncology Board
# Shortcuts for common development tasks inside the ROCm Docker container.
#
# Usage:
#   make smoke      — run the Day 1 smoke test
#   make api        — start the FastAPI server
#   make index      — index the oncology corpus into Qdrant
#   make vram       — start continuous VRAM monitoring
#   make lint       — run ruff linter
#   make test       — run pytest

PYTHON     := python3
PYTHONPATH := $(shell pwd)
PORT       := 8000

.PHONY: help smoke api index vram lint test clean

help:
	@echo ""
	@echo "  AOB — Autonomous Oncology Board"
	@echo ""
	@echo "  make smoke    Run smoke test (validates GigaPath + Llama in VRAM)"
	@echo "  make api      Start FastAPI server on port $(PORT)"
	@echo "  make index    Index oncology corpus into Qdrant"
	@echo "  make vram     Start live VRAM monitor (rocm-smi loop)"
	@echo "  make lint     Run ruff linter"
	@echo "  make test     Run pytest"
	@echo "  make clean    Remove __pycache__ and .pyc files"
	@echo ""

smoke:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/smoke_test.py

api:
	PYTHONPATH=$(PYTHONPATH) uvicorn ml.api:app \
		--host 0.0.0.0 \
		--port $(PORT) \
		--reload \
		--log-level info

index:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/index_corpus.py

index-force:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/index_corpus.py --force

vram:
	bash scripts/vram_monitor.sh

lint:
	ruff check ml/ scripts/

test:
	PYTHONPATH=$(PYTHONPATH) pytest tests/ -v

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete
	find . -name "*.pyo" -delete
