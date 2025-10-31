PYTHON ?= python3
VENV := .venv
BIN := $(VENV)/bin
UV := $(shell command -v uv 2>/dev/null)

export PYTHONPATH := src

.DEFAULT_GOAL := help

.PHONY: help
help:
	@echo "Available targets:"
	@echo "  make venv        Create a local virtual environment"
	@echo "  make dev         Install project with development dependencies"
	@echo "  make lint        Run Ruff (with fixes) and Black formatting"
	@echo "  make typecheck   Run MyPy static type checks"
	@echo "  make test        Run the unit test suite"
	@echo "  make test-all    Run the entire test suite including slow/integration"
	@echo "  make ci          Run lint, typecheck, and unit tests"

$(BIN)/python:
	$(PYTHON) -m venv $(VENV)

.PHONY: venv
venv: $(BIN)/python

.PHONY: dev
dev: venv
ifdef UV
	uv pip install -e '.[dev]'
else
	$(BIN)/pip install --upgrade pip
	$(BIN)/pip install -e '.[dev]'
endif

.PHONY: lint
lint: dev
	$(BIN)/ruff check src tests --fix
	$(BIN)/black src tests

.PHONY: typecheck
typecheck: dev
	$(BIN)/mypy src

.PHONY: test
test: dev
	$(BIN)/pytest

.PHONY: test-all
test-all: dev
	$(BIN)/pytest --override-ini="addopts=-q --strict-markers --disable-warnings"

.PHONY: ci
ci: lint typecheck test
