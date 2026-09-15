# =============================
#  Minimal Makefile for uv-based Python Project
# =============================

UV ?= uv

.DEFAULT_GOAL := help

.PHONY := venv sync dev hooks lint format pre_commit run clean help

# -----------------------------
#  Environment
# -----------------------------

# Create in-project virtual environment (.venv)
venv:
	$(UV) venv --project .

# Sync dependencies (install/update/remove)
sync:
	$(UV) sync

# Install development dependencies (ruff / black / isort / pre-commit)
dev:
	$(UV) sync --only-group dev --no-install-project

# Install pre-commit hooks
hooks:
	$(UV) run --no-sync pre-commit install

# -----------------------------
#  Code Quality
# -----------------------------

# Static analysis without auto-fixing
lint:
	$(UV) run --no-sync ruff check .

# Auto-format code (isort -> black keeps styles consistent)
format:
	$(UV) run --no-sync isort . --profile black
	$(UV) run --no-sync black .
	$(UV) run --no-sync ruff check . --fix

# -----------------------------
#  Utilities
# -----------------------------

# Run all pre-commit hooks manually
pre_commit:
	$(UV) run --no-sync pre-commit run --all-files

# Clean caches and venv
clean:
	rm -rf .venv .ruff_cache
	find . -type d -name "__pycache__" -prune -exec rm -rf {} +

# -----------------------------
#  Help Message
# -----------------------------

help:
	@echo "------------------ Available Commands ------------------"
	@echo "make venv        - Create in-project virtual environment"
	@echo "make sync        - Sync dependencies from pyproject/uv.lock"
	@echo "make dev         - Install development dependencies (dev group only)"
	@echo "make hooks       - Install pre-commit hooks"
	@echo "make format      - Auto-format code with isort/black/ruff"
	@echo "make lint        - Run Ruff in check-only mode"
	@echo "make pre_commit  - Run all pre-commit hooks"
	@echo "make clean       - Remove caches and virtual environment"
	@echo "--------------------------------------------------------"
