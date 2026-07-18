.PHONY: install sync lint format typecheck test build check demo up down logs

install:
	@command -v uv >/dev/null 2>&1 || { echo "uv is required" >&2; exit 1; }
	uv sync --all-extras
	uv run pre-commit install

sync:
	uv sync --all-extras

lint:
	uv run ruff check .

format:
	uv run ruff format .

typecheck:
	uv run basedpyright

test:
	uv run pytest -v --cov=agenttrace --cov-report=term-missing

build:
	uv build

check:
	uv run ruff format --check .
	uv run ruff check .
	uv run basedpyright
	uv run pytest -v --cov=agenttrace --cov-report=term-missing
	uv build

demo:
	uv run agenttrace demo --root outputs/demo

up:
	docker compose up --build -d

down:
	docker compose down

logs:
	docker compose logs -f collector otel-collector
