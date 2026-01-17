# Repository Guidelines

## Project Structure & Module Organization
- `main.py`: Core FastAPI app and domain models/utilities.
- `app/`: HTTP server wrapper and entrypoint (`app.main:run`).
- `aerospace_mcp/`: MCP server (`server.py`) and core logic (`core.py`).
- `tests/`: Pytest suite and fixtures.
- `docs/`, `Dockerfile`, `docker-compose.yml`, `kubernetes/`: Ops and docs.

## Build, Test, and Development Commands
- Setup (uv): `uv venv && source .venv/bin/activate && uv sync`
- Run HTTP API: `uv run aerospace-mcp-http` (or `uvicorn main:app --reload`)
- Run MCP (stdio): `uv run aerospace-mcp`
- Lint/Format: `ruff check . --fix && ruff format . && black .`
- Type check: `mypy .`
- Tests quick: `python run_tests.py` (or `pytest -q`)
- Coverage: `pytest --cov=aerospace_mcp --cov=app tests/`
- Docker (local): `docker compose up --build -d` → API on `:8080`

## Coding Style & Naming Conventions
- Python 3.11+, 4‑space indentation, UTF‑8, one import per line.
- Formatting: Black (target 88), Ruff formatter (line length 100, double quotes).
- Linting: Ruff rules enabled (imports, bugbear, pyupgrade, naming) with project ignores in `ruff.toml`.
- Types: Prefer explicit type hints; `mypy` strict config in `pyproject.toml`.
- Names: modules/files `snake_case.py`; classes `CamelCase`; functions/vars `snake_case`.

## Testing Guidelines
- Frameworks: `pytest`, `pytest-asyncio`, `httpx`, `pytest-cov`.
- Structure: unit and integration tests under `tests/` (see `TESTING.md`).
- Markers: `@pytest.mark.unit`, `@pytest.mark.integration`, `@pytest.mark.slow`.
- Coverage: maintain ≥80% for `aerospace_mcp/`, `app/`, and `main.py`.
- Examples: `pytest -m 'not slow'`, `pytest tests/test_airports.py::TestAirportResolution`.

## Commit & Pull Request Guidelines
- Messages: imperative, concise subject (≤72 chars), optional body with rationale.
  Example: `Add FastAPI health endpoint and tests`.
- PRs: clear description, link issues, include API examples or screenshots if UI/Docs change, note breaking changes.
- Checks: run `pre-commit` (`pre-commit install`), ensure `ruff`, `black`, `mypy`, and tests pass locally.

## Security & Configuration Tips
- Never commit secrets; use `.env` based on `.env.example`.
- Runtime env: `AEROSPACE_MCP_HOST`, `AEROSPACE_MCP_PORT`, `AEROSPACE_MCP_LOG_LEVEL`, optional `AEROSPACE_MCP_ENV`.
- Docker healthcheck expects `/health`; prefer read‑only FS and `tmpfs` as in `docker-compose.yml`.

## Architecture Overview
- Two interfaces: FastAPI HTTP (`app/`, `main.py`) and MCP tools (`aerospace_mcp/server.py`).
- OpenAP is optional; when unavailable, performance estimates are disabled gracefully.
