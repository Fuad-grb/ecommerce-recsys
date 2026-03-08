PYTHON = python3
DOCKER_COMPOSE = docker-compose

.PHONY: help lint format test up down

help:
	@echo "Commands:"
	@echo "  make lint    - Check code with linters (Ruff, Mypy)"
	@echo "  make format  - Automatically format code"
	@echo "  make test    - Run tests"
	@echo "  make up      - Bring up infrastructure in Docker"
	@echo "  make down    - Stop and remove containers"

# --- Quality of code ---

lint:
	ruff check .
	mypy .

format:
	ruff format .

test:
	pytest tests/

# --- Infrastructure ---

up:
	$(DOCKER_COMPOSE) up -d

down:
	$(DOCKER_COMPOSE) down

# --- Cleanup ---
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	rm -rf .pytest_cache .mypy_cache .ruff_cache
