.PHONY: help install install-hf ingest api ui eval eval-compare test lint fmt db-up db-down docker-api

help:
	@echo "Targets:"
	@echo "  install      Install Python dependencies"
	@echo "  install-hf   Install optional Hugging Face deps (free local models)"
	@echo "  ingest       Build the vector index from data/docs (RESET=1 to clear first)"
	@echo "  api          Run the FastAPI service on :8000"
	@echo "  ui           Run the Streamlit UI (needs the API running)"
	@echo "  eval         Run the evaluation harness (NO_RAGAS=1 to skip Ragas)"
	@echo "  eval-compare Benchmark vector vs hybrid retrieval (A/B)"
	@echo "  test         Run the test suite (keyless fake path)"
	@echo "  lint         Run ruff linting"
	@echo "  fmt          Auto-fix lint issues with ruff"
	@echo "  db-up        Start Postgres+pgvector via docker compose"
	@echo "  docker-api   Build and run the API in a container"

install:
	pip install -r requirements.txt

install-hf:
	pip install -r requirements-hf.txt

ingest:
	python -m src.rag.ingest $(if $(RESET),--reset,)

api:
	uvicorn src.rag.api:app --reload --port 8000

ui:
	streamlit run ui/streamlit_app.py

eval:
	python -m eval.run_eval $(if $(NO_RAGAS),--no-ragas,)

eval-compare:
	python -m eval.run_eval --compare

test:
	pytest -q

lint:
	ruff check .

fmt:
	ruff check --fix .

db-up:
	docker compose up -d db

db-down:
	docker compose down

docker-api:
	docker compose up --build api
