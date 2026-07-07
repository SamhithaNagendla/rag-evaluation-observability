.PHONY: install test lint run docker-up docker-down
install:
	pip install -e ".[dev]"
test:
	pytest --cov=rag_eval --cov-report=term-missing
lint:
	ruff check .
run:
	uvicorn rag_eval.api:app --reload
docker-up:
	docker compose up --build
docker-down:
	docker compose down -v
