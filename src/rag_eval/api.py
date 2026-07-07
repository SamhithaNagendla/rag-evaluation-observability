from __future__ import annotations

import json
from functools import lru_cache
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from rag_eval.config import get_settings
from rag_eval.evaluator import evaluate
from rag_eval.llm import BedrockLLMClient
from rag_eval.retrieval import BM25Retriever, HybridRetriever, TfidfRetriever
from rag_eval.store import EvaluationStore

RETRIEVERS = {
    "bm25": BM25Retriever,
    "tfidf": TfidfRetriever,
    "hybrid": HybridRetriever,
}


class Request(BaseModel):
    strategy: str = Field(default="hybrid", pattern="^(bm25|tfidf|hybrid)$")
    k: int = Field(default=3, ge=1, le=20)


@lru_cache
def get_documents():
    settings = get_settings()
    return json.loads(settings.documents_path.read_text(encoding="utf-8"))


@lru_cache
def get_qa():
    settings = get_settings()
    return json.loads(settings.qa_path.read_text(encoding="utf-8"))


@lru_cache
def get_llm():
    settings = get_settings()
    return BedrockLLMClient(
        region=settings.aws_region,
        model_id=settings.bedrock_model_id,
        max_tokens=settings.max_tokens,
        temperature=settings.temperature,
    )


@lru_cache
def get_store():
    return EvaluationStore(get_settings().results_db_path)


LLMDependency = Annotated[object, Depends(get_llm)]
StoreDependency = Annotated[EvaluationStore, Depends(get_store)]


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title=settings.app_name, version=settings.app_version)

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "healthy", "version": settings.app_version}

    @app.post("/evaluate")
    def run(request: Request, llm: LLMDependency, store: StoreDependency):
        retriever = RETRIEVERS[request.strategy](get_documents())
        result = evaluate(retriever, llm, get_qa(), request.k)
        run_id = store.save(request.strategy, request.k, result["summary"], result["rows"])
        return {"run_id": run_id, "strategy": request.strategy, **result}

    @app.get("/runs")
    def runs(
        store: StoreDependency,
        limit: int = Query(default=20, ge=1, le=100),
    ):
        return store.list(limit)

    @app.get("/runs/{run_id}")
    def get_run(run_id: str, store: StoreDependency):
        result = store.get(run_id)
        if result is None:
            raise HTTPException(status_code=404, detail="run not found")
        return result

    return app


app = create_app()
