from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from rag_eval.api import create_app, get_documents, get_llm, get_qa, get_store
from rag_eval.store import EvaluationStore

DOCS = [
    {"id": "a", "text": "Cats are mammals."},
    {"id": "b", "text": "Raft uses a leader and replicated log with majority agreement."},
]
QA = [
    {
        "id": "q",
        "question": "How does Raft replicate data?",
        "relevant_doc_ids": ["b"],
        "reference_answer": "Raft replicates a leader log with majority agreement.",
    }
]


class FakeLLM:
    def generate(self, question, documents):
        return f"Raft uses a leader and majority agreement [{documents[0]['id']}]."

    def judge(self, question, answer, documents, reference_answer):
        return {
            "answer_relevance": 0.95,
            "faithfulness": 0.9,
            "correctness": 0.92 if reference_answer else None,
            "rationale": "Supported by the retrieved document.",
        }


@pytest.fixture
def fake_llm():
    return FakeLLM()


@pytest.fixture
def api_client(tmp_path: Path, fake_llm):
    app = create_app()
    store = EvaluationStore(tmp_path / "runs.db")
    app.dependency_overrides[get_llm] = lambda: fake_llm
    app.dependency_overrides[get_store] = lambda: store
    app.dependency_overrides[get_documents] = lambda: DOCS
    app.dependency_overrides[get_qa] = lambda: QA
    with TestClient(app) as client:
        yield client
