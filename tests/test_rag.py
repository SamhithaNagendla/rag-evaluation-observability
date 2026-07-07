from rag_eval.evaluator import evaluate
from rag_eval.pipeline import RAGPipeline
from rag_eval.retrieval import BM25Retriever, HybridRetriever, TfidfRetriever

DOCS = [
    {"id": "a", "text": "cats are mammals"},
    {"id": "b", "text": "raft uses a leader and replicated log"},
]


class FakeLLM:
    def generate(self, question, documents):
        return f"Answer from {documents[0]['id']}"

    def judge(self, question, answer, documents, reference_answer):
        return {
            "answer_relevance": 0.8,
            "faithfulness": 0.9,
            "correctness": 1.0 if reference_answer else None,
            "rationale": "test",
        }


def test_retrievers() -> None:
    assert BM25Retriever(DOCS).search("raft leader")[0][0]["id"] == "b"
    assert TfidfRetriever(DOCS).search("cats")[0][0]["id"] == "a"
    assert HybridRetriever(DOCS).search("replicated log")[0][0]["id"] == "b"


def test_pipeline_uses_llm_generation() -> None:
    output = RAGPipeline(BM25Retriever(DOCS), FakeLLM()).answer("raft leader")
    assert output["answer"] == "Answer from b"


def test_evaluation_uses_llm_judgment() -> None:
    result = evaluate(
        BM25Retriever(DOCS),
        FakeLLM(),
        [
            {
                "id": "q",
                "question": "raft leader",
                "relevant_doc_ids": ["b"],
                "reference_answer": "leader log",
            }
        ],
    )
    assert result["summary"]["recall_at_k"] == 1
    assert result["summary"]["faithfulness"] == 0.9
    assert result["summary"]["correctness"] == 1.0
