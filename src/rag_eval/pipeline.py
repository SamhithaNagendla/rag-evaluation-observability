from __future__ import annotations

from rag_eval.llm import LLMClient


class RAGPipeline:
    def __init__(self, retriever, llm: LLMClient):
        self.retriever = retriever
        self.llm = llm

    def answer(self, question: str, k: int = 3) -> dict[str, object]:
        rows = self.retriever.search(question, k)
        documents = [document for document, _ in rows]
        return {
            "answer": self.llm.generate(question, documents),
            "documents": documents,
            "scores": [score for _, score in rows],
        }
