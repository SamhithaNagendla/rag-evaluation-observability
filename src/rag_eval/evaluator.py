from __future__ import annotations

from statistics import mean
from time import perf_counter

from rag_eval.metrics import recall_at_k, reciprocal_rank
from rag_eval.pipeline import RAGPipeline


def evaluate(retriever, llm, qa: list[dict[str, object]], k: int = 3):
    pipeline = RAGPipeline(retriever, llm)
    rows: list[dict[str, object]] = []
    for item in qa:
        started = perf_counter()
        output = pipeline.answer(str(item["question"]), k)
        generation_ms = (perf_counter() - started) * 1000
        documents = output["documents"]
        judgment = llm.judge(
            str(item["question"]),
            str(output["answer"]),
            documents,
            item.get("reference_answer"),
        )
        rows.append(
            {
                "id": item["id"],
                "question": item["question"],
                "answer": output["answer"],
                "retrieved_doc_ids": [document["id"] for document in documents],
                "recall_at_k": recall_at_k(documents, set(item["relevant_doc_ids"])),
                "mrr": reciprocal_rank(documents, set(item["relevant_doc_ids"])),
                "answer_relevance": judgment["answer_relevance"],
                "faithfulness": judgment["faithfulness"],
                "correctness": judgment.get("correctness"),
                "judge_rationale": judgment["rationale"],
                "generation_latency_ms": generation_ms,
            }
        )
    correctness = [row["correctness"] for row in rows if row["correctness"] is not None]
    summary = {
        "recall_at_k": mean(float(row["recall_at_k"]) for row in rows),
        "mrr": mean(float(row["mrr"]) for row in rows),
        "answer_relevance": mean(float(row["answer_relevance"]) for row in rows),
        "faithfulness": mean(float(row["faithfulness"]) for row in rows),
        "correctness": mean(float(score) for score in correctness) if correctness else None,
        "generation_latency_ms": mean(float(row["generation_latency_ms"]) for row in rows),
        "questions": len(rows),
    }
    return {"summary": summary, "rows": rows}
