from __future__ import annotations


def recall_at_k(retrieved: list[dict[str, object]], relevant: set[str]) -> float:
    return float(any(str(document["id"]) in relevant for document in retrieved))


def reciprocal_rank(retrieved: list[dict[str, object]], relevant: set[str]) -> float:
    for rank, document in enumerate(retrieved, 1):
        if str(document["id"]) in relevant:
            return 1 / rank
    return 0.0
