from __future__ import annotations

import math
import re
from collections import Counter

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


def tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


class BM25Retriever:
    def __init__(self, documents: list[dict[str, object]]):
        self.documents = documents
        self.tokenized = [tokens(str(document["text"])) for document in documents]
        self.average_length = sum(map(len, self.tokenized)) / max(len(documents), 1)
        self.document_frequency = Counter(
            token for document_tokens in self.tokenized for token in set(document_tokens)
        )

    def search(self, query: str, k: int = 3):
        query_tokens = tokens(query)
        scored: list[tuple[float, int]] = []
        document_count = len(self.documents)
        for index, document_tokens in enumerate(self.tokenized):
            term_frequency = Counter(document_tokens)
            score = 0.0
            for token in query_tokens:
                frequency = term_frequency[token]
                if not frequency:
                    continue
                inverse_document_frequency = math.log(
                    1
                    + (
                        document_count
                        - self.document_frequency[token]
                        + 0.5
                    )
                    / (self.document_frequency[token] + 0.5)
                )
                denominator = frequency + 1.5 * (
                    1 - 0.25 + 0.25 * len(document_tokens) / self.average_length
                )
                score += inverse_document_frequency * (frequency * 2.5) / denominator
            scored.append((score, index))
        return [
            (self.documents[index], score)
            for score, index in sorted(scored, reverse=True)[:k]
        ]


class TfidfRetriever:
    def __init__(self, documents: list[dict[str, object]]):
        self.documents = documents
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.matrix = self.vectorizer.fit_transform(
            str(document["text"]) for document in documents
        )

    def search(self, query: str, k: int = 3):
        scores = (
            self.matrix @ self.vectorizer.transform([query]).T
        ).toarray().ravel()
        indices = np.argsort(scores)[::-1][:k]
        return [
            (self.documents[int(index)], float(scores[index]))
            for index in indices
        ]


class HybridRetriever:
    def __init__(self, documents: list[dict[str, object]]):
        self.bm25 = BM25Retriever(documents)
        self.tfidf = TfidfRetriever(documents)

    def search(self, query: str, k: int = 3):
        combined: dict[str, float] = {}
        for retriever in (self.bm25, self.tfidf):
            rows = retriever.search(query, len(self.bm25.documents))
            maximum = max((score for _, score in rows), default=1.0) or 1.0
            for document, score in rows:
                document_id = str(document["id"])
                combined[document_id] = combined.get(document_id, 0.0) + score / maximum
        documents_by_id = {
            str(document["id"]): document for document in self.bm25.documents
        }
        ranked = sorted(combined.items(), key=lambda item: item[1], reverse=True)[:k]
        return [(documents_by_id[document_id], score) for document_id, score in ranked]
