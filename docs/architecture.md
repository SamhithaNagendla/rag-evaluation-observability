# Architecture

A retriever returns ranked context documents. A Bedrock client generates an answer constrained to
that context. The same client receives a separate evaluator prompt and returns structured answer
relevance, faithfulness, correctness, and rationale. Retrieval metrics and judge metrics are stored
with strategy, K, latency, retrieved document IDs, generated answer, and run metadata in SQLite.
