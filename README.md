# RAG Evaluation and Observability Harness

A tooling project for comparing retrieval strategies and evaluating generated answers. Unlike a
basic RAG chatbot, the focus is repeatable evaluation: BM25, TF-IDF, and hybrid retrieval are run on
a labeled Q&A set; Amazon Bedrock generates answers and acts as a structured judge for answer
relevance, faithfulness, and reference-answer correctness.

## What is real

- Bedrock `converse` integration for answer generation;
- a second Bedrock evaluation prompt returning structured numerical scores and rationale;
- deterministic retrieval metrics: Recall@K and MRR;
- reference answers in the evaluation set;
- request-level results and summaries persisted to SQLite;
- run-history API for comparing experiments;
- mock-injected unit tests that verify prompts and JSON parsing without making paid model calls.

## Run tests

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

## Run a live Bedrock evaluation

Configure AWS credentials with permission to invoke the selected model, then:

```bash
cp .env.example .env
uvicorn rag_eval.api:app --reload
curl -X POST http://localhost:8000/evaluate \
  -H 'Content-Type: application/json' \
  -d '{"strategy":"hybrid","k":3}'
```

The run is saved and can be retrieved through `/runs` and `/runs/{run_id}`.

## Important evaluation caveat

LLM-as-judge scores are useful comparative signals, not objective ground truth. The labeled retrieval
metrics remain deterministic, and production use should evaluate judge agreement against human
reviewers before relying on the scores for release decisions.
