from __future__ import annotations

import json
from typing import Protocol


class LLMClient(Protocol):
    def generate(self, question: str, documents: list[dict[str, object]]) -> str: ...

    def judge(
        self,
        question: str,
        answer: str,
        documents: list[dict[str, object]],
        reference_answer: str | None,
    ) -> dict[str, object]: ...


class BedrockLLMClient:
    def __init__(
        self,
        *,
        region: str,
        model_id: str,
        max_tokens: int = 500,
        temperature: float = 0.0,
        client=None,
    ):
        if client is None:
            import boto3

            client = boto3.client("bedrock-runtime", region_name=region)
        self.client = client
        self.model_id = model_id
        self.max_tokens = max_tokens
        self.temperature = temperature

    def _converse(self, prompt: str) -> str:
        response = self.client.converse(
            modelId=self.model_id,
            messages=[{"role": "user", "content": [{"text": prompt}]}],
            inferenceConfig={
                "maxTokens": self.max_tokens,
                "temperature": self.temperature,
            },
        )
        return response["output"]["message"]["content"][0]["text"]

    def generate(self, question: str, documents: list[dict[str, object]]) -> str:
        context = "\n".join(f"[{doc['id']}] {doc['text']}" for doc in documents)
        prompt = (
            "Answer the question using only the supplied context. "
            "Cite supporting document IDs in square brackets. If the context is insufficient, say so.\n\n"
            f"Question: {question}\n\nContext:\n{context}"
        )
        return self._converse(prompt).strip()

    def judge(
        self,
        question: str,
        answer: str,
        documents: list[dict[str, object]],
        reference_answer: str | None,
    ) -> dict[str, object]:
        context = "\n".join(f"[{doc['id']}] {doc['text']}" for doc in documents)
        prompt = f"""
You are evaluating a retrieval-augmented answer. Return only valid JSON with this schema:
{{"answer_relevance": number, "faithfulness": number, "correctness": number|null, "rationale": string}}
All numeric scores must be between 0 and 1.
- answer_relevance: how directly the answer addresses the question.
- faithfulness: how completely the answer is supported by the supplied context, with no unsupported claims.
- correctness: agreement with the reference answer when one is supplied; otherwise null.

Question: {question}
Answer: {answer}
Reference answer: {reference_answer or 'NOT PROVIDED'}
Context:
{context}
""".strip()
        raw = self._converse(prompt)
        start = raw.find("{")
        end = raw.rfind("}")
        if start < 0 or end < start:
            raise ValueError("judge did not return JSON")
        result = json.loads(raw[start : end + 1])
        for name in ("answer_relevance", "faithfulness"):
            result[name] = max(0.0, min(1.0, float(result[name])))
        if result.get("correctness") is not None:
            result["correctness"] = max(0.0, min(1.0, float(result["correctness"])))
        result["rationale"] = str(result.get("rationale", ""))
        return result
