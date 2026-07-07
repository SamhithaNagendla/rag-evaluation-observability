from rag_eval.llm import BedrockLLMClient


class FakeBedrock:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []

    def converse(self, **kwargs):
        self.calls.append(kwargs)
        text = next(self.responses)
        return {"output": {"message": {"content": [{"text": text}]}}}


def test_bedrock_generation_and_judge_parsing() -> None:
    client = FakeBedrock(
        [
            "Raft uses a majority [d1].",
            'prefix {"answer_relevance":1.2,"faithfulness":0.8,'
            '"correctness":0.9,"rationale":"grounded"} suffix',
        ]
    )
    llm = BedrockLLMClient(region="x", model_id="m", client=client)
    docs = [{"id": "d1", "text": "Raft uses majority agreement."}]
    assert "[d1]" in llm.generate("How?", docs)
    judgment = llm.judge("How?", "answer", docs, "reference")
    assert judgment["answer_relevance"] == 1.0
    assert judgment["faithfulness"] == 0.8
    assert client.calls[0]["modelId"] == "m"
