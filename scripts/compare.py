import json
from pathlib import Path

from rag_eval.config import get_settings
from rag_eval.evaluator import evaluate
from rag_eval.llm import BedrockLLMClient
from rag_eval.retrieval import BM25Retriever, HybridRetriever, TfidfRetriever

settings = get_settings()
documents = json.loads(settings.documents_path.read_text(encoding="utf-8"))
qa = json.loads(settings.qa_path.read_text(encoding="utf-8"))
llm = BedrockLLMClient(
    region=settings.aws_region,
    model_id=settings.bedrock_model_id,
    max_tokens=settings.max_tokens,
    temperature=settings.temperature,
)
retrievers = {
    "bm25": BM25Retriever,
    "tfidf": TfidfRetriever,
    "hybrid": HybridRetriever,
}
results = {
    name: evaluate(retriever(documents), llm, qa)
    for name, retriever in retrievers.items()
}
target = Path("benchmark-results")
target.mkdir(exist_ok=True)
(target / "comparison.json").write_text(
    json.dumps(results, indent=2),
    encoding="utf-8",
)
for name, result in results.items():
    print(name, result["summary"])
