from pathlib import Path

from rag_eval.store import EvaluationStore


def test_store_round_trip(tmp_path: Path) -> None:
    store = EvaluationStore(tmp_path / "runs.db")
    run_id = store.save("bm25", 3, {"mrr": 1.0}, [{"id": "q"}])
    assert store.list()[0]["summary"]["mrr"] == 1.0
    assert store.get(run_id)["rows"] == [{"id": "q"}]
    assert store.get("missing") is None
