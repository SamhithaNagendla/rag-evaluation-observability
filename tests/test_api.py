def test_evaluation_run_is_persisted(api_client) -> None:
    assert api_client.get("/health").json() == {"status": "healthy", "version": "0.2.0"}
    response = api_client.post("/evaluate", json={"strategy": "hybrid", "k": 2})
    assert response.status_code == 200
    body = response.json()
    assert body["summary"]["faithfulness"] == 0.9
    assert body["rows"][0]["judge_rationale"]

    runs = api_client.get("/runs").json()
    assert runs[0]["run_id"] == body["run_id"]
    assert api_client.get(f"/runs/{body['run_id']}").status_code == 200
    assert api_client.get("/runs/missing").status_code == 404
