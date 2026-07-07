from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

SCHEMA = """
CREATE TABLE IF NOT EXISTS evaluation_runs (
    run_id TEXT PRIMARY KEY,
    strategy TEXT NOT NULL,
    k INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    summary_json TEXT NOT NULL,
    rows_json TEXT NOT NULL
)
"""


class EvaluationStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute(SCHEMA)
            connection.commit()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        try:
            yield connection
        finally:
            connection.close()

    def save(
        self,
        strategy: str,
        k: int,
        summary: dict[str, object],
        rows: list[dict[str, object]],
    ) -> str:
        run_id = uuid4().hex
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO evaluation_runs VALUES (?,?,?,?,?,?)",
                (
                    run_id,
                    strategy,
                    k,
                    datetime.now(UTC).isoformat(),
                    json.dumps(summary, sort_keys=True),
                    json.dumps(rows, sort_keys=True),
                ),
            )
            connection.commit()
        return run_id

    def list(self, limit: int = 20) -> list[dict[str, object]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT run_id,strategy,k,created_at,summary_json FROM evaluation_runs "
                "ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [
            {
                "run_id": row["run_id"],
                "strategy": row["strategy"],
                "k": row["k"],
                "created_at": row["created_at"],
                "summary": json.loads(row["summary_json"]),
            }
            for row in rows
        ]

    def get(self, run_id: str) -> dict[str, object] | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM evaluation_runs WHERE run_id=?", (run_id,)
            ).fetchone()
        if row is None:
            return None
        return {
            "run_id": row["run_id"],
            "strategy": row["strategy"],
            "k": row["k"],
            "created_at": row["created_at"],
            "summary": json.loads(row["summary_json"]),
            "rows": json.loads(row["rows_json"]),
        }
