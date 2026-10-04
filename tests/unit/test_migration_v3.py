"""Schema v2 -> v3 migration (the datasets table) and the dataset records it holds."""
from __future__ import annotations

import sqlite3
import time
from pathlib import Path

import pytest

from sciai.store import db as db_module
from sciai.store import migrate
from sciai.store.db import Database
from sciai.store.repository import DatasetRecord, Repository

TABLES = ("sessions", "nodes", "session_links", "edges", "evidence", "tool_runs", "messages",
          "status_events", "notices", "meta")


def make_v2_store(path: Path) -> None:
    """A v2 store: today's schema without the datasets table, with rows and an extra view."""
    Database(path).close()
    conn = sqlite3.connect(path, isolation_level=None)
    conn.executescript("""
        DROP INDEX idx_datasets_name;
        DROP TABLE datasets;
        UPDATE meta SET value = '2' WHERE key = 'schema_version';
        INSERT INTO sessions VALUES ('s1', 'old session', 1, 1);
        INSERT INTO nodes (id, lineage_id, version, session_id, layer, type, domain, title, status, created_at,
                           updated_at)
            VALUES ('n1', 'n1', 1, 's1', 'reasoning', 'tool_result', 'math', 'node n1', 'verified', 1, 1),
                   ('n2', 'n2', 1, 's1', 'reasoning', 'tool_result', 'math', 'node n2', 'proposed', 1, 1);
        INSERT INTO edges VALUES ('n2', 'n1', 'depends_on', '{}');
        CREATE VIEW verified_nodes AS SELECT id FROM nodes WHERE status = 'verified';
    """)
    conn.close()


def counts(conn: sqlite3.Connection) -> dict[str, int]:
    return {t: conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] for t in TABLES}


def version(path: Path) -> str:
    conn = sqlite3.connect(path)
    try:
        return conn.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()[0]
    finally:
        conn.close()


def has_datasets(path: Path) -> bool:
    conn = sqlite3.connect(path)
    try:
        return conn.execute("SELECT 1 FROM sqlite_master WHERE name = 'datasets'").fetchone() is not None
    finally:
        conn.close()


@pytest.fixture
def v2(tmp_path):
    path = tmp_path / "knowledge.db"
    make_v2_store(path)
    conn = sqlite3.connect(path)
    before = counts(conn)
    conn.close()
    return path, before


def test_v2_store_gains_datasets_and_keeps_everything(v2, tmp_path):
    path, before = v2
    db = Database(path)
    assert db.query_one("SELECT value FROM meta WHERE key='schema_version'")["value"] == "3"
    assert counts(db._conn) == before
    assert db._conn.execute("SELECT COUNT(*) FROM verified_nodes").fetchone()[0] == 1
    assert db._conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    assert db._conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1
    db.close()
    assert has_datasets(path)
    backup = tmp_path / "knowledge.db.v2.bak"
    assert version(backup) == "2" and not has_datasets(backup)


def test_a_failed_v3_step_leaves_a_working_v2_store(v2, monkeypatch):
    path, before = v2
    real = migrate.v3_statements
    monkeypatch.setattr(migrate, "v3_statements", lambda schema: [*real(schema), "CREATE TABLE broken ("])
    with pytest.raises(RuntimeError, match="rolled back; the store is unchanged by that step and still works as a v2"):
        Database(path)
    assert version(path) == "2" and not has_datasets(path)
    conn = sqlite3.connect(path)
    assert counts(conn) == before
    conn.close()
    monkeypatch.undo()
    Database(path).close()  # a later attempt succeeds
    assert version(path) == "3"


def test_integrity_failure_restores_the_v2_backup(v2, monkeypatch):
    path, before = v2
    real = db_module.migrate_v2_to_v3

    def corrupt_after(conn, db_path, schema):
        backup = real(conn, db_path, schema)
        raise migrate.IntegrityFailure("simulated: page 4 is never used", backup)

    monkeypatch.setattr(db_module, "migrate_v2_to_v3", corrupt_after)
    with pytest.raises(RuntimeError, match="still a v2 store"):
        Database(path)
    assert version(path) == "2" and not has_datasets(path)


def _rec(sha: str, name: str, imported_at: float, rows: int = 3) -> DatasetRecord:
    return DatasetRecord(id=f"id-{sha}", sha256=sha * 64, name=name, format="csv", options={"delimiter": ","},
                         stored_path=f"/data/{sha}.csv", rows=rows, columns=2,
                         schema=[{"name": "x", "type": "number"}, {"name": "g", "type": "text", "levels": ["a", "b"]}],
                         original_path=f"/home/{name}", imported_at=imported_at)


def test_dataset_records(tmp_path):
    db = Database(tmp_path / "k.db")
    repo = Repository(db)
    t = time.time()
    repo.add_dataset(_rec("a", "trial.csv", t))
    repo.add_dataset(_rec("b", "Trial.csv", t + 1, rows=4))   # the file changed: a new dataset, same name
    repo.add_dataset(_rec("c", "growth.xlsx", t + 2))
    assert repo.dataset("id-a").rows == 3
    assert repo.dataset_by_name("TRIAL.CSV").id == "id-b"     # the latest import of that name
    assert [r.id for r in repo.current_datasets()] == ["id-c", "id-b"]

    repo.add_dataset(_rec("a", "trial.csv", t + 3))           # the old bytes imported again
    assert repo.dataset_by_name("trial.csv").id == "id-a"
    assert db.query_one("SELECT COUNT(*) AS n FROM datasets")["n"] == 3  # no duplicate row

    assert [r.id for r in repo.datasets_named_in("compare scores in trial.csv by group")] == ["id-a"]
    assert [r.id for r in repo.datasets_named_in("Is growth higher in the treated trial?")] == ["id-a", "id-c"]
    assert [r.id for r in repo.datasets_named_in("load /home/me/trial.csv.")] == ["id-a"]
    assert repo.datasets_named_in("trials and growths") == []
    assert repo.datasets_named_in("my-trial.csv.bak") == []
    text = repo.dataset("id-a").schema_text()
    assert text == "trial.csv: 3 rows; columns: x (number), g (text; levels a, b)"
    db.close()
