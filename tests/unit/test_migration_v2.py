"""Schema v1 -> v2 migration: backup, rebuild, checks, rollback and restore. Opening a v1 store
runs v1 -> v2 and then v2 -> v3 (tests/unit/test_migration_v3.py covers the second step)."""
from __future__ import annotations

import shutil
import sqlite3
from pathlib import Path

import pytest

from sciai.graph.model import Layer, Node, NodeType, Status
from sciai.store import db as db_module
from sciai.store import migrate
from sciai.store.db import SCHEMA_VERSION, Database
from sciai.store.repository import Repository

V1_SCHEMA = (Path(__file__).resolve().parents[1] / "fixtures" / "schema_v1.sql").read_text(encoding="utf-8")
TABLES = ("sessions", "nodes", "session_links", "edges", "evidence", "tool_runs", "messages",
          "status_events", "notices", "meta")
NODE_COLS = ("id, lineage_id, version, session_id, layer, type, domain, title, status, created_at, updated_at")


def _node(conn, nid, type_="tool_result", domain="math", status="verified"):
    conn.execute(f"INSERT INTO nodes ({NODE_COLS}) VALUES (?,?,1,'s1','reasoning',?,?,?,?,1,1)",
                 (nid, nid, type_, domain, f"node {nid}", status))


def make_v1_store(path: Path) -> sqlite3.Connection:
    """A v1 store with rows in every table, an extra view and trigger, and the last rows still in
    the WAL file. Returns the open writer connection, which keeps those rows out of the main file."""
    conn = sqlite3.connect(path, isolation_level=None)
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA wal_autocheckpoint = 0")
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(V1_SCHEMA)
    conn.execute("INSERT INTO meta VALUES ('schema_version', '1')")
    conn.execute("INSERT INTO sessions VALUES ('s1', 'old session', 1, 1)")
    for i in range(1, 6):
        _node(conn, f"n{i}")
    _node(conn, "chem1", domain="chem", status="hypothesis")
    conn.executescript("""
        INSERT INTO session_links VALUES ('s1', 'n1', 'n1'), ('s1', 'n2', 'n2');
        INSERT INTO edges VALUES ('n2', 'n1', 'depends_on', '{}'), ('n3', 'n2', 'depends_on', '{}');
        INSERT INTO evidence VALUES ('e1', 'n2', 'n3', 'units', 'units.check_convert', '{}', 'pass', '{}', 1);
        INSERT INTO tool_runs VALUES ('r1', 's1', 'n2', 'sympy.diff', '{}', '{}', NULL, 1.0, 1);
        INSERT INTO messages VALUES ('m1', 's1', 'n2', 'user', 'hi', NULL, NULL, NULL, 0, 1);
        INSERT INTO status_events VALUES ('ev1', 'n2', NULL, 'verified', 'created', NULL, 1);
        INSERT INTO notices VALUES ('no1', 's1', 'n2', 'note', '{}', 0, 1);
        CREATE VIEW verified_nodes AS SELECT id, title FROM nodes WHERE status = 'verified';
        CREATE TRIGGER touch_session AFTER INSERT ON messages BEGIN
            UPDATE sessions SET updated_at = NEW.created_at WHERE id = NEW.session_id;
        END;
    """)
    return conn


def counts(conn: sqlite3.Connection) -> dict[str, int]:
    return {t: conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] for t in TABLES}


def schema_objects(conn: sqlite3.Connection) -> set[tuple[str, str]]:
    return {(r[0], r[1]) for r in conn.execute(
        "SELECT type, name FROM sqlite_master WHERE type IN ('index', 'trigger', 'view') AND sql IS NOT NULL")}


def nodes_sql(conn: sqlite3.Connection) -> str:
    return conn.execute("SELECT sql FROM sqlite_master WHERE name = 'nodes'").fetchone()[0]


@pytest.fixture
def v1(tmp_path):
    path = tmp_path / "knowledge.db"
    writer = make_v1_store(path)
    before = counts(writer)
    objects = schema_objects(writer)
    yield path, writer, before, objects
    writer.close()


def test_migration_keeps_every_row_and_object(v1):
    path, writer, before, objects = v1
    assert Path(str(path) + "-wal").stat().st_size > 0  # recent rows are only in the WAL file

    db = Database(path)
    conn = db._conn
    assert db.query_one("SELECT value FROM meta WHERE key='schema_version'")["value"] == str(SCHEMA_VERSION) == "4"
    assert counts(conn) == before
    # the three node indexes, the chem trigger, the extra view and trigger, plus the indexes
    # v3 and v4 add. Every object the v1 store had must still be here: that comparison is what
    # proves a migration did not quietly drop the chem rule's trigger.
    assert schema_objects(conn) == objects | {("index", "idx_datasets_name"),
                                              ("index", "idx_molecules_skeleton"),
                                              ("index", "idx_molecules_name")}
    assert "'assumption'" in nodes_sql(conn)
    assert conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    assert conn.execute("PRAGMA foreign_key_check").fetchall() == []
    assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1
    assert conn.execute("SELECT COUNT(*) FROM verified_nodes").fetchone()[0] == 5  # the view still works

    with pytest.raises(sqlite3.DatabaseError, match="chem"):
        conn.execute("UPDATE nodes SET status = 'verified' WHERE id = 'chem1'")
    with pytest.raises(sqlite3.IntegrityError):  # foreign keys are enforced again
        conn.execute("INSERT INTO edges VALUES ('n1', 'missing', 'depends_on', '{}')")

    repo = Repository(db)
    assert repo.get_node("n2").title == "node n2"
    a = Node(session_id="s1", layer=Layer.REASONING, type=NodeType.ASSUMPTION, title="no air resistance")
    repo.insert_node(a)
    assert repo.get_node(a.id).type == NodeType.ASSUMPTION
    db.close()


def test_backup_holds_the_rows_a_file_copy_would_miss(v1, tmp_path):
    path, writer, before, _ = v1
    naive = tmp_path / "naive-copy.db"
    shutil.copyfile(path, naive)  # what a plain file copy of knowledge.db gets
    try:
        naive_nodes = sqlite3.connect(naive).execute("SELECT COUNT(*) FROM nodes").fetchone()[0]
    except sqlite3.DatabaseError:
        naive_nodes = 0
    assert naive_nodes < before["nodes"]

    Database(path).close()
    backup = sqlite3.connect(tmp_path / "knowledge.db.v1.bak")
    assert counts(backup) == before
    assert backup.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()[0] == "1"
    assert "'assumption'" not in nodes_sql(backup)
    backup.close()


def test_a_failure_partway_rolls_back_to_a_working_v1(v1, monkeypatch, tmp_path):
    path, writer, before, objects = v1

    def boom(conn, statements):
        raise sqlite3.OperationalError("simulated failure while recreating indexes")

    monkeypatch.setattr(migrate, "recreate_saved", boom)
    with pytest.raises(RuntimeError, match="rolled back; the store is unchanged"):
        Database(path)

    check = sqlite3.connect(path, isolation_level=None)
    assert check.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()[0] == "1"
    assert counts(check) == before
    assert schema_objects(check) == objects
    assert "'assumption'" not in nodes_sql(check)
    assert check.execute("SELECT name FROM sqlite_master WHERE name = 'nodes_v2'").fetchone() is None
    _node(check, "n9")  # still a working v1 store
    check.close()

    monkeypatch.undo()
    db = Database(path)  # a later attempt succeeds, with a new backup name
    assert db.query_one("SELECT value FROM meta WHERE key='schema_version'")["value"] == str(SCHEMA_VERSION)
    assert len(list(tmp_path.glob("knowledge.db.v1*.bak"))) == 2
    db.close()


def test_integrity_failure_restores_the_backup(v1, monkeypatch):
    path, writer, before, _ = v1
    writer.close()
    real = db_module.migrate_v1_to_v2

    def corrupt_after(conn, db_path, schema):
        backup = real(conn, db_path, schema)
        raise migrate.IntegrityFailure("simulated: row 3 missing from index", backup)

    monkeypatch.setattr(db_module, "migrate_v1_to_v2", corrupt_after)
    with pytest.raises(RuntimeError, match="restored from"):
        Database(path)
    check = sqlite3.connect(path)
    assert check.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()[0] == "1"
    assert counts(check) == before
    check.close()


def test_new_store_is_current_and_newer_store_is_refused(tmp_path):
    db = Database(tmp_path / "fresh.db")
    assert db.query_one("SELECT value FROM meta WHERE key='schema_version'")["value"] == str(SCHEMA_VERSION)
    db.close()
    conn = sqlite3.connect(tmp_path / "fresh.db", isolation_level=None)
    conn.execute("UPDATE meta SET value = ?", (str(SCHEMA_VERSION + 1),))
    conn.close()
    with pytest.raises(RuntimeError, match="newer than this app"):
        Database(tmp_path / "fresh.db")


def test_migrated_table_matches_a_fresh_one(v1, tmp_path):
    path, writer, _, _ = v1
    migrated = Database(path)
    fresh = Database(tmp_path / "fresh.db")
    norm = lambda s: " ".join(s.replace('"nodes"', "nodes").replace("IF NOT EXISTS ", "").split())  # noqa: E731
    assert norm(nodes_sql(migrated._conn)) == norm(nodes_sql(fresh._conn))
    migrated.close()
    fresh.close()


def test_status_rules_unchanged_for_assumptions(tmp_path):
    db = Database(tmp_path / "a.db")
    repo = Repository(db)
    sid = repo.create_session("s")
    a = Node(session_id=sid, layer=Layer.REASONING, type=NodeType.ASSUMPTION, title="ideal wires",
             status=Status.FAILED)
    repo.insert_node(a)
    assert repo.get_node(a.id).status == Status.FAILED
    db.close()
