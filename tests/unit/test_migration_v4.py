"""The v3 to v4 migration, and the molecules table it adds.

The point that matters beyond "the table exists": the chem rule lives in the nodes table's
CHECK and in the chem_never_verified trigger, and a migration is exactly the moment those could
be dropped by accident. So the last test here opens a store that was migrated from v3 and
tries, four ways, to get a verified chem row into it.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from sciai.store.db import SCHEMA_VERSION, Database
from sciai.store.migrate import MigrationError, v4_statements
from sciai.store.repository import MoleculeRecord, Repository

SCHEMA = (Path(__file__).resolve().parents[2] / "src" / "sciai" / "store" / "schema.sql")


def v3_store(path: Path) -> None:
    """A store at v3: the current schema minus the molecules table, stamped as version 3."""
    sql = SCHEMA.read_text(encoding="utf-8")
    for statement in v4_statements(sql):
        sql = sql.replace(statement, "")
    conn = sqlite3.connect(path, isolation_level=None)
    conn.executescript(sql)
    conn.execute("INSERT INTO meta(key, value) VALUES ('schema_version', '3')")
    conn.execute("INSERT INTO sessions (id, title, created_at, updated_at) "
                 "VALUES ('s1', 'kept', 0, 0)")
    conn.execute("INSERT INTO nodes (id, lineage_id, version, session_id, layer, type, domain, "
                 "title, status, created_at, updated_at) VALUES "
                 "('n1','n1',1,'s1','reasoning','claim','math','kept','verified',0,0)")
    conn.close()


def test_a_v3_store_migrates_and_keeps_its_rows(tmp_path: Path) -> None:
    path = tmp_path / "k.db"
    v3_store(path)
    db = Database(path)
    try:
        assert db.query_one("SELECT value FROM meta WHERE key='schema_version'")["value"] == \
            str(SCHEMA_VERSION) == "4"
        assert db.query_one("SELECT title FROM nodes WHERE id='n1'")["title"] == "kept"
        assert db.query_one("SELECT count(*) c FROM molecules")["c"] == 0
    finally:
        db.close()


def test_the_migration_leaves_a_backup(tmp_path: Path) -> None:
    path = tmp_path / "k.db"
    v3_store(path)
    db = Database(path)
    db.close()
    backups = list(tmp_path.glob("k.db.v3.bak"))
    assert backups, sorted(p.name for p in tmp_path.iterdir())
    # the backup is a working v3 store, not a truncated file
    conn = sqlite3.connect(backups[0], isolation_level=None)
    try:
        assert conn.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()[0] == "3"
        assert conn.execute("SELECT title FROM nodes WHERE id='n1'").fetchone()[0] == "kept"
        assert conn.execute("SELECT name FROM sqlite_master WHERE name='molecules'").fetchone() is None
    finally:
        conn.close()


def test_a_v3_store_still_has_its_indexes_and_triggers(tmp_path: Path) -> None:
    path = tmp_path / "k.db"
    v3_store(path)
    before = _objects(sqlite3.connect(path))
    db = Database(path)
    try:
        after = _objects(db._conn)
        assert before - after == set(), before - after
        assert ("trigger", "chem_never_verified") in after
    finally:
        db.close()


def test_the_schema_must_define_the_molecules_table() -> None:
    with pytest.raises(MigrationError, match="no molecules table"):
        v4_statements("CREATE TABLE nodes (id TEXT);")


# ----------------------------------------------------------------- the table itself

def rec(**kw) -> MoleculeRecord:
    base = dict(inchikey="BSYNRYMUTXBXSQ-UHFFFAOYSA-N", skeleton="BSYNRYMUTXBXSQ",
                canonical_smiles="CC(=O)Oc1ccccc1C(=O)O", inchi="InChI=1S/C9H8O4/...",
                formula="C9H8O4", atoms=13, bonds=13, charge=0, input_form="aspirin",
                standardized=[], screen={"action": "allow"})
    return MoleculeRecord(**{**base, **kw})


@pytest.fixture
def repo(tmp_path):
    db = Database(tmp_path / "t.db")
    yield Repository(db)
    db.close()


def test_a_molecule_round_trips(repo) -> None:
    repo.add_molecule(rec(name="aspirin"))
    got = repo.molecule("BSYNRYMUTXBXSQ-UHFFFAOYSA-N")
    assert got is not None
    assert got.formula == "C9H8O4" and got.name == "aspirin"
    assert "InChIKey BSYNRYMUTXBXSQ" in got.summary()


def test_the_same_structure_twice_is_one_row(repo) -> None:
    repo.add_molecule(rec(name="aspirin", imported_at=1.0))
    repo.add_molecule(rec(imported_at=2.0))
    assert len(repo.recent_molecules()) == 1
    kept = repo.molecule("BSYNRYMUTXBXSQ-UHFFFAOYSA-N")
    assert kept.name == "aspirin"          # a later import without a name keeps the old one
    assert kept.imported_at == 2.0


def test_stereoisomers_are_separate_rows_under_one_skeleton(repo) -> None:
    repo.add_molecule(rec(inchikey="QNAYBMKLOCPYGJ-REOHCLBHSA-N", skeleton="QNAYBMKLOCPYGJ",
                          name="L-alanine", canonical_smiles="C[C@@H](N)C(=O)O"))
    repo.add_molecule(rec(inchikey="QNAYBMKLOCPYGJ-UWTATZPHSA-N", skeleton="QNAYBMKLOCPYGJ",
                          name="D-alanine", canonical_smiles="C[C@H](N)C(=O)O"))
    both = repo.molecules_with_skeleton("QNAYBMKLOCPYGJ")
    assert [m.name for m in both] == ["L-alanine", "D-alanine"]
    assert repo.molecules_with_skeleton("QNAYBMKLOCPYGJ-REOHCLBHSA-N") == both


def test_lookup_by_name_is_case_insensitive_and_takes_the_latest(repo) -> None:
    repo.add_molecule(rec(name="Aspirin", imported_at=1.0))
    repo.add_molecule(rec(inchikey="OTHER-KEY-N", skeleton="OTHER", name="aspirin",
                          imported_at=5.0))
    assert repo.molecule_by_name("ASPIRIN").inchikey == "OTHER-KEY-N"


def test_the_screen_verdict_and_the_steps_are_stored(repo) -> None:
    repo.add_molecule(rec(standardized=["kept the largest of 2 fragments", "neutralized"],
                          screen={"action": "flag", "rule": "nitrogen_mustard"}))
    got = repo.molecule("BSYNRYMUTXBXSQ-UHFFFAOYSA-N")
    assert got.standardized == ["kept the largest of 2 fragments", "neutralized"]
    assert got.screen["rule"] == "nitrogen_mustard"
    assert "standardized: kept the largest" in got.summary()


def test_delete_removes_the_row(repo) -> None:
    repo.add_molecule(rec())
    repo.delete_molecule("BSYNRYMUTXBXSQ-UHFFFAOYSA-N")
    assert repo.molecule("BSYNRYMUTXBXSQ-UHFFFAOYSA-N") is None


def test_an_empty_molecule_is_refused_by_the_table(repo) -> None:
    with pytest.raises(sqlite3.IntegrityError):
        repo.add_molecule(rec(atoms=0))


def test_the_molecules_table_has_no_status_column(repo) -> None:
    """A molecule record is bookkeeping; a claim about a molecule is a node."""
    columns = {r["name"] for r in repo.db.query("PRAGMA table_info(molecules)")}
    assert "status" not in columns
    assert {"inchikey", "skeleton", "screen", "standardized"} <= columns


# ----------------------------------------------------------------- the rule survives v4

def test_a_migrated_store_still_refuses_a_verified_chem_node(tmp_path: Path) -> None:
    path = tmp_path / "k.db"
    v3_store(path)
    db = Database(path)
    try:
        assert db.query_one("SELECT value FROM meta WHERE key='schema_version'")["value"] == "4"
        # 1. a fresh insert
        with pytest.raises(sqlite3.IntegrityError):
            with db.tx() as c:
                c.execute("INSERT INTO nodes (id, lineage_id, version, session_id, layer, type, "
                          "domain, title, status, created_at, updated_at) VALUES "
                          "('c1','c1',1,'s1','domain','tool_result','chem','m','verified',0,0)")
        # 2. an update to verified on an existing chem row
        with db.tx() as c:
            c.execute("INSERT INTO nodes (id, lineage_id, version, session_id, layer, type, "
                      "domain, title, status, created_at, updated_at) VALUES "
                      "('c2','c2',1,'s1','domain','tool_result','chem','m','hypothesis',0,0)")
        with pytest.raises(sqlite3.IntegrityError):
            with db.tx() as c:
                c.execute("UPDATE nodes SET status='verified' WHERE id='c2'")
        # 3. flipping a verified non-chem row into the chem domain
        with pytest.raises(sqlite3.IntegrityError):
            with db.tx() as c:
                c.execute("UPDATE nodes SET domain='chem' WHERE id='n1'")
        # 4. both at once
        with pytest.raises(sqlite3.IntegrityError):
            with db.tx() as c:
                c.execute("UPDATE nodes SET domain='chem', status='verified' WHERE id='c2'")
        assert db.query_one("SELECT status FROM nodes WHERE id='c2'")["status"] == "hypothesis"
    finally:
        db.close()


def test_the_trigger_message_is_still_the_explaining_one(tmp_path: Path) -> None:
    path = tmp_path / "k.db"
    v3_store(path)
    db = Database(path)
    try:
        sql = db.query_one("SELECT sql FROM sqlite_master WHERE name='chem_never_verified'")["sql"]
        assert "can never be verified" in sql
    finally:
        db.close()


def _objects(conn: sqlite3.Connection) -> set[tuple[str, str]]:
    rows = conn.execute("SELECT type, name FROM sqlite_master WHERE type IN "
                        "('index','trigger','view') AND sql IS NOT NULL").fetchall()
    out = {(r[0], r[1]) for r in rows}
    conn.close()
    return out


def test_the_record_serializes_its_json_fields_deterministically(repo) -> None:
    repo.add_molecule(rec(screen={"b": 2, "a": 1}))
    raw = repo.db.query_one("SELECT screen FROM molecules")["screen"]
    assert raw == json.dumps({"a": 1, "b": 2}, sort_keys=True)
