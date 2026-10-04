"""Schema migrations, run when an older knowledge store is opened.

v1 -> v2 adds 'assumption' to the CHECK on ``nodes.type``. SQLite can't alter a
CHECK constraint, so the table is rebuilt, following SQLite's documented
procedure (https://www.sqlite.org/lang_altertable.html#otheralter):

1. Back up with ``VACUUM INTO``. The store runs in WAL mode, so a plain file copy
   can miss committed changes that are still in ``knowledge.db-wal``.
2. Turn foreign keys off *outside* the transaction (SQLite ignores the pragma
   inside one). Deferring them is not enough: with foreign keys on, ``DROP TABLE
   nodes`` first deletes every node, which edges and evidence still reference,
   so the commit fails every time.
3. In one transaction: save every index and trigger on ``nodes`` and every view
   and trigger in the store (read from ``sqlite_master``, not a fixed list), drop
   the views and triggers, rebuild the table, then recreate everything saved.
4. ``PRAGMA foreign_key_check`` must come back empty, or the transaction rolls back.
5. Commit, turn foreign keys back on, then ``PRAGMA integrity_check`` must say ok.
"""
from __future__ import annotations

import re
import sqlite3
import time
from pathlib import Path


class MigrationError(RuntimeError):
    """The migration rolled back; the store is unchanged."""


class IntegrityFailure(RuntimeError):
    """The migration committed but the store failed its integrity check."""

    def __init__(self, message: str, backup: Path | None) -> None:
        super().__init__(message)
        self.backup = backup


_NODES_DDL = re.compile(r"CREATE TABLE IF NOT EXISTS nodes \((.*?)\n\);", re.DOTALL)


def nodes_table_body(schema_sql: str) -> str:
    m = _NODES_DDL.search(schema_sql)
    if m is None:
        raise MigrationError("schema.sql has no nodes table definition")
    return m.group(1)


def backup_path(db_path: Path) -> Path:
    """knowledge.db.v1.bak, or a timestamped name if an earlier attempt left one behind."""
    path = db_path.with_name(db_path.name + ".v1.bak")
    if path.exists():
        path = db_path.with_name(f"{db_path.name}.v1.{time.strftime('%Y%m%d-%H%M%S')}.bak")
    n = 1
    while path.exists():
        path = db_path.with_name(f"{db_path.name}.v1.{time.strftime('%Y%m%d-%H%M%S')}-{n}.bak")
        n += 1
    return path


def _columns(conn: sqlite3.Connection, table: str) -> list[str]:
    return [r[1] for r in conn.execute(f"PRAGMA table_info({table})")]


def _rebuild_nodes(conn: sqlite3.Connection, schema_sql: str) -> None:
    saved = conn.execute(
        "SELECT type, name, sql FROM sqlite_master WHERE sql IS NOT NULL AND "
        "(type IN ('view', 'trigger') OR (type = 'index' AND tbl_name = 'nodes')) "
        "ORDER BY CASE type WHEN 'index' THEN 0 WHEN 'view' THEN 1 ELSE 2 END, name"
    ).fetchall()
    for kind, name, _ in saved:
        if kind in ("view", "trigger"):
            conn.execute(f'DROP {kind.upper()} "{name}"')
    before = conn.execute("SELECT COUNT(*) FROM nodes").fetchone()[0]
    conn.execute(f"CREATE TABLE nodes_v2 ({nodes_table_body(schema_sql)}\n)")
    old = set(_columns(conn, "nodes"))
    cols = ", ".join(c for c in _columns(conn, "nodes_v2") if c in old)
    conn.execute(f"INSERT INTO nodes_v2 ({cols}) SELECT {cols} FROM nodes")
    after = conn.execute("SELECT COUNT(*) FROM nodes_v2").fetchone()[0]
    if after != before:
        raise MigrationError(f"copied {after} of {before} nodes")
    conn.execute("DROP TABLE nodes")
    conn.execute("ALTER TABLE nodes_v2 RENAME TO nodes")
    recreate_saved(conn, [sql for _, _, sql in saved])


def recreate_saved(conn: sqlite3.Connection, statements: list[str]) -> None:
    for sql in statements:
        conn.execute(sql)


def migrate_v1_to_v2(conn: sqlite3.Connection, db_path: str, schema_sql: str) -> Path | None:
    """Returns the backup path (None for an in-memory store). ``conn`` must be in autocommit mode."""
    backup = None
    if db_path != ":memory:":
        backup = backup_path(Path(db_path))
        conn.execute("VACUUM INTO ?", (str(backup),))
    conn.execute("PRAGMA foreign_keys = OFF")
    try:
        conn.execute("BEGIN IMMEDIATE")
        try:
            _rebuild_nodes(conn, schema_sql)
            problems = conn.execute("PRAGMA foreign_key_check").fetchall()
            if problems:
                raise MigrationError(f"foreign_key_check found {len(problems)} broken reference(s), "
                                     f"first: {tuple(problems[0])}")
            conn.execute("UPDATE meta SET value = '2' WHERE key = 'schema_version'")
            conn.execute("COMMIT")
        except BaseException:
            conn.execute("ROLLBACK")
            raise
    finally:
        conn.execute("PRAGMA foreign_keys = ON")
    result = [r[0] for r in conn.execute("PRAGMA integrity_check").fetchall()]
    if result != ["ok"]:
        raise IntegrityFailure(f"integrity_check after the v2 migration: {'; '.join(result[:5])}", backup)
    return backup
