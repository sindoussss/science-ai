"""SQLite connection management.

One connection, guarded by a re-entrant lock, shared by the controller thread
and the UI thread. WAL mode keeps readers from blocking the writer, and every
write happens inside ``Database.tx()`` so a crash never leaves half a cascade.
"""
from __future__ import annotations

import shutil
import sqlite3
import threading
from contextlib import contextmanager
from importlib import resources
from pathlib import Path
from typing import Iterator

from sciai.store.migrate import IntegrityFailure, migrate_v1_to_v2

SCHEMA_VERSION = 2


class Database:
    def __init__(self, path: str | Path) -> None:
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False, isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        if self.path != ":memory:":
            self._conn.execute("PRAGMA journal_mode = WAL")
        self._conn.execute("PRAGMA synchronous = NORMAL")
        self._depth = 0
        self._migrate()

    def _stored_version(self) -> int | None:
        try:
            row = self._conn.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()
        except sqlite3.OperationalError:  # a new, empty store has no meta table yet
            return None
        return None if row is None else int(row["value"])

    def _migrate(self) -> None:
        schema = resources.files("sciai.store").joinpath("schema.sql").read_text(encoding="utf-8")
        with self._lock:
            version = self._stored_version()
            if version is not None and version > SCHEMA_VERSION:
                raise RuntimeError(f"database schema v{version} is newer than this app (v{SCHEMA_VERSION})")
            if version == 1:
                self._upgrade_v1(schema)
            self._conn.executescript(schema)
            if version is None:
                self._conn.execute(
                    "INSERT INTO meta(key, value) VALUES ('schema_version', ?)", (str(SCHEMA_VERSION),)
                )

    def _upgrade_v1(self, schema: str) -> None:
        try:
            migrate_v1_to_v2(self._conn, self.path, schema)
        except IntegrityFailure as exc:
            self._conn.close()
            if exc.backup is not None:
                shutil.copyfile(exc.backup, self.path)
                for suffix in ("-wal", "-shm"):
                    Path(self.path + suffix).unlink(missing_ok=True)
            raise RuntimeError(
                f"the knowledge store at {self.path} failed its integrity check after the upgrade to "
                f"schema v2 ({exc}). It was restored from {exc.backup}, which is still a v1 store."
            ) from exc
        except Exception as exc:
            self._conn.close()
            raise RuntimeError(
                f"upgrading the knowledge store at {self.path} to schema v2 failed and was rolled back; "
                f"the store is unchanged ({type(exc).__name__}: {exc})"
            ) from exc

    @contextmanager
    def tx(self) -> Iterator[sqlite3.Connection]:
        """Transaction that nests: only the outermost level commits or rolls back."""
        with self._lock:
            outer = self._depth == 0
            if outer:
                self._conn.execute("BEGIN IMMEDIATE")
            self._depth += 1
            try:
                yield self._conn
            except BaseException:
                self._depth -= 1
                if outer:
                    self._conn.execute("ROLLBACK")
                raise
            else:
                self._depth -= 1
                if outer:
                    self._conn.execute("COMMIT")

    def query(self, sql: str, params: tuple | list = ()) -> list[sqlite3.Row]:
        with self._lock:
            return self._conn.execute(sql, params).fetchall()

    def query_one(self, sql: str, params: tuple | list = ()) -> sqlite3.Row | None:
        with self._lock:
            return self._conn.execute(sql, params).fetchone()

    def close(self) -> None:
        with self._lock:
            self._conn.close()
