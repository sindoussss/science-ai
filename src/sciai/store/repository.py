"""Persistence for graph objects. The GraphEngine is the only caller that writes."""
from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from sciai.graph.model import (
    CASCADE_KINDS,
    Domain,
    Edge,
    EdgeKind,
    Evidence,
    LadderStage,
    Layer,
    Node,
    NodeType,
    Status,
    new_id,
    now,
)
from sciai.store.db import Database

_NODE_COLS = (
    "id, lineage_id, version, session_id, layer, type, domain, title, content, content_canonical, "
    "fingerprint, tool_name, tool_inputs, result, status, locked, needs_recheck, confidence, risk, "
    "flags, ladder_stage, role, superseded_by, created_at, updated_at"
)


def _dumps(value: Any) -> str | None:
    return None if value is None else json.dumps(value, sort_keys=True, default=str)


def _loads(value: str | None) -> Any:
    return None if value is None else json.loads(value)


def node_from_row(row: sqlite3.Row) -> Node:
    return Node(
        id=row["id"],
        lineage_id=row["lineage_id"],
        version=row["version"],
        session_id=row["session_id"],
        layer=Layer(row["layer"]),
        type=NodeType(row["type"]),
        domain=Domain(row["domain"]),
        title=row["title"],
        content=row["content"],
        content_canonical=row["content_canonical"],
        fingerprint=row["fingerprint"],
        tool_name=row["tool_name"],
        tool_inputs=_loads(row["tool_inputs"]),
        result=_loads(row["result"]),
        status=Status(row["status"]),
        locked=bool(row["locked"]),
        needs_recheck=bool(row["needs_recheck"]),
        confidence=row["confidence"],
        risk=_loads(row["risk"]) or {},
        flags=_loads(row["flags"]) or [],
        ladder_stage=LadderStage(row["ladder_stage"]),
        role=row["role"],
        superseded_by=row["superseded_by"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


@dataclass
class DatasetRecord:
    """An imported data file (see the datasets table in schema.sql)."""

    id: str
    sha256: str
    name: str
    format: str
    options: dict[str, Any]
    stored_path: str
    rows: int
    columns: int
    schema: list[dict[str, Any]]
    original_path: str | None = None
    imported_at: float = field(default_factory=now)

    def schema_text(self, max_levels: int = 6) -> str:
        """What the model sees: names, types, units, missing counts and the levels of small
        categorical columns. Never row values."""
        parts = []
        for col in self.schema:
            bits = [col["type"]]
            if col.get("unit"):
                bits.append(f"unit {col['unit']}")
            if col.get("missing"):
                bits.append(f"{col['missing']} missing")
            levels = col.get("levels")
            if levels:
                shown = ", ".join(map(str, levels[:max_levels])) + (", ..." if len(levels) > max_levels else "")
                bits.append(f"levels {shown}")
            parts.append(f"{col['name']} ({'; '.join(bits)})")
        return f"{self.name}: {self.rows} rows; columns: " + ", ".join(parts)


def _dataset_from_row(row: sqlite3.Row) -> DatasetRecord:
    return DatasetRecord(id=row["id"], sha256=row["sha256"], name=row["name"], format=row["format"],
                         options=json.loads(row["options"]), stored_path=row["stored_path"],
                         original_path=row["original_path"], rows=row["rows"], columns=row["columns"],
                         schema=json.loads(row["schema"]), imported_at=row["imported_at"])


class Repository:
    def __init__(self, db: Database) -> None:
        self.db = db

    # ---------------------------------------------------------------- sessions
    def create_session(self, title: str) -> str:
        sid = new_id()
        t = now()
        with self.db.tx() as c:
            c.execute("INSERT INTO sessions VALUES (?,?,?,?)", (sid, title, t, t))
        return sid

    def list_sessions(self) -> list[sqlite3.Row]:
        return self.db.query("SELECT * FROM sessions ORDER BY updated_at DESC")

    def rename_session(self, sid: str, title: str) -> None:
        with self.db.tx() as c:
            c.execute("UPDATE sessions SET title=?, updated_at=? WHERE id=?", (title, now(), sid))

    def touch_session(self, sid: str) -> None:
        with self.db.tx() as c:
            c.execute("UPDATE sessions SET updated_at=? WHERE id=?", (now(), sid))

    # ------------------------------------------------------------------- nodes
    def insert_node(self, node: Node) -> None:
        with self.db.tx() as c:
            c.execute(
                f"INSERT INTO nodes ({_NODE_COLS}) VALUES ({','.join('?' * 25)})",
                self._node_values(node),
            )

    def update_node(self, node: Node) -> None:
        node.updated_at = now()
        vals = self._node_values(node)
        cols = [c.strip() for c in _NODE_COLS.split(",")]
        assigns = ", ".join(f"{c}=?" for c in cols[1:])
        with self.db.tx() as c:
            c.execute(f"UPDATE nodes SET {assigns} WHERE id=?", (*vals[1:], node.id))

    @staticmethod
    def _node_values(n: Node) -> tuple:
        return (
            n.id, n.lineage_id, n.version, n.session_id, n.layer.value, n.type.value, n.domain.value,
            n.title, n.content, n.content_canonical, n.fingerprint, n.tool_name, _dumps(n.tool_inputs),
            _dumps(n.result), n.status.value, int(n.locked), int(n.needs_recheck), n.confidence,
            _dumps(n.risk) or "{}", _dumps(n.flags) or "[]", n.ladder_stage.value, n.role,
            n.superseded_by, n.created_at, n.updated_at,
        )

    def get_node(self, node_id: str) -> Node | None:
        row = self.db.query_one(f"SELECT {_NODE_COLS} FROM nodes WHERE id=?", (node_id,))
        return node_from_row(row) if row else None

    def get_nodes(self, ids: Iterable[str]) -> list[Node]:
        ids = list(ids)
        if not ids:
            return []
        out: list[Node] = []
        for i in range(0, len(ids), 500):
            chunk = ids[i : i + 500]
            rows = self.db.query(
                f"SELECT {_NODE_COLS} FROM nodes WHERE id IN ({','.join('?' * len(chunk))})", chunk
            )
            out.extend(node_from_row(r) for r in rows)
        return out

    def lineage(self, lineage_id: str) -> list[Node]:
        rows = self.db.query(
            f"SELECT {_NODE_COLS} FROM nodes WHERE lineage_id=? ORDER BY version", (lineage_id,)
        )
        return [node_from_row(r) for r in rows]

    def delete_node(self, node_id: str) -> None:
        with self.db.tx() as c:
            c.execute("DELETE FROM edges WHERE src=? OR dst=?", (node_id, node_id))
            c.execute("UPDATE evidence SET check_node_id=NULL WHERE check_node_id=?", (node_id,))
            c.execute("DELETE FROM evidence WHERE node_id=?", (node_id,))
            c.execute("DELETE FROM session_links WHERE node_id=?", (node_id,))
            c.execute("UPDATE nodes SET superseded_by=NULL WHERE superseded_by=?", (node_id,))
            c.execute("DELETE FROM nodes WHERE id=?", (node_id,))

    # ------------------------------------------------------------ session view
    def link(self, session_id: str, node_id: str) -> str:
        """Put a node into a session's view and return its handle (idempotent)."""
        with self.db.tx() as c:
            row = c.execute(
                "SELECT handle FROM session_links WHERE session_id=? AND node_id=?", (session_id, node_id)
            ).fetchone()
            if row:
                return row["handle"]
            n = c.execute(
                "SELECT COUNT(*) AS n FROM session_links WHERE session_id=?", (session_id,)
            ).fetchone()["n"]
            handle = f"n{n + 1}"
            while c.execute(
                "SELECT 1 FROM session_links WHERE session_id=? AND handle=?", (session_id, handle)
            ).fetchone():
                n += 1
                handle = f"n{n + 1}"
            c.execute("INSERT INTO session_links VALUES (?,?,?)", (session_id, node_id, handle))
            return handle

    def session_view(self, session_id: str) -> list[tuple[Node, str]]:
        rows = self.db.query(
            f"SELECT {', '.join('n.' + c.strip() for c in _NODE_COLS.split(','))}, l.handle "
            "FROM session_links l JOIN nodes n ON n.id = l.node_id WHERE l.session_id=? "
            "ORDER BY n.created_at",
            (session_id,),
        )
        return [(node_from_row(r), r["handle"]) for r in rows]

    def sessions_containing(self, node_ids: Iterable[str]) -> dict[str, list[str]]:
        ids = list(node_ids)
        out: dict[str, list[str]] = {}
        for i in range(0, len(ids), 500):
            chunk = ids[i : i + 500]
            for r in self.db.query(
                f"SELECT session_id, node_id FROM session_links WHERE node_id IN ({','.join('?' * len(chunk))})",
                chunk,
            ):
                out.setdefault(r["session_id"], []).append(r["node_id"])
        return out

    # ------------------------------------------------------------------- edges
    def add_edge(self, edge: Edge) -> None:
        with self.db.tx() as c:
            c.execute(
                "INSERT OR IGNORE INTO edges VALUES (?,?,?,?)",
                (edge.src, edge.dst, edge.kind.value, _dumps(edge.meta) or "{}"),
            )

    def edges_touching(self, ids: Iterable[str]) -> list[Edge]:
        ids = list(ids)
        out: list[Edge] = []
        for i in range(0, len(ids), 400):
            chunk = ids[i : i + 400]
            ph = ",".join("?" * len(chunk))
            rows = self.db.query(
                f"SELECT * FROM edges WHERE src IN ({ph}) OR dst IN ({ph})", chunk + chunk
            )
            out.extend(Edge(r["src"], r["dst"], EdgeKind(r["kind"]), _loads(r["meta"]) or {}) for r in rows)
        uniq = {(e.src, e.dst, e.kind): e for e in out}
        return list(uniq.values())

    def dependencies(self, node_id: str) -> list[str]:
        return [
            r["dst"]
            for r in self.db.query(
                "SELECT dst FROM edges WHERE src=? AND kind='depends_on'", (node_id,)
            )
        ]

    def ancestors(self, node_id: str) -> list[str]:
        rows = self.db.query(
            """WITH RECURSIVE anc(id) AS (
                   SELECT dst FROM edges WHERE src=? AND kind='depends_on'
                   UNION SELECT e.dst FROM edges e JOIN anc a ON e.src=a.id WHERE e.kind='depends_on')
               SELECT id FROM anc""",
            (node_id,),
        )
        return [r["id"] for r in rows]

    def dependents_transitive(self, node_id: str) -> list[str]:
        kinds = ",".join(f"'{k.value}'" for k in CASCADE_KINDS)
        rows = self.db.query(
            f"""WITH RECURSIVE dep(id) AS (
                    SELECT src FROM edges WHERE dst=? AND kind IN ({kinds})
                    UNION SELECT e.src FROM edges e JOIN dep d ON e.dst=d.id WHERE e.kind IN ({kinds}))
                SELECT id FROM dep""",
            (node_id,),
        )
        return [r["id"] for r in rows]

    # ----------------------------------------------------------------- lookups
    def find_by_fingerprint(self, fingerprint: str, statuses: Iterable[Status]) -> list[Node]:
        sts = [s.value for s in statuses]
        rows = self.db.query(
            f"SELECT {_NODE_COLS} FROM nodes WHERE fingerprint=? AND status IN ({','.join('?' * len(sts))}) "
            "AND superseded_by IS NULL ORDER BY updated_at DESC",
            (fingerprint, *sts),
        )
        return [node_from_row(r) for r in rows]

    def verified_finals_for_problem(self, problem_fingerprint: str) -> list[Node]:
        rows = self.db.query(
            f"""SELECT {', '.join('f.' + c.strip() for c in _NODE_COLS.split(','))}
                FROM nodes p JOIN edges e ON e.dst=p.id AND e.kind='depends_on'
                JOIN nodes f ON f.id=e.src
                WHERE p.fingerprint=? AND p.type='problem' AND f.type='final' AND f.status='verified'
                ORDER BY f.updated_at DESC""",
            (problem_fingerprint,),
        )
        return [node_from_row(r) for r in rows]

    def verified_results(self, limit: int = 200) -> list[Node]:
        rows = self.db.query(
            f"SELECT {_NODE_COLS} FROM nodes WHERE status='verified' AND type IN ('tool_result','final') "
            "ORDER BY updated_at DESC LIMIT ?", (limit,))
        return [node_from_row(r) for r in rows]

    def flag_unverified(self) -> int:
        """On startup: unverified nodes from earlier runs become hints that need re-checking."""
        with self.db.tx() as c:
            cur = c.execute(
                "UPDATE nodes SET needs_recheck=1 WHERE status IN ('proposed','hypothesis') "
                "AND type NOT IN ('problem','error')"
            )
            return cur.rowcount

    # ----------------------------------------------------------------- history
    def log_status(self, node_id: str, old: str | None, new: str, reason: str, caused_by: str | None) -> None:
        with self.db.tx() as c:
            c.execute(
                "INSERT INTO status_events VALUES (?,?,?,?,?,?,?)",
                (new_id(), node_id, old, new, reason, caused_by, now()),
            )

    def status_history(self, node_id: str) -> list[sqlite3.Row]:
        return self.db.query(
            "SELECT * FROM status_events WHERE node_id=? ORDER BY created_at", (node_id,)
        )

    def add_evidence(self, ev: Evidence) -> None:
        with self.db.tx() as c:
            c.execute(
                "INSERT INTO evidence VALUES (?,?,?,?,?,?,?,?,?)",
                (ev.id, ev.node_id, ev.check_node_id, ev.method, ev.tool_name, _dumps(ev.inputs),
                 ev.outcome, _dumps(ev.detail), ev.created_at),
            )

    def evidence_for(self, node_id: str) -> list[Evidence]:
        rows = self.db.query("SELECT * FROM evidence WHERE node_id=? ORDER BY created_at", (node_id,))
        return [
            Evidence(
                id=r["id"], node_id=r["node_id"], check_node_id=r["check_node_id"], method=r["method"],
                tool_name=r["tool_name"], inputs=_loads(r["inputs"]), outcome=r["outcome"],
                detail=_loads(r["detail"]), created_at=r["created_at"],
            )
            for r in rows
        ]

    def log_tool_run(self, session_id: str, node_id: str | None, tool: str, inputs: dict,
                     output: Any, error: str | None, duration_ms: float) -> str:
        rid = new_id()
        with self.db.tx() as c:
            c.execute(
                "INSERT INTO tool_runs VALUES (?,?,?,?,?,?,?,?,?)",
                (rid, session_id, node_id, tool, _dumps(inputs), _dumps(output), error, duration_ms, now()),
            )
        return rid

    def attach_run(self, run_id: str, node_id: str) -> None:
        with self.db.tx() as c:
            c.execute("UPDATE tool_runs SET node_id=? WHERE id=?", (node_id, run_id))

    def runs_for(self, node_id: str) -> list[sqlite3.Row]:
        return self.db.query("SELECT * FROM tool_runs WHERE node_id=? ORDER BY created_at", (node_id,))

    def add_message(self, session_id: str, author: str, text: str, node_id: str | None = None,
                    pin_number: int | None = None, pin_xy: tuple[float, float] | None = None,
                    sent: bool = False) -> str:
        mid = new_id()
        with self.db.tx() as c:
            c.execute(
                "INSERT INTO messages VALUES (?,?,?,?,?,?,?,?,?,?)",
                (mid, session_id, node_id, author, text, pin_number,
                 pin_xy[0] if pin_xy else None, pin_xy[1] if pin_xy else None, int(sent), now()),
            )
        return mid

    def messages(self, session_id: str, node_id: str | None = None) -> list[sqlite3.Row]:
        if node_id is None:
            return self.db.query(
                "SELECT * FROM messages WHERE session_id=? AND node_id IS NULL ORDER BY created_at",
                (session_id,),
            )
        return self.db.query(
            "SELECT * FROM messages WHERE node_id=? ORDER BY created_at", (node_id,)
        )

    def next_pin_number(self, node_id: str) -> int:
        row = self.db.query_one(
            "SELECT COALESCE(MAX(pin_number), 0) AS m FROM messages WHERE node_id=?", (node_id,)
        )
        return int(row["m"]) + 1

    def add_notice(self, session_id: str, origin: str, text: str, detail: dict) -> None:
        with self.db.tx() as c:
            c.execute(
                "INSERT INTO notices VALUES (?,?,?,?,?,?,?)",
                (new_id(), session_id, origin, text, _dumps(detail), 0, now()),
            )

    def notices(self, session_id: str, unseen_only: bool = True) -> list[sqlite3.Row]:
        q = "SELECT * FROM notices WHERE session_id=?" + (" AND seen=0" if unseen_only else "")
        return self.db.query(q + " ORDER BY created_at", (session_id,))

    def mark_notices_seen(self, session_id: str) -> None:
        with self.db.tx() as c:
            c.execute("UPDATE notices SET seen=1 WHERE session_id=?", (session_id,))

    # ---------------------------------------------------------------- datasets
    def add_dataset(self, rec: DatasetRecord) -> DatasetRecord:
        """Insert, or for a re-import of the same bytes read the same way, refresh its name and
        import time so it becomes the current dataset of that name again."""
        with self.db.tx() as c:
            c.execute(
                "INSERT INTO datasets (id, sha256, name, format, options, stored_path, original_path, rows, "
                "columns, schema, imported_at) VALUES (?,?,?,?,?,?,?,?,?,?,?) ON CONFLICT(id) DO UPDATE SET "
                "name=excluded.name, original_path=excluded.original_path, imported_at=excluded.imported_at",
                (rec.id, rec.sha256, rec.name, rec.format, json.dumps(rec.options, sort_keys=True),
                 rec.stored_path, rec.original_path, rec.rows, rec.columns, json.dumps(rec.schema), rec.imported_at))
        return rec

    def dataset(self, dataset_id: str) -> DatasetRecord | None:
        row = self.db.query_one("SELECT * FROM datasets WHERE id=?", (dataset_id,))
        return None if row is None else _dataset_from_row(row)

    def dataset_by_name(self, name: str) -> DatasetRecord | None:
        """The current dataset of that name (the latest import), matched case-insensitively."""
        row = self.db.query_one("SELECT * FROM datasets WHERE lower(name)=lower(?) "
                                "ORDER BY imported_at DESC, rowid DESC LIMIT 1", (name,))
        return None if row is None else _dataset_from_row(row)

    def current_datasets(self, limit: int = 50) -> list[DatasetRecord]:
        """The latest import of each name, newest first."""
        rows = self.db.query(
            "SELECT * FROM datasets d WHERE d.rowid = (SELECT d2.rowid FROM datasets d2 WHERE "
            "lower(d2.name)=lower(d.name) ORDER BY d2.imported_at DESC, d2.rowid DESC LIMIT 1) "
            "ORDER BY d.imported_at DESC LIMIT ?", (limit,))
        return [_dataset_from_row(r) for r in rows]

    def datasets_named_in(self, text: str) -> list[DatasetRecord]:
        """Current datasets whose name ("trial.csv") or name without its extension ("trial")
        appears in ``text`` as a whole word."""
        lower = text.lower()
        out = []
        for rec in self.current_datasets(500):
            names = {rec.name.lower(), Path(rec.name).stem.lower()}
            if any(n and re.search(r"(?<![\w.-])" + re.escape(n) + r"(?!\w|[.-]\w)", lower) for n in names):
                out.append(rec)
        return out
