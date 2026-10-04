"""The graph engine: the single writer of node state.

It keeps an in-memory networkx view of the open session (plus nodes linked in
from earlier sessions) and writes every change through to SQLite immediately.
All status rules live here so no caller (controller, UI, model) can bypass them.
"""
from __future__ import annotations

from typing import Iterable

import networkx as nx

from sciai.graph.cascade import cascade_invalidate
from sciai.graph.events import EventBus
from sciai.graph.model import (
    CascadeReport,
    Domain,
    Edge,
    EdgeKind,
    LadderStage,
    Node,
    NodeType,
    Status,
    TERMINAL_BAD,
)
from sciai.store.repository import Repository

# When a node has no tool, its domain comes from its dependencies, most restrictive first.
_DOMAIN_PRIORITY = [Domain.CHEM, Domain.DATA, Domain.PHYSICS, Domain.MATH]


class GraphRuleError(RuntimeError):
    """A requested change would break a graph rule (chem verified, locked delete, ...)."""


class GraphEngine:
    def __init__(self, repo: Repository, bus: EventBus | None = None) -> None:
        self.repo = repo
        self.bus = bus or EventBus()
        self.session_id: str | None = None
        self.g: nx.DiGraph = nx.DiGraph()
        self.nodes: dict[str, Node] = {}
        self.handles: dict[str, str] = {}
        self.by_handle: dict[str, str] = {}

    # ---------------------------------------------------------------- sessions
    def new_session(self, title: str) -> str:
        sid = self.repo.create_session(title)
        self.open_session(sid)
        return sid

    def open_session(self, session_id: str) -> None:
        self.session_id = session_id
        self.g = nx.DiGraph()
        self.nodes.clear()
        self.handles.clear()
        self.by_handle.clear()
        for node, handle in self.repo.session_view(session_id):
            self._cache(node, handle)
        for e in self.repo.edges_touching(list(self.nodes)):
            if e.src in self.nodes and e.dst in self.nodes:
                self.g.add_edge(e.src, e.dst, kind=e.kind)
        self.bus.emit("session_opened", session_id=session_id)

    def _require_session(self) -> str:
        if self.session_id is None:
            raise GraphRuleError("no session is open")
        return self.session_id

    def _cache(self, node: Node, handle: str) -> None:
        self.nodes[node.id] = node
        self.handles[node.id] = handle
        self.by_handle[handle] = node.id
        self.g.add_node(node.id)

    def snapshot(self) -> tuple[list[tuple[Node, str]], list[tuple[str, str, EdgeKind]]]:
        """Deep copy of the open session for another thread (the UI) to render."""
        import copy

        nodes = [(copy.deepcopy(n), self.handles[nid]) for nid, n in self.nodes.items()]
        edges = [(u, v, d["kind"]) for u, v, d in self.g.edges(data=True)]
        return nodes, edges

    # ----------------------------------------------------------------- lookups
    def resolve(self, ref: str) -> Node:
        """Accept a handle (n3) or a node id."""
        nid = self.by_handle.get(ref, ref)
        node = self.nodes.get(nid) or self.repo.get_node(nid)
        if node is None:
            raise KeyError(f"unknown node {ref!r}")
        return node

    def node(self, node_id: str) -> Node:
        return self.resolve(node_id)

    def handle(self, node_id: str) -> str:
        return self.handles.get(node_id, node_id[:8])

    def dependencies(self, node_id: str) -> list[str]:
        return self.repo.dependencies(node_id)

    def ancestors(self, node_id: str) -> list[str]:
        return self.repo.ancestors(node_id)

    def dependents_count(self, node_id: str) -> int:
        return len(self.repo.dependents_transitive(node_id))

    def latest_in_lineage(self, lineage_id: str) -> Node:
        return self.repo.lineage(lineage_id)[-1]

    def root(self) -> Node | None:
        problems = [n for n in self.nodes.values() if n.type == NodeType.PROBLEM and n.session_id == self.session_id]
        return problems[0] if problems else None

    # ------------------------------------------------------------------ writes
    def add_node(self, node: Node, depends_on: Iterable[str] = (), *,
                 tool_domain: Domain | None = None, replaces: str | None = None) -> Node:
        """Insert a node. Domain comes only from ``tool_domain`` (the tool registry)
        or is derived from dependencies; whatever the caller put on ``node.domain`` is ignored."""
        sid = self._require_session()
        node.session_id = sid
        dep_ids = [self.resolve(d).id for d in depends_on]
        for d in dep_ids:
            dep = self.resolve(d)
            if dep.status in TERMINAL_BAD and node.type not in (NodeType.CHECK, NodeType.ERROR):
                raise GraphRuleError(
                    f"cannot build on {self.handle(d)}: it is {dep.status.value}"
                )

        node.domain = tool_domain if tool_domain is not None else self._derive_domain(dep_ids)
        if node.domain == Domain.CHEM and node.status == Status.VERIFIED:
            node.status = Status.HYPOTHESIS
        if node.domain == Domain.CHEM and node.status == Status.PROPOSED:
            node.status = Status.HYPOTHESIS
        if node.status == Status.VERIFIED and node.type != NodeType.CHECK:
            # A check node's status is its own deterministic outcome (pass = verified).
            raise GraphRuleError("new nodes start unverified; only the verifier can verify")

        prev: Node | None = None
        if replaces is not None:
            prev = self.latest_in_lineage(self.resolve(replaces).lineage_id)
            node.lineage_id = prev.lineage_id
            node.version = prev.version + 1

        with self.repo.db.tx():
            self.repo.insert_node(node)
            handle = self.repo.link(sid, node.id)
            for d in dep_ids:
                self.repo.add_edge(Edge(node.id, d, EdgeKind.DEPENDS_ON))
            if prev is not None:
                prev.superseded_by = node.id
                self.repo.update_node(prev)
            self.repo.log_status(node.id, None, node.status.value, "created", None)
        self._cache(node, handle)
        for d in dep_ids:
            if d in self.nodes:
                self.g.add_edge(node.id, d, kind=EdgeKind.DEPENDS_ON)
        if prev is not None:
            self._refresh(prev.id)
        self.bus.emit("node_added", node=node, handle=handle, depends_on=dep_ids)
        return node

    def _derive_domain(self, dep_ids: list[str]) -> Domain:
        domains = {self.resolve(d).domain for d in dep_ids}
        for d in _DOMAIN_PRIORITY:
            if d in domains:
                return d
        return Domain.GENERAL

    def add_edge(self, src: str, dst: str, kind: EdgeKind, meta: dict | None = None) -> None:
        s, d = self.resolve(src).id, self.resolve(dst).id
        self.repo.add_edge(Edge(s, d, kind, meta or {}))
        if s in self.nodes and d in self.nodes:
            self.g.add_edge(s, d, kind=kind)
        self.bus.emit("edge_added", src=s, dst=d, edge_kind=kind)

    def link_existing(self, node_id: str) -> str:
        """Bring a node from the knowledge store into this session (reuse, no copy)."""
        sid = self._require_session()
        node = self.repo.get_node(node_id)
        if node is None:
            raise KeyError(node_id)
        handle = self.repo.link(sid, node.id)
        self._cache(node, handle)
        for e in self.repo.edges_touching([node.id]):
            if e.src in self.nodes and e.dst in self.nodes:
                self.g.add_edge(e.src, e.dst, kind=e.kind)
        self.bus.emit("node_added", node=node, handle=handle, depends_on=[], reused=True)
        return handle

    def update(self, node: Node) -> None:
        """Persist non-status field changes (result, risk, flags, ladder stage...)."""
        stored = self.repo.get_node(node.id)
        if stored is not None and stored.status != node.status:
            raise GraphRuleError("use set_status() to change status")
        self.repo.update_node(node)
        if node.id in self.nodes:
            self.nodes[node.id] = node
        self.bus.emit("node_updated", node=node)

    def _refresh(self, node_id: str) -> Node | None:
        node = self.repo.get_node(node_id)
        if node is not None and node_id in self.nodes:
            self.nodes[node_id] = node
            self.bus.emit("node_updated", node=node)
        return node

    def set_status(self, ref: str, status: Status, reason: str,
                   caused_by: str | None = None) -> CascadeReport | None:
        node = self.resolve(ref)
        if node.status == status:
            return None
        if status == Status.VERIFIED:
            if node.domain == Domain.CHEM:
                raise GraphRuleError("chemistry nodes are hypotheses and can never be verified")
            if not any(ev.outcome == "pass" for ev in self.repo.evidence_for(node.id)):
                raise GraphRuleError("a node can only be verified with passing evidence")
        if node.status == Status.INVALIDATED and status != Status.INVALIDATED:
            raise GraphRuleError("invalidated is terminal; create a new version instead")
        old = node.status
        node.status = status
        if status == Status.VERIFIED:
            node.needs_recheck = False
        with self.repo.db.tx():
            self.repo.update_node(node)
            self.repo.log_status(node.id, old.value, status.value, reason, caused_by)
        if node.id in self.nodes:
            self.nodes[node.id] = node
        self.bus.emit("node_updated", node=node)

        report = None
        if status in TERMINAL_BAD:
            report = cascade_invalidate(self.repo, node.id, f"{self.handle(node.id)} {status.value}: {reason}",
                                        self.session_id)
            for nid in report.invalidated:
                self._refresh(nid)
            self.bus.emit("cascade", report=report)
        return report

    def set_ladder(self, ref: str, stage: LadderStage) -> None:
        node = self.resolve(ref)
        node.ladder_stage = stage
        self.update(node)

    # ------------------------------------------------------- user-side actions
    def lock(self, ref: str, locked: bool = True) -> None:
        node = self.resolve(ref)
        node.locked = locked
        self.update(node)

    def reject_assumption(self, ref: str, reason: str = "rejected by user") -> CascadeReport | None:
        """The user withdraws a modelling assumption: it fails, and every result built on it
        (in any session) is invalidated. It is unlocked first so no warning sticks to it."""
        node = self.resolve(ref)
        if node.type != NodeType.ASSUMPTION:
            raise GraphRuleError(f"{self.handle(node.id)} is not an assumption")
        if node.status in TERMINAL_BAD:
            return None
        if node.locked:
            node.locked = False
            self.update(node)
        return self.set_status(node.id, Status.FAILED, reason)

    def demote(self, ref: str, *, automatic: bool, reason: str = "demoted") -> None:
        """Verified -> proposed (needs re-check). Automatic demotion skips locked nodes."""
        node = self.resolve(ref)
        if node.status != Status.VERIFIED:
            return
        if automatic and node.locked:
            node.needs_recheck = True
            self.update(node)
            return
        old = node.status
        node.status = Status.PROPOSED
        node.needs_recheck = True
        with self.repo.db.tx():
            self.repo.update_node(node)
            self.repo.log_status(node.id, old.value, node.status.value, reason, None)
        self.bus.emit("node_updated", node=node)

    def delete(self, ref: str, *, automatic: bool = False) -> CascadeReport | None:
        """User delete. Dependents are invalidated first. Locked nodes refuse deletion."""
        node = self.resolve(ref)
        if node.locked:
            if automatic:
                return None
            raise GraphRuleError("node is locked; unlock it before deleting")
        report = None
        if node.status != Status.INVALIDATED:
            report = self.set_status(node.id, Status.INVALIDATED, "deleted by user")
        self.repo.delete_node(node.id)
        handle = self.handles.pop(node.id, None)
        if handle:
            self.by_handle.pop(handle, None)
        self.nodes.pop(node.id, None)
        if node.id in self.g:
            self.g.remove_node(node.id)
        self.bus.emit("node_removed", node_id=node.id)
        return report

    def prune_invalidated(self) -> int:
        """Housekeeping: drop invalidated nodes of this session. Never touches locked nodes."""
        count = 0
        for node in list(self.nodes.values()):
            if node.status == Status.INVALIDATED and not node.locked and node.session_id == self.session_id:
                if not self.repo.dependents_transitive(node.id):
                    self.delete(node.id, automatic=True)
                    count += 1
        return count
