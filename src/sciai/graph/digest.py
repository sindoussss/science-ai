"""Compact text view of the session graph for the controller prompt.

The digest is sized to a token budget derived from num_ctx, so a long graph
never overflows the context window. When it must drop lines it keeps, in
order: the root problem, nodes that need attention (failed, pending check,
flagged), the most recent nodes, then verified results.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from sciai.graph.model import EdgeKind, Node, NodeType, Status

if TYPE_CHECKING:
    from sciai.graph.engine import GraphEngine

MAX_RESULT_CHARS = 140
RECENT = 6


def _line(engine: "GraphEngine", node: Node) -> str:
    deps = [engine.handle(d) for d in engine.g.successors(node.id)
            if engine.g.edges[node.id, d].get("kind") == EdgeKind.DEPENDS_ON and d in engine.handles]
    status = node.status.value
    if node.needs_recheck:
        status += ",needs-recheck"
    if node.risk.get("pending_checks"):
        status += ",check-pending:" + "|".join(node.risk["pending_checks"])
    res = node.display_result()
    if len(res) > MAX_RESULT_CHARS:
        res = res[: MAX_RESULT_CHARS - 3] + "..."
    what = node.tool_name or node.type.value
    h = engine.handle(node.id)
    if node.type == NodeType.PROBLEM:
        return f"{h} [problem] {node.content}{_givens(node)}"
    if node.type == NodeType.ASSUMPTION:
        tag = "rejected" if node.status == Status.FAILED else (
            "assumed" if node.status != Status.INVALIDATED else "invalidated")
        if tag == "assumed" and any(f.startswith("doubtful") for f in node.flags):
            tag = "doubtful"
        return f"{h} [{tag}] {node.content}"
    if node.type == NodeType.ENTITY and (node.result or {}).get("kind") == "dataset":
        return f"{h} [dataset {status}] {node.content[:MAX_RESULT_CHARS]}"
    if node.type == NodeType.ENTITY:
        return f"{h} [entity] {node.title}: {node.content[:MAX_RESULT_CHARS]}"
    text = f"{h} [{status}] {what}"
    if node.tool_inputs and node.type == NodeType.TOOL_RESULT:
        entities = {}
        for d in engine.g.successors(node.id):
            dep = engine.nodes.get(d)
            if dep is not None and dep.type == NodeType.ENTITY and d in engine.handles:
                entities[(dep.tool_inputs or {}).get("arg")] = engine.handle(d)
        args = ", ".join(f"{k}={entities.get(k, engine.handles.get(v, v) if isinstance(v, str) else v)}"
                         for k, v in node.tool_inputs.items() if k != "assumptions")
        text += f"({args[:120]})"
    if res:
        text += f" = {res}"
    if deps:
        text += f" <- {','.join(deps)}"
    return text


def _quantity_text(q: dict) -> str:
    kind = f" ({q['kind']})" if q.get("kind") else ""
    return f"{q.get('value')} {q.get('unit', '')}".rstrip() + kind


def _givens(node: Node) -> str:
    givens = (node.tool_inputs or {}).get("givens") or {}
    if not givens:
        return ""
    return " | givens: " + ", ".join(f"{k}={_quantity_text(v)}" for k, v in givens.items())


def build_digest(engine: "GraphEngine", budget_tokens: int, chars_per_token: float = 3.0) -> str:
    nodes = [n for n in engine.nodes.values() if n.type not in (NodeType.CHECK,)]
    nodes.sort(key=lambda n: n.created_at)
    if not nodes:
        return "(empty graph)"
    budget = int(budget_tokens * chars_per_token)
    order = {n.id: i for i, n in enumerate(nodes)}

    def priority(n: Node) -> tuple[int, int]:
        idx = order[n.id]
        if n.type == NodeType.PROBLEM:
            return (0, 0)
        if n.status in (Status.FAILED,) or n.risk.get("pending_checks") or n.flags or n.needs_recheck:
            return (1, -idx)
        if idx >= len(nodes) - RECENT:
            return (2, -idx)
        if n.status == Status.VERIFIED:
            return (3, -idx)
        if n.status == Status.INVALIDATED:
            return (5, -idx)
        return (4, -idx)

    chosen: list[tuple[Node, str]] = []
    used = 0
    for n in sorted(nodes, key=priority):
        line = _line(engine, n)
        if used + len(line) + 1 > budget - 60:
            continue
        chosen.append((n, line))
        used += len(line) + 1
    chosen.sort(key=lambda t: order[t[0].id])
    lines = [line for _, line in chosen]
    omitted = len(nodes) - len(chosen)
    if omitted:
        lines.append(f"(+{omitted} older nodes omitted; they remain in the graph)")
    return "\n".join(lines)
