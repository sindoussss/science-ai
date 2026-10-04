"""Layered layout: a node's column is its dependency depth; check nodes sit beside their target."""
from __future__ import annotations

from sciai.graph.model import EdgeKind, Node, NodeType

COL_W = 232
ROW_H = 112


def layered_positions(nodes: dict[str, Node], edges: list[tuple[str, str, EdgeKind]]) -> dict[str, tuple[float, float]]:
    parents: dict[str, list[str]] = {nid: [] for nid in nodes}
    check_target: dict[str, str] = {}
    for src, dst, kind in edges:
        if src not in nodes or dst not in nodes:
            continue
        if kind in (EdgeKind.DEPENDS_ON, EdgeKind.RENDERS):
            parents[src].append(dst)
        elif kind == EdgeKind.CHECKS:
            check_target[src] = dst

    depth: dict[str, int] = {}

    def d(nid: str, stack: frozenset = frozenset()) -> int:
        if nid in depth:
            return depth[nid]
        if nid in stack:  # defensive: the graph should be acyclic
            return 0
        ps = parents.get(nid, [])
        depth[nid] = 0 if not ps else 1 + max(d(p, stack | {nid}) for p in ps)
        return depth[nid]

    order = sorted(nodes.values(), key=lambda n: n.created_at)
    for n in order:
        if n.id in check_target:
            continue
        d(n.id)
    columns: dict[int, list[str]] = {}
    for n in order:
        if n.id in check_target or n.type == NodeType.CHECK and n.id not in check_target:
            continue
        columns.setdefault(depth[n.id], []).append(n.id)

    pos: dict[str, tuple[float, float]] = {}
    for col, ids in columns.items():
        for row, nid in enumerate(ids):
            pos[nid] = (col * COL_W, row * ROW_H)
    # checks: stacked under-right of their target, side by side with it
    stacked: dict[str, int] = {}
    for n in order:
        tgt = check_target.get(n.id)
        if tgt is None:
            if n.type == NodeType.CHECK and n.id not in pos:
                pos[n.id] = (-COL_W, len(pos) * 20.0)
            continue
        tx, ty = pos.get(tgt, (0.0, 0.0))
        k = stacked.get(tgt, 0)
        stacked[tgt] = k + 1
        pos[n.id] = (tx, ty + 72 + 30 * k)
    return pos
