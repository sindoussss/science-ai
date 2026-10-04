"""Layered layout: a node's column is its dependency depth; check pills hang under their target.

Rows are a shared grid so siblings line up across columns; each row is as tall
as its tallest block (node plus the check pills under it).
"""
from __future__ import annotations

from sciai.graph.model import EdgeKind, Node, NodeType

# Node geometry (scene px) shared with node_item.
NODE_W, NODE_H = 176, 64
PLOT_H = 112
PILL_H = 22
PILL_INSET = 12  # pills are narrower than their parent, centered under it
PILL_TOP_GAP = 6
PILL_GAP = 4
COL_GAP = 56
ROW_GAP = 36


def node_height(node: Node) -> float:
    if node.type == NodeType.CHECK:
        return PILL_H
    if node.result and node.result.get("kind") == "plotspec":
        return PLOT_H
    return NODE_H


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
        ps = [p for p in parents.get(nid, []) if p not in check_target]
        depth[nid] = 0 if not ps else 1 + max(d(p, stack | {nid}) for p in ps)
        return depth[nid]

    order = sorted(nodes.values(), key=lambda n: n.created_at)
    checks_of: dict[str, list[str]] = {}
    orphans: list[str] = []
    columns: dict[int, list[str]] = {}
    for n in order:
        if n.id in check_target:
            checks_of.setdefault(check_target[n.id], []).append(n.id)
        elif n.type == NodeType.CHECK:
            orphans.append(n.id)
        else:
            columns.setdefault(d(n.id), []).append(n.id)

    def block_h(nid: str) -> float:
        k = len(checks_of.get(nid, []))
        return node_height(nodes[nid]) + (PILL_TOP_GAP + k * PILL_H + (k - 1) * PILL_GAP if k else 0)

    n_rows = max((len(ids) for ids in columns.values()), default=0)
    row_h = [max((block_h(ids[r]) for ids in columns.values() if r < len(ids)), default=NODE_H)
             for r in range(n_rows)]
    row_y = [sum(row_h[:r]) + r * ROW_GAP for r in range(n_rows)]

    pos: dict[str, tuple[float, float]] = {}
    for col, ids in columns.items():
        for row, nid in enumerate(ids):
            pos[nid] = (col * (NODE_W + COL_GAP), row_y[row])
    for tgt, ids in checks_of.items():
        tx, ty = pos.get(tgt, (0.0, 0.0))
        top = ty + node_height(nodes[tgt]) + PILL_TOP_GAP if tgt in nodes else ty
        for k, cid in enumerate(ids):
            pos[cid] = (tx + PILL_INSET, top + k * (PILL_H + PILL_GAP))
    for k, cid in enumerate(orphans):
        pos[cid] = (-(NODE_W + COL_GAP), k * (PILL_H + PILL_GAP))
    return pos
