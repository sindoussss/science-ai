"""Top-to-bottom layered layout: a node's row is its dependency depth, so arrows point down
from what a step uses to the step itself. Siblings sit side by side under the mean x of
their parents; check pills hang under the node they check.

Each row is as tall as its tallest block (node plus the check pills under it). Modelling
assumptions stand in a column to the left of everything else, so a long checklist doesn't
widen the top row past the workspace.
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
COL_GAP = 40
ROW_GAP = 44
ASSUMPTION_GAP = 12


# Result kinds drawn as a picture in the node body rather than as a line of text. Both the
# layout height and the item's own size() read this, so a new kind cannot get a thumbnail in
# one place and a 64px box in the other.
THUMBNAIL_KINDS = ("plotspec", "molecule")


def has_thumbnail(node: Node) -> bool:
    result = node.result or {}
    if result.get("kind") not in THUMBNAIL_KINDS:
        return False
    return result.get("kind") != "molecule" or bool(result.get("canonical_smiles"))


def node_height(node: Node) -> float:
    if node.type == NodeType.CHECK:
        return PILL_H
    if has_thumbnail(node):
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
    assumptions: list[str] = []
    rows: dict[int, list[str]] = {}
    for n in order:
        if n.id in check_target:
            checks_of.setdefault(check_target[n.id], []).append(n.id)
        elif n.type == NodeType.CHECK:
            orphans.append(n.id)
        elif n.type == NodeType.ASSUMPTION:
            assumptions.append(n.id)
        else:
            rows.setdefault(d(n.id), []).append(n.id)

    def block_h(nid: str) -> float:
        k = len(checks_of.get(nid, []))
        return node_height(nodes[nid]) + (PILL_TOP_GAP + k * PILL_H + (k - 1) * PILL_GAP if k else 0)

    pitch = NODE_W + COL_GAP
    pos: dict[str, tuple[float, float]] = {}
    y = 0.0
    for level in sorted(rows):
        ids = rows[level]
        if not pos:
            xs = [k * pitch for k in range(len(ids))]
            shift = 0.0
        else:
            # want each node under the mean x of its (already placed) parents, in that order
            def want(nid: str) -> float:
                px = [pos[p][0] for p in parents.get(nid, []) if p in pos and p not in check_target]
                return sum(px) / len(px) if px else 0.0
            ids = sorted(ids, key=lambda nid: want(nid))  # stable: ties keep creation order
            wanted = [want(nid) for nid in ids]
            xs = []
            for k, w in enumerate(wanted):
                xs.append(w if k == 0 else max(w, xs[-1] + pitch))
            # pushing right drifts the row; shift it back so it is centered on what it wanted
            shift = (sum(wanted) - sum(xs)) / len(xs)
        for nid, x in zip(ids, xs):
            pos[nid] = (x + shift, y)
        y += max(block_h(nid) for nid in ids) + ROW_GAP
    for tgt, ids in checks_of.items():
        tx, ty = pos.get(tgt, (0.0, 0.0))
        top = ty + node_height(nodes[tgt]) + PILL_TOP_GAP if tgt in nodes else ty
        for k, cid in enumerate(ids):
            pos[cid] = (tx + PILL_INSET, top + k * (PILL_H + PILL_GAP))
    left = min((x for x, _ in pos.values()), default=0.0)
    for k, aid in enumerate(assumptions):
        pos[aid] = (left - pitch, k * (NODE_H + ASSUMPTION_GAP))
    if assumptions:
        left -= pitch
    for k, cid in enumerate(orphans):
        pos[cid] = (left - pitch, k * (PILL_H + PILL_GAP))
    return pos
