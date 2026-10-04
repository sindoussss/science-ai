"""Failure ladder: retry (different method) -> backtrack -> escalate.

Position on the ladder is the number of versions in a lineage that failed a
check, so it survives restarts and is the same no matter who asks.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from sciai.graph.engine import GraphEngine
from sciai.graph.model import EdgeKind, LadderStage, Node, NodeType, Status
from sciai.tools.registry import get as get_tool


@dataclass
class LadderDecision:
    stage: LadderStage
    message: str
    frontier: list[str] = field(default_factory=list)  # handles to re-derive from
    conflict: list[str] = field(default_factory=list)   # node ids shown side by side


def failed_versions(engine: GraphEngine, node: Node) -> list[Node]:
    out = []
    for v in engine.repo.lineage(node.lineage_id):
        if any(ev.outcome == "fail" for ev in engine.repo.evidence_for(v.id)):
            out.append(v)
    return out


def _alternatives(node: Node) -> str:
    if not node.tool_name:
        return ""
    spec = get_tool(node.tool_name)
    used = (node.tool_inputs or {}).get("method", "default")
    others = [m for m in spec.methods if m != used]
    return f" Other methods for {spec.name}: {', '.join(others)}." if others else ""


def decide(engine: GraphEngine, node: Node) -> LadderDecision:
    n_failed = len(failed_versions(engine, node))
    h = engine.handle(node.id)
    if n_failed <= 1:
        engine.set_ladder(node.id, LadderStage.RETRIED)
        return LadderDecision(
            LadderStage.RETRIED,
            f"{h} failed its check. Its dependents were invalidated. Retry once with a different method "
            f"or tool, setting replaces={h}.{_alternatives(node)}",
        )
    if n_failed == 2:
        if engine.resolve(node.id).status != Status.INVALIDATED:
            engine.set_status(node.id, Status.INVALIDATED, "backtrack after a failed retry")
        engine.set_ladder(node.id, LadderStage.BACKTRACKED)
        frontier = []
        for anc_id in engine.ancestors(node.id):
            anc = engine.resolve(anc_id)
            if anc.status == Status.VERIFIED or anc.type == NodeType.PROBLEM:
                frontier.append(engine.handle(anc.id))
        frontier = sorted(set(frontier), key=lambda s: int(s[1:]) if s[1:].isdigit() else 0)
        return LadderDecision(
            LadderStage.BACKTRACKED,
            f"{h} failed again after a retry. Backtracking: {h} and everything built on it is invalidated. "
            f"Re-derive it from the last verified nodes ({', '.join(frontier)}), setting replaces={h} and "
            f"using a different approach.{_alternatives(node)}",
            frontier=frontier,
        )
    engine.set_ladder(node.id, LadderStage.ESCALATED)
    conflict: list[str] = []
    for v in failed_versions(engine, node)[-2:]:
        conflict.append(v.id)
        for ev in engine.repo.evidence_for(v.id):
            if ev.outcome == "fail" and ev.check_node_id:
                conflict.append(ev.check_node_id)
                engine.add_edge(v.id, ev.check_node_id, EdgeKind.CONFLICTS_WITH)
    return LadderDecision(
        LadderStage.ESCALATED,
        f"{h} kept failing after retry and backtrack. Stopping so you can compare the conflicting results.",
        conflict=conflict,
    )
