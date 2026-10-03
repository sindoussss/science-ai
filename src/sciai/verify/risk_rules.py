"""Hard-coded risk rules. The controller never decides riskiness by feel:
these rules decide whether a check is required, and which checks are offered."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from sciai.config import RiskConfig
from sciai.graph.model import Node, Status, TERMINAL_BAD
from sciai.tools.registry import ToolSpec
from sciai.verify.sanity import sanity_issues

if TYPE_CHECKING:
    from sciai.graph.engine import GraphEngine


@dataclass
class RiskReport:
    fired: list[dict[str, str]] = field(default_factory=list)

    @property
    def required(self) -> bool:
        return bool(self.fired)

    def add(self, rule: str, reason: str) -> None:
        self.fired.append({"rule": rule, "reason": reason})

    def rules(self) -> list[str]:
        return [f["rule"] for f in self.fired]


def _same_result(a: dict[str, Any] | None, b: dict[str, Any] | None) -> bool:
    if a is None or b is None:
        return a is b
    if a.get("kind") == "expr" and b.get("kind") == "expr":
        return a.get("srepr") == b.get("srepr")
    if a.get("kind") == "number" and b.get("kind") == "number":
        x, y = float(a["value"]), float(b["value"])
        return abs(x - y) <= 1e-9 * max(1.0, abs(x), abs(y))
    if a.get("kind") == "list" and b.get("kind") == "list":
        return len(a["value"]) == len(b["value"]) and all(
            _same_result(p, q) for p, q in zip(a["value"], b["value"]))
    return a == b


def assess(engine: "GraphEngine", node: Node, spec: ToolSpec | None, cfg: RiskConfig,
           *, user_facing: bool = False, extra_dependents: int = 0) -> RiskReport:
    """``extra_dependents`` counts dependents about to be added (e.g. the final answer)."""
    report = RiskReport()
    args = node.tool_inputs or {}

    # Stakes: many downstream nodes depend on it, or the user will see it.
    n_dep = engine.dependents_count(node.id) + extra_dependents
    if n_dep >= cfg.stakes_threshold:
        report.add("stakes", f"{n_dep} downstream nodes depend on it")
    if user_facing:
        report.add("stakes", "it is part of the final answer")

    # Surprise: sanity failures, or a contradiction with existing graph content.
    for issue in sanity_issues(node.result, args):
        report.add("surprise", issue)
    if node.fingerprint:
        for other in engine.repo.find_by_fingerprint(
                node.fingerprint, [Status.VERIFIED, Status.PROPOSED, Status.HYPOTHESIS]):
            if other.id != node.id and other.status not in TERMINAL_BAD and not _same_result(other.result, node.result):
                report.add("surprise", f"contradicts an earlier result for the same inputs ({other.id[:8]})")
                break
    unsourced = [f for f in node.flags if f.startswith("unsourced_numbers")]
    if unsourced:
        report.add("surprise", unsourced[0].replace("_", " "))

    # Confidence: the producing tool reported low confidence.
    if node.confidence is not None and node.confidence < cfg.min_confidence:
        report.add("confidence", f"tool confidence {node.confidence:.2f} < {cfg.min_confidence}")

    # Step type: the always-check list.
    if spec is not None and spec.requires_check(args):
        report.add("step_type", f"{spec.name} is on the always-check list")
    return report
