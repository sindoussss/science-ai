"""The verifier: pure code, no LLM. Runs independent checks and records evidence."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from sciai.graph.engine import GraphEngine
from sciai.graph.model import EdgeKind, Evidence, Layer, Node, NodeType, Status, TERMINAL_BAD
from sciai.tools.registry import get as get_tool
from sciai.tools.sandbox import Runner
from sciai.verify.checks import CheckPlan, plans_for


@dataclass
class CheckRun:
    plan: CheckPlan
    outcome: str
    detail: dict[str, Any]
    check_node: Node | None


@dataclass
class VerifyOutcome:
    status: str  # verified | failed | inconclusive | unavailable
    runs: list[CheckRun] = field(default_factory=list)

    @property
    def failed_run(self) -> CheckRun | None:
        return next((r for r in self.runs if r.outcome == "fail"), None)


class Verifier:
    def __init__(self, engine: GraphEngine, runner: Runner, plausibility: bool = True) -> None:
        self.engine = engine
        self.runner = runner
        self.plausibility = plausibility

    def offered(self, node: Node) -> list[tuple[CheckPlan, dict[str, Any]]]:
        plans = plans_for(node.tool_name, node.tool_inputs, node.result)
        if not self.plausibility:
            plans = [(p, a) for p, a in plans if p.method != "plausibility"]
        return plans

    def run_plan(self, node: Node, plan: CheckPlan, args: dict[str, Any]) -> CheckRun:
        outcome = self.runner.run(plan.checker, args)
        repo = self.engine.repo
        if not outcome.ok:
            result = {"kind": "check", "outcome": "inconclusive", "error": outcome.error}
        else:
            result = outcome.value["result"]
        verdict = result.get("outcome", "inconclusive")
        status = {"pass": Status.VERIFIED, "fail": Status.FAILED}.get(verdict, Status.PROPOSED)
        check = Node(
            session_id=self.engine.session_id or "", layer=Layer.REASONING, type=NodeType.CHECK,
            title=f"check {self.engine.handle(node.id)}: {plan.method}", content=f"{plan.checker}",
            tool_name=plan.checker, tool_inputs=args, result=result, status=status, role="verifier",
        )
        self.engine.add_node(check, tool_domain=get_tool(plan.checker).domain)
        self.engine.add_edge(check.id, node.id, EdgeKind.CHECKS)
        run_id = repo.log_tool_run(self.engine.session_id or "", check.id, plan.checker, args,
                                   result, outcome.error, outcome.duration_ms)
        _ = run_id
        repo.add_evidence(Evidence(node_id=node.id, check_node_id=check.id, method=plan.method,
                                   tool_name=plan.checker, inputs=args, outcome=verdict, detail=result))
        return CheckRun(plan, verdict, result, check)

    def verify(self, node: Node, only_method: str | None = None) -> VerifyOutcome:
        """Run every required check, then optional ones cheapest-first until one passes.

        Any fail fails the node. It is verified when every ``must_pass`` check passed,
        at least one check passed, and, if optional checks were offered, one of them
        passed: a required check such as units can't confirm the value on its own."""
        node = self.engine.resolve(node.id)
        if node.status in TERMINAL_BAD:
            return VerifyOutcome("unavailable")
        plans = self.offered(node)
        required = [(p, a) for p, a in plans if p.required]
        optional = [(p, a) for p, a in plans if not p.required]
        if only_method is not None:
            if not any(only_method in (p.method, p.checker) for p, _ in plans):
                return VerifyOutcome("unavailable")
            # Choosing an optional check narrows the optional ones; naming a required one
            # changes nothing, since required checks always run.
            chosen = [(p, a) for p, a in optional if only_method in (p.method, p.checker)]
            optional_offered, optional = bool(optional), chosen or optional
        else:
            optional_offered = bool(optional)
        if not plans:
            return VerifyOutcome("unavailable")
        out = VerifyOutcome("inconclusive")
        blocked = optional_passed = False
        for plan, args in required + optional:
            run = self.run_plan(node, plan, args)
            out.runs.append(run)
            if run.outcome == "fail":
                self.engine.set_status(node.id, Status.FAILED, f"{plan.method} check failed",
                                       run.check_node.id if run.check_node else None)
                out.status = "failed"
                return out
            if run.outcome != "pass":
                blocked = blocked or plan.must_pass
            elif not plan.required:
                optional_passed = True
                break  # one optional pass is enough
        passes = [r for r in out.runs if r.outcome == "pass"]
        if passes and not blocked and (optional_passed or not optional_offered):
            last = passes[-1]
            if node.status != Status.HYPOTHESIS:  # chem keeps its evidence but stays a hypothesis
                self.engine.set_status(node.id, Status.VERIFIED, f"{last.plan.method} check passed",
                                       last.check_node.id if last.check_node else None)
            out.status = "verified"
        return out
