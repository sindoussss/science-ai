"""The controller loop: reason -> one tool call -> observe -> decide -> stop.

The model only chooses actions. Code validates them, runs tools in the
sandbox, writes graph nodes, applies the risk rules, runs the verifier and
walks the failure ladder. Every LLM call is counted; reuse paths make none.
"""
from __future__ import annotations

import json
import logging
import re
import threading
from dataclasses import dataclass, field
from typing import Any, Callable

import jsonschema

from sciai.config import Config
from sciai.controller import lookup
from sciai.controller.answer import render
from sciai.controller.provenance import numbers_in, unsourced
from sciai.domains.physics import quantities as Q
from sciai.domains.physics.assumptions import checklist, normalize_type
from sciai.graph.digest import build_digest
from sciai.graph.engine import GraphEngine, GraphRuleError
from sciai.graph.fingerprint import normalize_question, question_fingerprint, tool_fingerprint
from sciai.graph.model import (
    Domain,
    EdgeKind,
    Evidence,
    LadderStage,
    Layer,
    Node,
    NodeType,
    Status,
    TERMINAL_BAD,
)
from sciai.llm.actions import Action, ActionError, check_answer_template, parse_action
from sciai.llm.client import LLM, ContextOverflow, LLMUnavailable, estimate_tokens
from sciai.llm.roles import ROLES, retry_role
from sciai.tools.registry import ToolSpec
from sciai.tools.registry import get as get_tool
from sciai.tools.sandbox import Runner
from sciai.verify import ladder
from sciai.verify.risk_rules import assess
from sciai.verify.verifier import Verifier, VerifyOutcome

log = logging.getLogger(__name__)

STEP_ACTIONS = ("call_tool", "run_check", "finish", "ask_user")
PROMPT_MARGIN_TOKENS = 200

# confirm_root returns True (accept), False (reject), an edited statement, or a dict
# {"statement": edited text or None, "rejected": [assumption texts the user unticked]}.
ConfirmRoot = Callable[[Node], "bool | str | dict[str, Any]"]
GIVEN_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,30}$")
SYMBOL_ASSUMPTIONS = ("real", "positive", "negative", "nonnegative", "nonpositive", "integer", "nonzero")


@dataclass
class TaskResult:
    status: str  # answered | reused | escalated | needs_user | step_limit | stopped | error | rejected
    answer: str = ""
    final_node: str | None = None
    verified: bool = False
    llm_calls: int = 0
    steps: int = 0
    conflict: list[str] = field(default_factory=list)
    question: str | None = None
    detail: str = ""
    assumptions: list[str] = field(default_factory=list)  # confirmed modelling assumptions


@dataclass
class StepOutcome:
    lines: list[str]
    result: TaskResult | None = None
    next_role: str = "controller"


def _problem_sources(root: Node) -> dict[str, Any]:
    """What a problem node vouches for: the question, symbol assumptions and the givens found in
    the question. The goal is the model's own call and sources nothing by itself."""
    inputs = root.tool_inputs or {}
    bad = set(inputs.get("unsourced_givens") or [])
    return {"question": inputs.get("question", ""), "assumptions": inputs.get("assumptions") or {},
            "givens": {k: v for k, v in (inputs.get("givens") or {}).items() if k not in bad}}


class StepError(RuntimeError):
    """A valid action that could not be carried out (bad expression, tool error...)."""


class Controller:
    def __init__(self, engine: GraphEngine, runner: Runner, llm: LLM, cfg: Config,
                 confirm_root: ConfirmRoot | None = None) -> None:
        self.engine = engine
        self.runner = runner
        self.llm = llm
        self.cfg = cfg
        self.confirm_root = confirm_root
        self.verifier = Verifier(engine, runner, plausibility=cfg.physics.plausibility)
        self.llm_calls = 0
        self._notes: list[str] = []
        self._notes_lock = threading.Lock()
        self._stop = threading.Event()

    # ------------------------------------------------------------- public API
    def steer(self, text: str) -> None:
        with self._notes_lock:
            self._notes.append(text)

    def stop(self) -> None:
        self._stop.set()

    def run(self, question: str) -> TaskResult:
        self._stop.clear()
        self.llm_calls = 0
        sid = self.engine._require_session()
        repo = self.engine.repo
        repo.add_message(sid, "user", question)
        try:
            result = self._run(question)
        except LLMUnavailable as exc:
            result = TaskResult("error", detail=str(exc))
        result.llm_calls = self.llm_calls
        repo.add_message(sid, "controller", self._summary(result))
        repo.touch_session(sid)
        self.engine.bus.emit("task_done", result=result)
        return result

    def recheck(self, ref: str, note: str = "") -> str:
        """Pinned-comment re-check: deterministic, no model call."""
        node = self.engine.resolve(ref)
        h = self.engine.handle(node.id)
        if node.status in TERMINAL_BAD:
            return f"{h} is {node.status.value}; nothing to re-check."
        if node.status == Status.VERIFIED:
            self.engine.demote(node.id, automatic=True, reason=f"re-check requested: {note}" if note else "re-check")
        outcome = self.verifier.verify(self.engine.resolve(node.id))
        if outcome.status == "unavailable":
            return f"No independent check exists for {h}; it stays {self.engine.resolve(node.id).status.value}."
        if outcome.status == "failed":
            lines = self._after_verify(self.engine.resolve(node.id), outcome).lines
            return " ".join(lines)
        return f"Re-check of {h}: {outcome.status}."

    # ---------------------------------------------------------------- phases
    def _run(self, question: str) -> TaskResult:
        hit = lookup.verified_answer_for_question(self.engine.repo, question)
        if hit is not None:
            return self._reuse_answer(hit)

        root_or_result = self._formalize(question)
        if isinstance(root_or_result, TaskResult):
            return root_or_result
        root = root_or_result

        goal = (root.tool_inputs or {}).get("goal")
        observation = "Start: plan the first step."
        if goal:
            done, observation = self._try_goal(root, goal)
            if done is not None:
                return done
        return self._loop(root, observation)

    def _reuse_answer(self, final: Node) -> TaskResult:
        self.engine.link_existing(final.id)
        for dep in self.engine.dependencies(final.id):
            self.engine.link_existing(dep)
        return TaskResult("reused", answer=final.content, final_node=final.id, verified=True,
                          detail="answered from a verified result in the knowledge store",
                          assumptions=list((final.tool_inputs or {}).get("assumptions") or []))

    def _formalize(self, question: str) -> Node | TaskResult:
        role = ROLES["formalizer"]
        givens: dict[str, dict[str, Any]] = {}

        def validate(a: Action) -> None:
            goal = a.get("goal")
            if goal:
                self._validate_tool_call(goal.get("tool", ""), goal.get("args", {}), role.name)
            for name, kind in (a.get("assumptions") or {}).items():
                if kind not in SYMBOL_ASSUMPTIONS:
                    raise ActionError(f"assumption {kind!r} for {name!r} is not allowed")
            givens.clear()
            givens.update(self._normalize_givens(a.get("givens") or {}))

        user = f"QUESTION:\n{question}\n\nReply with one JSON formalize action."
        try:
            action = self._ask(role.name, user, ("formalize",), validate)
        except ActionError as exc:
            self._error_node("could not formalize the question", str(exc), [])
            return TaskResult("error", detail=f"formalization failed: {exc}")

        assumptions = action.get("assumptions") or {}
        goal = action.get("goal")
        if goal is not None and assumptions:
            spec = get_tool(goal["tool"])
            if "assumptions" in spec.schema.get("properties", {}):
                goal.setdefault("args", {}).setdefault("assumptions", assumptions)
        problem_type = normalize_type(action.get("problem_type"))
        items = checklist(problem_type, action.get("modelling_assumptions"))
        # A given is a source for later tool calls only if its number is in the question.
        in_question = numbers_in(question)
        unsourced_givens = sorted(k for k, v in givens.items() if not numbers_in(v["value"]) <= in_question)
        root = Node(
            session_id="", layer=Layer.REASONING, type=NodeType.PROBLEM, title="Problem",
            content=action["statement"], content_canonical=normalize_question(question),
            fingerprint=question_fingerprint(question),
            tool_inputs={"question": question, "assumptions": assumptions, "goal": goal,
                         "problem_type": problem_type, "givens": givens, "checklist": items,
                         "unsourced_givens": unsourced_givens},
            role=role.name,
            flags=[f"unsourced_givens: {', '.join(unsourced_givens)}"] if unsourced_givens else [],
        )
        self.engine.add_node(root)
        rejected: set[str] = set()
        if not self.cfg.controller.auto_confirm_root and self.confirm_root is not None:
            verdict = self.confirm_root(root)
            if verdict is False:
                return TaskResult("rejected", detail="the formalized problem was not confirmed")
            statement = verdict.get("statement") if isinstance(verdict, dict) else verdict
            if isinstance(verdict, dict):
                rejected = {str(t) for t in verdict.get("rejected") or []}
            if isinstance(statement, str) and statement.strip() and statement.strip() != root.content:
                root.content = statement.strip()
                root.tool_inputs = {**(root.tool_inputs or {}), "goal": None, "edited_by_user": True}
                self.engine.update(root)
        self._create_assumptions(root, items, rejected)
        return root

    @staticmethod
    def _normalize_givens(raw: dict[str, Any]) -> dict[str, dict[str, Any]]:
        out: dict[str, dict[str, Any]] = {}
        for name, obj in raw.items():
            if not GIVEN_NAME.match(name):
                raise ActionError(f"given name {name!r} must be a plain identifier like v0 or R1")
            try:
                Q.from_object(obj)
            except Q.QuantityError as exc:
                raise ActionError(f"given {name}: {exc}") from None
            out[name] = {"value": obj["value"], "unit": obj["unit"], **({"kind": obj["kind"]} if obj.get("kind") else {})}
        return out

    def _create_assumptions(self, root: Node, items: list[dict[str, Any]], rejected: set[str]) -> None:
        """Confirmed items become locked assumption nodes the problem depends on, so rejecting one
        later invalidates everything built on it. Unticked items are recorded as failed and unlinked."""
        domain = Domain.PHYSICS if (root.tool_inputs or {}).get("problem_type") else Domain.GENERAL
        confirmed: list[str] = []
        for item in items:
            ok = item["text"] not in rejected
            node = Node(session_id="", layer=Layer.REASONING, type=NodeType.ASSUMPTION, title=item["text"][:80],
                        content=item["text"], tool_inputs={"source": item["source"]},
                        status=Status.PROPOSED if ok else Status.FAILED, locked=ok, role="user")
            self.engine.add_node(node, tool_domain=domain)
            if ok:
                self.engine.add_edge(root.id, node.id, EdgeKind.DEPENDS_ON)
                confirmed.append(node.id)
        if items:
            root.tool_inputs = {**(root.tool_inputs or {}), "assumption_nodes": confirmed}
            self.engine.update(root)

    def _confirmed_assumptions(self, root: Node) -> list[str]:
        ids = (self.engine.resolve(root.id).tool_inputs or {}).get("assumption_nodes") or []
        return [self.engine.resolve(i).content for i in ids if self.engine.resolve(i).status not in TERMINAL_BAD]

    def _try_goal(self, root: Node, goal: dict[str, Any]) -> tuple[TaskResult | None, str]:
        """If the formalized goal is one tool call, reuse or run it without asking the model."""
        tool, args = goal["tool"], dict(goal.get("args") or {})
        fp, err = lookup.canonical_fingerprint(self.runner, tool, args)
        if fp is None:
            return None, f"The goal could not be parsed ({err}). Plan the first step."
        hit = lookup.verified_by_fingerprint(self.engine.repo, fp)
        if hit is not None:
            handle = self.engine.link_existing(hit.id)
            return self._finalize(root, "Answer: {{%s}}" % handle, [handle]), ""
        try:
            out = self._execute_tool(root, tool, args, [root.id], None, "controller", title="goal")
        except StepError as exc:
            return None, f"The goal tool call failed: {exc}. Plan another approach."
        if out.result is not None:
            return out.result, ""
        node = self._last_tool_node
        if node is not None and self.engine.resolve(node.id).status not in TERMINAL_BAD:
            node = self.engine.resolve(node.id)
            pre = self._ensure_verified([node])
            if pre.result is not None:
                return pre.result, ""
            if not pre.lines:
                handle = self.engine.handle(node.id)
                return self._finalize(root, "Answer: {{%s}}" % handle, [handle]), ""
            return None, "\n".join(out.lines + pre.lines)
        return None, "\n".join(out.lines)

    def _loop(self, root: Node, observation: str) -> TaskResult:
        role = "controller"
        invalid_streak = 0
        for step in range(1, self.cfg.controller.max_steps + 1):
            if self._stop.is_set():
                return TaskResult("stopped", steps=step - 1)
            self.engine.bus.emit("loop_state", step=step, role=role)
            try:
                action = self._ask(role, self._prompt(role, root, observation), STEP_ACTIONS,
                                   lambda a: self._validate_step(a, role))
            except ActionError as exc:
                invalid_streak += 1
                self._error_node("invalid controller output", str(exc), [root.id])
                if invalid_streak > self.cfg.controller.max_invalid_streak:
                    return TaskResult("error", steps=step, detail=f"repeated invalid output: {exc}")
                observation = f"Your previous reply was invalid and was discarded: {exc}"
                continue
            except ContextOverflow as exc:
                return TaskResult("error", steps=step, detail=str(exc))
            invalid_streak = 0
            if action.get("thought"):
                self.engine.bus.emit("thought", step=step, text=action["thought"])
            try:
                outcome = self._dispatch(root, action, role)
            except StepError as exc:
                self._error_node(f"{action.kind} failed", str(exc), [root.id])
                outcome = StepOutcome([f"That {action.kind} failed: {exc}"])
            if outcome.result is not None:
                outcome.result.steps = step
                return outcome.result
            observation = "\n".join(outcome.lines) or "(no output)"
            role = outcome.next_role
        return TaskResult("step_limit", steps=self.cfg.controller.max_steps,
                          detail="stopped at the step limit; the graph keeps everything so far")

    # -------------------------------------------------------------- LLM calls
    def _ask(self, role: str, user: str, allowed: tuple[str, ...],
             validate: Callable[[Action], None] | None = None) -> Action:
        """One call, and one retry if the output is invalid."""
        system = ROLES[role].system_prompt()
        from sciai.llm.actions import ACTION_SCHEMA

        last_err: ActionError | None = None
        for attempt in range(2):
            prompt = user if attempt == 0 else (
                f"{user}\n\nYOUR PREVIOUS REPLY WAS INVALID: {last_err}\nReply again with one valid JSON action.")
            self.llm_calls += 1
            reply = self.llm.chat(system, prompt, ACTION_SCHEMA)
            try:
                action = parse_action(reply.text, allowed)
                if validate is not None:
                    validate(action)
                return action
            except ActionError as exc:
                last_err = exc
        assert last_err is not None
        raise last_err

    def _prompt(self, role: str, root: Node, observation: str) -> str:
        system = ROLES[role].system_prompt()
        with self._notes_lock:
            notes, self._notes = self._notes, []
        notes_text = "".join(f"\nUSER NOTE: {n}" for n in notes)
        head = f"TASK: {root.content}\n\nGRAPH (handle [status] tool(args) = result <- depends_on):\n"
        tail = f"\n\nLAST RESULT:\n{observation}{notes_text}\n\nReply with one JSON action."
        budget = (self.cfg.model.num_ctx - self.cfg.model.num_predict - estimate_tokens(system)
                  - estimate_tokens(head) - estimate_tokens(tail) - PROMPT_MARGIN_TOKENS)
        if budget < 100:
            tail = tail[-2000:]
            budget = 100
        return head + build_digest(self.engine, budget) + tail

    # ------------------------------------------------------------- validation
    def _validate_tool_call(self, tool: str, args: dict[str, Any], role: str) -> ToolSpec:
        try:
            spec = get_tool(tool)
        except KeyError:
            raise ActionError(f"unknown tool {tool!r}") from None
        if spec.kind != "solver":
            raise ActionError(f"{tool} is a checker; use run_check instead")
        if not ROLES[role].allows(tool):
            raise ActionError(f"role {role} may not call {tool}")
        try:
            spec.validate(args)
        except jsonschema.ValidationError as exc:
            raise ActionError(f"bad args for {tool}: {exc.message}") from None
        return spec

    def _resolve_handle(self, ref: str) -> Node:
        if ref not in self.engine.by_handle:
            raise ActionError(f"unknown node {ref}")
        return self.engine.resolve(ref)

    def _validate_step(self, a: Action, role: str) -> None:
        if a.kind == "call_tool":
            spec = self._validate_tool_call(a["tool"], a["args"], role)
            if spec.node_arg is not None:
                node = self._resolve_handle(str(a["args"].get(spec.node_arg, "")))
                if node.status in TERMINAL_BAD:
                    raise ActionError(f"{a['args'][spec.node_arg]} is {node.status.value}")
            for ref in a.get("depends_on") or []:
                node = self._resolve_handle(ref)
                if node.status in TERMINAL_BAD:
                    raise ActionError(f"{ref} is {node.status.value}; build on verified or proposed nodes only")
                if node.risk.get("pending_checks"):
                    raise ActionError(f"{ref} must be checked first (run_check)")
            if a.get("replaces"):
                self._resolve_handle(a["replaces"])
        elif a.kind == "run_check":
            node = self._resolve_handle(a["node"])
            offered = {p.method for p, _ in self.verifier.offered(node)} | {
                p.checker for p, _ in self.verifier.offered(node)}
            if a.get("check") and a["check"] not in offered:
                raise ActionError(f"check {a['check']!r} is not offered for {a['node']}; offered: {sorted(offered)}")
        elif a.kind == "finish":
            check_answer_template(a["answer_template"], a["answer_nodes"])
            for ref in a["answer_nodes"]:
                node = self._resolve_handle(ref)
                if node.status in TERMINAL_BAD:
                    raise ActionError(f"{ref} is {node.status.value} and cannot be in the answer")

    # ---------------------------------------------------------------- actions
    def _dispatch(self, root: Node, a: Action, role: str) -> StepOutcome:
        if a.kind == "call_tool":
            deps = [self.engine.resolve(r).id for r in (a.get("depends_on") or [])] or [root.id]
            replaces = self.engine.resolve(a["replaces"]).id if a.get("replaces") else None
            return self._execute_tool(root, a["tool"], dict(a["args"]), deps, replaces, role,
                                      title=a.get("title"))
        if a.kind == "run_check":
            node = self.engine.resolve(a["node"])
            node.risk.pop("pending_checks", None)
            self.engine.update(node)
            outcome = self.verifier.verify(node, only_method=a.get("check"))
            return self._after_verify(self.engine.resolve(node.id), outcome)
        if a.kind == "finish":
            nodes = [self.engine.resolve(r) for r in a["answer_nodes"]]
            pre = self._ensure_verified(nodes)
            if pre.result is not None or pre.lines:
                return pre
            return StepOutcome([], self._finalize(root, a["answer_template"], a["answer_nodes"]))
        if a.kind == "ask_user":
            return StepOutcome([], TaskResult("needs_user", question=a["question"]))
        raise StepError(f"unhandled action {a.kind}")

    _last_tool_node: Node | None = None

    def _execute_tool(self, root: Node, tool: str, args: dict[str, Any], deps: list[str],
                      replaces: str | None, role: str, title: str | None = None) -> StepOutcome:
        spec = get_tool(tool)
        self._last_tool_node = None
        root_assumptions = (root.tool_inputs or {}).get("assumptions") or {}
        if root_assumptions and "assumptions" in spec.schema.get("properties", {}):
            args.setdefault("assumptions", root_assumptions)

        run_args = args
        if spec.node_arg is not None:
            # the handle becomes the node's id (handles are per session) and its result is the data
            ref = self.engine.resolve(str(args.get(spec.node_arg, "")))
            if ref.status in TERMINAL_BAD:
                raise StepError(f"{self.engine.handle(ref.id)} is {ref.status.value}")
            args[spec.node_arg] = ref.id
            deps = list(dict.fromkeys([*deps, ref.id]))
            run_args = {**args, "data": ref.result}

        fp, err = lookup.canonical_fingerprint(self.runner, tool, run_args)
        if fp is None:
            raise StepError(err or "could not parse the arguments")

        if replaces is not None:
            prev = self.engine.repo.lineage(self.engine.resolve(replaces).lineage_id)
            if any(v.fingerprint == fp for v in prev):
                raise StepError("a retry must use a different method or tool than the failed attempt")
        else:
            hit = lookup.verified_by_fingerprint(self.engine.repo, fp)
            if hit is not None:
                handle = self.engine.link_existing(hit.id)
                self._last_tool_node = hit
                return StepOutcome([f"Reused verified result {handle} = {hit.display_result()}"])
        hint = lookup.hint_by_fingerprint(self.engine.repo, fp)

        sources: list[Any] = [root.content, _problem_sources(root)]
        for d in deps:
            for nid in [d, *self.engine.ancestors(d)]:
                n = self.engine.resolve(nid)
                if n.type == NodeType.PROBLEM:
                    sources += [n.content, _problem_sources(n)]
                else:
                    sources += [n.content if n.type == NodeType.ASSUMPTION else "", n.tool_inputs, n.result]
        scanned = {k: v for k, v in args.items() if k != spec.node_arg}  # a node id is not a number
        missing = unsourced(scanned, sources, self.cfg.risk.structural_int_max)

        outcome = self.runner.run(tool, run_args, self.cfg.tools.timeout)
        run_id = self.engine.repo.log_tool_run(self.engine.session_id or "", None, tool, args,
                                               outcome.value if outcome.ok else None, outcome.error,
                                               outcome.duration_ms)
        if not outcome.ok:
            err_node = self._error_node(f"{tool} error", outcome.error or "tool failed", deps,
                                        tool=tool, args=args)
            self.engine.repo.attach_run(run_id, err_node.id)
            raise StepError(outcome.error or "tool failed")

        entity = self._entity_for(root, spec, args)
        if entity is not None:
            deps = [*deps, entity.id]
        node = Node(
            session_id="", layer=Layer.PLOT if outcome.value["result"].get("kind") == "plotspec" else Layer.REASONING,
            type=NodeType.PLOT if outcome.value["result"].get("kind") == "plotspec" else NodeType.TOOL_RESULT,
            title=title or tool, content=f"{tool}({json.dumps(args, sort_keys=True)})",
            content_canonical=fp, fingerprint=fp, tool_name=tool, tool_inputs=args,
            result=outcome.value["result"], confidence=outcome.value.get("confidence"), role=role,
            flags=[f"unsourced_numbers: {', '.join(missing)}"] if missing else [],
        )
        try:
            self.engine.add_node(node, deps, tool_domain=spec.domain, replaces=replaces)
        except GraphRuleError as exc:
            raise StepError(str(exc)) from None
        self.engine.repo.attach_run(run_id, node.id)
        self._last_tool_node = node
        h = self.engine.handle(node.id)
        lines = [f"{h} = {node.display_result()}"]
        if missing:
            lines.append(f"{h} uses numbers not found in the problem or its inputs ({', '.join(missing)}); "
                         "it will be checked.")
        if hint is not None and hint.result != node.result:
            lines.append(f"Note: an unverified earlier result for the same call differs ({hint.display_result()}).")

        step = self._assess_and_check(node, user_facing=False)
        lines += step.lines
        if step.result is not None:
            return StepOutcome(lines, step.result)
        sweep = self._stakes_sweep(node)
        lines += sweep.lines
        next_role = step.next_role if step.next_role != "controller" else sweep.next_role
        return StepOutcome(lines, sweep.result, next_role)

    def _entity_for(self, root: Node, spec: ToolSpec, args: dict[str, Any]) -> Node | None:
        """The domain entity a tool argument holds (a circuit netlist), as one node per distinct
        entity in the session. Results depend on it, so a new version of it cascades to them."""
        if spec.entity_arg is None or spec.entity_arg not in args:
            return None
        value = args[spec.entity_arg]
        out = self.runner.canonicalize(spec.name, {spec.entity_arg: value})
        if not out.ok:
            return None
        fp = tool_fingerprint(f"entity.{spec.entity_arg}", out.value["canonical"].get(spec.entity_arg))
        for n in self.engine.nodes.values():
            if (n.type == NodeType.ENTITY and n.fingerprint == fp and n.session_id == self.engine.session_id
                    and n.status not in TERMINAL_BAD):
                return n
        describe = spec.describe_entity or (lambda v: json.dumps(v, sort_keys=True))
        entity = Node(session_id="", layer=Layer.DOMAIN, type=NodeType.ENTITY, title=spec.entity_arg,
                      content=describe(value), content_canonical=fp, fingerprint=fp,
                      tool_inputs={"arg": spec.entity_arg, "value": value}, role="controller")
        self.engine.add_node(entity, [root.id], tool_domain=spec.domain)
        return entity

    # ----------------------------------------------------------- verification
    def _assess_and_check(self, node: Node, user_facing: bool, extra_dependents: int = 0,
                          auto: bool = False) -> StepOutcome:
        """Apply the risk rules; run the check if one is required. With several offered
        checks the controller chooses, unless ``auto`` (finishing) picks the cheapest."""
        node = self.engine.resolve(node.id)
        spec = get_tool(node.tool_name) if node.tool_name else None
        report = assess(self.engine, node, spec, self.cfg.risk, user_facing=user_facing,
                        extra_dependents=extra_dependents, plausibility=self.cfg.physics.plausibility)
        node.risk = {**node.risk, "fired": report.fired}
        self.engine.update(node)
        h = self.engine.handle(node.id)
        if not report.required:
            return StepOutcome([])
        rules = ", ".join(sorted(set(report.rules())))
        plans = self.verifier.offered(node)
        if not plans:
            node.risk["unverifiable"] = True
            self.engine.update(node)
            return StepOutcome([f"{h} is risky ({rules}) but no independent check exists; it stays unverified."])
        # Required checks always run, so the model only chooses among the optional ones.
        optional = [p for p, _ in plans if not p.required]
        if len(optional) > 1 and not (user_facing or auto):
            node.risk["pending_checks"] = [p.method for p in optional]
            self.engine.update(node)
            return StepOutcome([f"{h} needs a check ({rules}). Offered checks: "
                                f"{', '.join(p.method for p in optional)}. Choose one with run_check."])
        outcome = self.verifier.verify(node)
        return self._after_verify(self.engine.resolve(node.id), outcome, rules)

    def _after_verify(self, node: Node, outcome: VerifyOutcome, rules: str = "") -> StepOutcome:
        h = self.engine.handle(node.id)
        why = f" (risk: {rules})" if rules else ""
        if outcome.status == "verified":
            return StepOutcome([f"{h} verified by {outcome.runs[-1].plan.method}{why}."])
        if outcome.status in ("inconclusive", "unavailable"):
            return StepOutcome([f"{h} could not be checked conclusively; it stays unverified{why}."])
        decision = ladder.decide(self.engine, node)
        if decision.stage == LadderStage.ESCALATED:
            return StepOutcome([decision.message], TaskResult(
                "escalated", conflict=decision.conflict, detail=decision.message))
        failed = outcome.failed_run
        detail = ""
        if failed is not None:
            d = {k: v for k, v in failed.detail.items() if k not in ("kind",)}
            detail = f" Check said: {json.dumps(d, default=str)[:300]}"
        next_role = "controller"
        if decision.stage == LadderStage.RETRIED and node.tool_name:
            next_role = retry_role(node.tool_name)
        return StepOutcome([decision.message + detail], next_role=next_role)

    def _stakes_sweep(self, new_node: Node) -> StepOutcome:
        """A new dependent can push an unchecked ancestor over the stakes threshold."""
        lines: list[str] = []
        for anc_id in self.engine.ancestors(new_node.id):
            anc = self.engine.resolve(anc_id)
            if (anc.type != NodeType.TOOL_RESULT or anc.status != Status.PROPOSED
                    or anc.risk.get("unverifiable") or anc.risk.get("pending_checks")
                    or self.engine.repo.evidence_for(anc.id)):
                continue
            step = self._assess_and_check(anc, user_facing=False)
            lines += step.lines
            if step.result is not None or step.next_role != "controller":
                return StepOutcome(lines, step.result, step.next_role)
        return StepOutcome(lines)

    def _ensure_verified(self, nodes: list[Node]) -> StepOutcome:
        """Stakes rule at the answer: the answer nodes are user-facing, and every
        unchecked ancestor is re-assessed counting the final node about to depend on it.
        Ancestors go first (oldest first) so a bad intermediate is caught at its source."""
        answer_ids = [n.id for n in nodes]
        ancestor_ids: list[str] = []
        for nid in answer_ids:
            for a in self.engine.ancestors(nid):
                if a not in ancestor_ids and a not in answer_ids:
                    ancestor_ids.append(a)
        ancestors = sorted((self.engine.resolve(a) for a in ancestor_ids), key=lambda n: n.created_at)
        queue = [(n, False, 1) for n in ancestors] + [(self.engine.resolve(i), True, 0) for i in answer_ids]
        for node, user_facing, extra in queue:
            node = self.engine.resolve(node.id)
            if (node.type != NodeType.TOOL_RESULT or node.status != Status.PROPOSED
                    or node.risk.get("unverifiable") or self.engine.repo.evidence_for(node.id)):
                continue
            node.risk.pop("pending_checks", None)
            step = self._assess_and_check(node, user_facing=user_facing, extra_dependents=extra, auto=True)
            if step.result is not None:
                return step
            if self.engine.resolve(node.id).status in TERMINAL_BAD:
                return StepOutcome(step.lines, next_role=step.next_role)
        return StepOutcome([])

    # ------------------------------------------------------------------ final
    def _finalize(self, root: Node, template: str, handles: list[str]) -> TaskResult:
        nodes = [self.engine.resolve(h) for h in handles]
        all_verified = all(n.status == Status.VERIFIED for n in nodes)
        assumed = self._confirmed_assumptions(root)
        final = Node(session_id="", layer=Layer.REASONING, type=NodeType.FINAL, title="Answer",
                     content=render(self.engine, template),
                     tool_inputs={"template": template, "answer_nodes": [n.id for n in nodes],
                                  "assumptions": assumed})
        if not all_verified:
            final.flags = ["unverified inputs"]
        dep_ids = list(dict.fromkeys([n.id for n in nodes] + [root.id]))
        self.engine.add_node(final, dep_ids)
        if all_verified:
            self.engine.repo.add_evidence(Evidence(
                node_id=final.id, method="inputs_verified", tool_name="controller.render",
                inputs={"template": template}, outcome="pass",
                detail={"answer_nodes": [n.id for n in nodes]}))
            self.engine.set_status(final.id, Status.VERIFIED, "all answer values verified")
        return TaskResult("answered", answer=final.content, final_node=final.id, verified=all_verified,
                          assumptions=assumed)

    def _error_node(self, title: str, message: str, deps: list[str], tool: str | None = None,
                    args: dict[str, Any] | None = None) -> Node:
        node = Node(session_id="", layer=Layer.REASONING, type=NodeType.ERROR, title=title,
                    content=message, status=Status.FAILED, tool_name=tool, tool_inputs=args)
        deps = [d for d in deps if self.engine.resolve(d).status not in TERMINAL_BAD]
        self.engine.add_node(node, deps)
        return node

    @staticmethod
    def _summary(r: TaskResult) -> str:
        if r.status in ("answered", "reused"):
            tag = "" if r.verified else " (not fully verified)"
            return f"{r.answer}{tag}"
        if r.status == "needs_user":
            return r.question or "I need more information."
        return f"[{r.status}] {r.detail}"
