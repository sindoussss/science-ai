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
from sciai.domains.data.datasets import load_args
from sciai.domains.data.report import diagnostic_text
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
from sciai.store.repository import DatasetRecord
from sciai.verify.risk_rules import assess, family_members, family_of
from sciai.verify.verifier import Verifier, VerifyOutcome

log = logging.getLogger(__name__)

STEP_ACTIONS = ("call_tool", "run_check", "finish", "ask_user")
PROMPT_MARGIN_TOKENS = 200

# confirm_root returns True (accept), False (reject), an edited statement, or a dict
# {"statement": edited text or None, "rejected": [assumption texts the user unticked]}.
ConfirmRoot = Callable[[Node], "bool | str | dict[str, Any]"]
GIVEN_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,30}$")
HANDLE = re.compile(r"^n[0-9]+$")
MAX_DATASETS_IN_PROMPT = 8
MAX_SCHEMA_CHARS = 1500
SIGNIFICANT = re.compile(r"\bsignifican", re.IGNORECASE)
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
        if hit is not None and not self._stale_data(hit):
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

    def _stale_data(self, final: Node) -> bool:
        """An answer read a dataset that has since been re-imported with different contents
        (the same name, other bytes): asking again must use the current file."""
        repo = self.engine.repo
        for nid in self.engine.ancestors(final.id):
            n = self.engine.resolve(nid)
            if n.tool_name == "data.load" and n.tool_inputs:
                rec = repo.dataset_by_name(n.tool_inputs.get("name", ""))
                if rec is None or rec.sha256 != n.tool_inputs.get("sha256"):
                    return True
        return False

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
                spec = self._validate_tool_call(goal.get("tool", ""), goal.get("args", {}), role.name)
                self._validate_data_args(spec, goal.get("args", {}), question)
            for name, kind in (a.get("assumptions") or {}).items():
                if kind not in SYMBOL_ASSUMPTIONS:
                    raise ActionError(f"assumption {kind!r} for {name!r} is not allowed")
            givens.clear()
            givens.update(self._normalize_givens(a.get("givens") or {}))

        data = self._datasets_text(question)
        user = f"QUESTION:\n{question}\n\n" + (f"DATASETS (imported files: name, rows, columns):\n{data}\n\n"
                                                if data else "") + "Reply with one JSON formalize action."
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
        try:
            run_args, _ = self._prepare_dataset(get_tool(tool), args)
        except StepError as exc:
            return None, f"The goal could not be run ({exc}). Plan the first step."
        fp, err = lookup.canonical_fingerprint(self.runner, tool, run_args)
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
        data = self._datasets_text((root.tool_inputs or {}).get("question", root.content))
        head = f"TASK: {root.content}\n\n" + (
            f"DATASETS (imported files; use the name or a dataset node handle):\n{data}\n\n" if data else "") + \
            "GRAPH (handle [status] tool(args) = result <- depends_on):\n"
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
            root = self.engine.root()
            self._validate_data_args(spec, a["args"], (root.tool_inputs or {}).get("question", "") if root else "")
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
            kinds = set()
            for ref in a["answer_nodes"]:
                node = self._resolve_handle(ref)
                if node.status in TERMINAL_BAD:
                    raise ActionError(f"{ref} is {node.status.value} and cannot be in the answer")
                kinds.add((node.result or {}).get("kind"))
            if SIGNIFICANT.search(a["answer_template"]) and not kinds & {"stats", "adjusted"}:
                raise ActionError('the word "significant" needs the test result node in answer_nodes')

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

        run_args, dataset_id = self._prepare_dataset(spec, args)
        if dataset_id is not None:
            deps = list(dict.fromkeys([*deps, dataset_id]))
        if spec.node_arg is not None:
            # the handle becomes the node's id (handles are per session) and its result is the data
            ref = self.engine.resolve(str(args.get(spec.node_arg, "")))
            if ref.status in TERMINAL_BAD:
                raise StepError(f"{self.engine.handle(ref.id)} is {ref.status.value}")
            args[spec.node_arg] = ref.id
            deps = list(dict.fromkeys([*deps, ref.id]))
            run_args = {**run_args, spec.node_arg: ref.id, "data": ref.result}

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
                for dep in self.engine.dependencies(hit.id):  # its dataset and test assumptions
                    if self.engine.resolve(dep).type in (NodeType.ENTITY, NodeType.ASSUMPTION) \
                            and dep not in self.engine.nodes:
                        self.engine.link_existing(dep)
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
        if dataset_id is not None:
            sources.append({"alpha": self.cfg.data.alpha})  # the configured significance level
        # a node id is not a number
        scanned = {k: v for k, v in args.items() if k not in (spec.node_arg, spec.dataset_arg)}
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
        if node.result.get("kind") == "stats":
            self._diagnostic_assumptions(node)
        h = self.engine.handle(node.id)
        lines = [f"{h} = {node.display_result()}"]
        lines += self._alternative_offer(node)
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
        template, handles, family_note = self._family_rule(root, template, list(handles))
        nodes = [self.engine.resolve(h) for h in handles]
        all_verified = all(n.status == Status.VERIFIED for n in nodes)
        assumed = self._confirmed_assumptions(root)
        tested, doubtful = self._test_assumptions(nodes)
        assumed += [t for t in tested if t not in assumed]
        content = render(self.engine, template)
        if doubtful:
            content += "\n" + "\n".join(f"Caution: {d}." for d in doubtful)
        if family_note:
            content += f"\n{family_note}"
        final = Node(session_id="", layer=Layer.REASONING, type=NodeType.FINAL, title="Answer",
                     content=content,
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

    # ------------------------------------------------------------------ data
    def _datasets_text(self, question: str) -> str:
        """The imported datasets the model may use, by schema (names, types, units, missing counts,
        category levels): never row values. Datasets the question names come first."""
        repo = self.engine.repo
        named = repo.datasets_named_in(question)
        seen = {r.id for r in named}
        recs = named + [r for r in repo.current_datasets(MAX_DATASETS_IN_PROMPT) if r.id not in seen]
        lines = []
        for rec in recs[:MAX_DATASETS_IN_PROMPT]:
            text = rec.schema_text()
            lines.append(text if len(text) <= MAX_SCHEMA_CHARS else text[: MAX_SCHEMA_CHARS - 4] + " ...")
        return "\n".join(lines)

    def _dataset_record(self, name: str) -> DatasetRecord | None:
        repo = self.engine.repo
        rec = repo.dataset_by_name(name)
        if rec is not None:
            return rec
        stem = [r for r in repo.current_datasets(500) if r.name.rsplit(".", 1)[0].lower() == name.strip().lower()]
        return stem[0] if len(stem) == 1 else None

    def _validate_data_args(self, spec: ToolSpec, args: dict[str, Any], question: str) -> None:
        """A dataset is named, never passed as data; alpha comes from the question or the config."""
        if spec.dataset_arg is not None:
            ref = args.get(spec.dataset_arg)
            if not isinstance(ref, str):
                raise ActionError(f"{spec.dataset_arg} must be a dataset node handle like n3 or an imported file "
                                  "name; never pass data")
            if HANDLE.match(ref.strip()):
                if ref.strip() in self.engine.by_handle:
                    node = self.engine.resolve(ref.strip())
                    if (node.result or {}).get("kind") != "dataset":
                        raise ActionError(f"{ref} is not a dataset")
                    if node.status in TERMINAL_BAD:
                        raise ActionError(f"{ref} is {node.status.value}")
                else:
                    raise ActionError(f"unknown node {ref}")
            elif self._dataset_record(ref) is None:
                names = [r.name for r in self.engine.repo.current_datasets(20)]
                raise ActionError(f"no imported dataset is named {ref!r}; imported: {', '.join(names) or 'none'}")
        alpha = args.get("alpha")
        if alpha is not None and "alpha" in spec.schema.get("properties", {}) and \
                not numbers_in(alpha) <= numbers_in(question):
            raise ActionError(f"alpha {alpha} is not in the question; leave alpha out to use "
                              f"{self.cfg.data.alpha}")

    def _dataset_node(self, rec: DatasetRecord) -> Node:
        """The dataset as a node: reused if this session or the store already has it read and
        verified, else read (data.load) and verified by an independent re-read. It depends on
        nothing, so it is shared by every question that uses the file."""
        args = load_args(rec, self.cfg.data)
        fp, err = lookup.canonical_fingerprint(self.runner, "data.load", args)
        if fp is None:
            raise StepError(err or f"{rec.name} could not be read")
        for n in self.engine.nodes.values():
            if n.fingerprint == fp and n.type == NodeType.ENTITY and n.status not in TERMINAL_BAD:
                return n
        hit = lookup.verified_by_fingerprint(self.engine.repo, fp)
        if hit is not None:
            self.engine.link_existing(hit.id)
            return self.engine.resolve(hit.id)
        out = self.runner.run("data.load", args, self.cfg.tools.timeout)
        run_id = self.engine.repo.log_tool_run(self.engine.session_id or "", None, "data.load", args,
                                               out.value if out.ok else None, out.error, out.duration_ms)
        if not out.ok:
            raise StepError(f"{rec.name} could not be read: {out.error}")
        node = Node(session_id="", layer=Layer.DOMAIN, type=NodeType.ENTITY, title=f"dataset {rec.name}",
                    content=rec.schema_text(), content_canonical=fp, fingerprint=fp, tool_name="data.load",
                    tool_inputs=args, result=out.value["result"], role="controller")
        self.engine.add_node(node, [], tool_domain=Domain.DATA)
        self.engine.repo.attach_run(run_id, node.id)
        outcome = self.verifier.verify(node)
        if outcome.status != "verified":
            raise StepError(f"{rec.name} did not read the same way twice; re-import it ({outcome.status})")
        return self.engine.resolve(node.id)

    def _prepare_dataset(self, spec: ToolSpec, args: dict[str, Any]) -> tuple[dict[str, Any], str | None]:
        """For a data tool: store the dataset node's id in ``args`` (fingerprints use the data's
        key, not the handle), fill in the configured alpha, and return the arguments the tool
        runs with (the node's descriptor in place of the name) and the dataset node's id."""
        if spec.dataset_arg is None or spec.dataset_arg not in args:
            return args, None
        ref = str(args[spec.dataset_arg]).strip()
        if ref in self.engine.by_handle or ref in self.engine.nodes:
            node = self.engine.resolve(ref)
            if (node.result or {}).get("kind") != "dataset" or node.status in TERMINAL_BAD:
                raise StepError(f"{self.engine.handle(node.id)} is not a usable dataset")
        else:
            rec = self._dataset_record(ref)
            if rec is None:
                raise StepError(f"no imported dataset is named {ref!r}")
            node = self._dataset_node(rec)
        args[spec.dataset_arg] = node.id
        if "alpha" in spec.schema.get("properties", {}) and args.get("alpha") is None:
            args["alpha"] = self.cfg.data.alpha
        return {**args, spec.dataset_arg: node.result}, node.id

    def _diagnostic_assumptions(self, node: Node) -> None:
        """Each assumption a test relies on becomes an assumption node the test depends on, with
        its diagnostic. A diagnostic that speaks against it marks it doubtful, not failed: failing
        it would invalidate the test and everything after it, including a robust alternative."""
        for d in (node.result or {}).get("diagnostics") or []:
            doubtful = d.get("ok") is False
            a = Node(session_id="", layer=Layer.REASONING, type=NodeType.ASSUMPTION, title=d["label"][:80],
                     content=diagnostic_text(d), tool_inputs={"source": "diagnostic", "diagnostic": d},
                     locked=True, role="verifier", flags=[f"doubtful: {d.get('note', '')}"] if doubtful else [])
            self.engine.add_node(a, tool_domain=Domain.DATA)
            self.engine.add_edge(node.id, a.id, EdgeKind.DEPENDS_ON)

    def _alternative_offer(self, node: Node) -> list[str]:
        """A doubtful assumption does not block the answer; the model is told the robust
        alternative so it can run that test as well."""
        r = node.result or {}
        bad = [d for d in r.get("diagnostics") or [] if d.get("ok") is False]
        if r.get("kind") != "stats" or not bad:
            return []
        h = self.engine.handle(node.id)
        what = "; ".join(f"{d['label']} is doubtful ({d.get('note', '')})" for d in bad)
        alternative = {"welch_t": "stats.mannwhitney", "student_t": "stats.ttest with equal_var false, or "
                       "stats.mannwhitney", "paired_t": "a sign-flip permutation is its check; report it with "
                       "the caution", "one_sample_t": "report it with the caution", "anova": "stats.kruskal",
                       "pearson": "stats.correlation with method spearman",
                       "chi2": "combine sparse levels with data.derive or data.filter"}.get(r.get("test"))
        return [f"For {h}: {what}." + (f" The alternative is {alternative}." if alternative else "")]

    def _test_assumptions(self, nodes: list[Node]) -> tuple[list[str], list[str]]:
        """(assumption texts, doubtful ones) behind the answer's tests."""
        texts, doubtful = [], []
        for n in nodes:
            for a in sorted((self.engine.resolve(d) for d in self.engine.dependencies(n.id)),
                            key=lambda x: x.created_at):
                if a.type != NodeType.ASSUMPTION or (a.tool_inputs or {}).get("source") != "diagnostic" \
                        or a.status in TERMINAL_BAD:
                    continue
                if a.content not in texts:
                    texts.append(a.content)
                if any(f.startswith("doubtful") for f in a.flags) and a.content not in doubtful:
                    doubtful.append(a.content)
        return texts, doubtful

    def _family_rule(self, root: Node, template: str, handles: list[str]) -> tuple[str, list[str], str]:
        """Several tests on the same data: run the Holm adjustment over all of them and add it
        to the answer. Returns the template, the answer handles and a note when it could not run."""
        nodes = [self.engine.resolve(h) for h in handles]
        if any((n.result or {}).get("kind") == "adjusted" for n in nodes):
            return template, handles, ""
        note = ""
        for key in dict.fromkeys(family_of(n) for n in nodes if family_of(n)):
            members = family_members(self.engine, key)
            if len(members) < 2:
                continue
            alphas = {float(m.result["alpha"]) for m in members}
            args = {"p": [float(m.result["p"]) for m in members],
                    "labels": [f"{self.engine.handle(m.id)} {m.result.get('label', m.result['test']).split(' of ')[0]}"
                               for m in members],
                    "method": "holm", "alpha": alphas.pop() if len(alphas) == 1 else self.cfg.data.alpha,
                    "family": (members[0].result.get("dataset") or {}).get("name") or "the same data"}
            try:
                self._execute_tool(root, "stats.adjust", args, [m.id for m in members], None, "controller",
                                   title="Holm adjustment")
            except StepError as exc:
                note = f"Caution: the multiple-testing adjustment could not be computed ({exc})."
                continue
            adj = self._last_tool_node
            if adj is None or self.engine.resolve(adj.id).status in TERMINAL_BAD:
                note = "Caution: the multiple-testing adjustment failed its check."
                continue
            h = self.engine.handle(adj.id)
            template += "\nCorrected for multiple tests: {{%s}}" % h
            handles.append(h)
        return template, handles, note

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
