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
from sciai.controller import lookup, recipes
from sciai.controller.answer import render
from sciai.controller.provenance import numbers_in, unsourced
from sciai.domains.chem import decline as chem_decline
from sciai.domains.chem import routing as chem_routing
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
from sciai.llm.actions import (
    Action,
    ActionError,
    check_answer_template,
    check_question,
    parse_action,
)
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


class Declined(Exception):
    """A request nothing registered can serve. Carries the refusal, not an error message."""

    def __init__(self, refusal: chem_decline.Decline) -> None:
        super().__init__(refusal.message)
        self.refusal = refusal


class Unroutable(ActionError):
    """The route cannot answer this question, and the reply that said so is the model's second.

    Both subclasses are ``ActionError``, so the first one costs the one re-prompt the controller
    already makes with the message as written -- which is where a model that chose the wrong
    recipe or copied an example's numbers gets its chance to correct itself. A second failure
    means no reply is going to work, and the question is answered out of scope rather than
    computed from values it does not contain.
    """

    def __init__(self, message: str, detail: str) -> None:
        super().__init__(message)
        self.detail = detail


class UnsourcedSlots(Unroutable):
    """Slots the model filled with values the question does not contain."""

    def __init__(self, recipe: str, slots: tuple[str, ...], shown: str) -> None:
        names = ", ".join(slots)
        super().__init__(
            f"{names} is not in the question ({shown}). Copy only numbers and structures the "
            f"question itself contains, never a value from an example; if {recipe} does not fit "
            f'this question, reply with recipe "none" or the recipe that does',
            f"the {recipe} recipe was filled with {shown}, which the question does not contain")
        self.recipe, self.slots = recipe, slots


class WrongRoute(Unroutable):
    """A molecule question routed to a recipe that is not about molecules."""

    def __init__(self, route: str, offered: tuple[str, ...]) -> None:
        super().__init__(
            f"this question is about a molecule, so {route} cannot answer it. Use one of "
            f'{", ".join(offered)}, or "molecule_question" with an "operation", or "none"',
            f"a molecule question cannot be answered by {route}")
        self.route = route


@dataclass
class TaskResult:
    status: str  # answered | reused | escalated | needs_user | step_limit | stopped | error
                 #        | rejected | declined | out_of_scope
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
        # Capability, in code, before anything else. A question asking for an operation no tool
        # performs is declined here rather than routed, so a synthesis route or a dose comes
        # back as the one sentence that says why not, never as a formalization error and never
        # after a model call. ``decline_for_question`` still asks the registry, so this refuses
        # exactly what the system cannot do.
        refusal = chem_decline.decline_for_question(question)
        if refusal is not None:
            return self._decline(refusal)

        hit = lookup.verified_answer_for_question(self.engine.repo, question)
        if hit is not None and not self._stale_data(hit):
            return self._reuse_answer(hit)

        self._pending_plan = None
        root_or_result = self._formalize(question)
        if isinstance(root_or_result, TaskResult):
            return root_or_result
        root = root_or_result
        plan, self._pending_plan = self._pending_plan, None
        if plan is not None:
            return self._run_recipe(root, plan)

        goal = (root.tool_inputs or {}).get("goal")
        observation = "Start: plan the first step."
        self._next_role = "controller"
        if goal:
            done, observation = self._try_goal(root, goal)
            if done is not None:
                return done
        return self._loop(root, observation, self._next_role)

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
        """One model call: which recipe, and its slots.

        This is the router. The model's only planning freedom is the ``recipe`` enum, so a
        question that fits a recipe is answered by code from here on and costs this one call.
        Three enum values are not recipes: ``none`` says the question is out of scope, and the
        two delegated routes hand it to the dataset and chemistry paths, which do their own
        validation below exactly as before.
        """
        role = ROLES["formalizer"]
        givens: dict[str, dict[str, Any]] = {}
        plan: list[recipes.Plan] = []

        def validate(a: Action) -> None:
            route = a["recipe"]
            plan.clear()
            givens.clear()
            if route == recipes.NONE:
                return
            if route in recipes.RECIPES:
                # A molecule question may only become a chemistry recipe, the molecule path or
                # out of scope. The question's own text decides that, in code: the live run
                # had the router send "the molecular weight of caffeine" to photon_energy and
                # the model fill it from that recipe's worked example.
                if route not in recipes.CHEM_RECIPES and chem_routing.looks_chemical(question):
                    raise WrongRoute(route, recipes.CHEM_RECIPES)
                try:
                    built = recipes.RECIPES[route].plan(a.get("slots") or {}, question)
                except recipes.SlotError as exc:
                    # A slot holding the wrong kind of thing is the one failure worth a retry:
                    # the model wrote something, and the message says what the slot takes.
                    raise ActionError(str(exc)) from None
                if built.unsourced:
                    # Nothing is computed from a value the question does not contain, not even
                    # one that matches an example. This is raised before any tool runs.
                    raise UnsourcedSlots(route, built.unsourced,
                                         ", ".join(f"{s} = {built.values[s]}"
                                                   for s in built.unsourced))
                plan.append(built)
                return
            goal = a.get("goal")
            # Capability first, and before the goal is validated: a request nothing serves is
            # declined once, rather than sent back to the model as an invalid action.
            refusal = self._refusal(a.get("operation"), goal, question)
            if refusal is not None:
                raise Declined(refusal)
            if goal:
                spec = self._validate_tool_call(goal.get("tool", ""), goal.get("args", {}), role.name)
                self._validate_data_args(spec, goal.get("args", {}), question)
            for name, kind in (a.get("assumptions") or {}).items():
                if kind not in SYMBOL_ASSUMPTIONS:
                    raise ActionError(f"assumption {kind!r} for {name!r} is not allowed")
            givens.update(self._normalize_givens(a.get("givens") or {}))

        data = self._datasets_text(question)
        user = f"QUESTION:\n{question}\n\n" + (f"DATASETS (imported files: name, rows, columns):\n{data}\n\n"
                                                if data else "") + "Reply with one JSON formalize action."
        try:
            action = self._ask(role.name, user, ("formalize",), validate)
        except Declined as exc:
            return self._decline(exc.refusal)
        except recipes.MissingSlots as missing:
            # The question does not contain a value the recipe needs. No amount of re-prompting
            # produces a number the question never had, so the user is asked, not the model.
            return self._needs_slots(question, missing)
        except Unroutable as exc:
            # Two replies, neither of which this system can answer from. Saying so is the
            # honest outcome; computing from the second reply's numbers is not.
            return self._out_of_scope(question, exc.detail)
        except ActionError as exc:
            self._error_node("could not formalize the question", str(exc), [])
            return TaskResult("error", detail=f"formalization failed: {exc}")

        self._pending_plan = plan[0] if plan else None
        statement = self._statement(action, question, self._pending_plan)
        if action["recipe"] == recipes.NONE:
            return self._out_of_scope(question, statement)
        assumptions = action.get("assumptions") or {}
        goal = action.get("goal")
        if goal is not None and assumptions:
            spec = get_tool(goal["tool"])
            if "assumptions" in spec.schema.get("properties", {}):
                goal.setdefault("args", {}).setdefault("assumptions", assumptions)
        # A recipe names its own problem type, so the default-assumptions checklist is not
        # the model's choice either: picking the recipe is what decided the kind of problem.
        plan_recipe = recipes.RECIPES.get(action["recipe"])
        problem_type = normalize_type(plan_recipe.problem_type if plan_recipe
                                      else action.get("problem_type"))
        items = checklist(problem_type, action.get("modelling_assumptions"))
        # A recipe's slots are this problem's givens, so they are sources in exactly the same
        # way, and a slot the model filled with a number the question never had is unsourced.
        supplied = givens.keys()
        if self._pending_plan is not None and plan_recipe is not None:
            givens = plan_recipe.sources(self._pending_plan.values)
            supplied = {k for k in givens if k in self._pending_plan.supplied}
        # A given is a source for later tool calls only if its number is in the question.
        in_question = numbers_in(question)
        unsourced_givens = sorted(k for k, v in givens.items()
                                  if k in supplied and not numbers_in(v["value"]) <= in_question)
        if self._pending_plan is not None and self._pending_plan.from_question:
            taken = ", ".join(self._pending_plan.from_question)
            statement = f"{statement} ({taken} read from the question)"
        root = Node(
            session_id="", layer=Layer.REASONING, type=NodeType.PROBLEM, title="Problem",
            content=statement, content_canonical=normalize_question(question),
            fingerprint=question_fingerprint(question),
            tool_inputs={"question": question, "assumptions": assumptions, "goal": goal,
                         "problem_type": problem_type, "givens": givens, "checklist": items,
                         "unsourced_givens": unsourced_givens, "recipe": action["recipe"],
                         "slots": dict(self._pending_plan.values) if self._pending_plan else {},
                         # Slots code read out of the question instead of taking the model's
                         # copy, so the root the user confirms says where its values came from.
                         "from_question": list(self._pending_plan.from_question)
                         if self._pending_plan else []},
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
                self._pending_plan = None
        self._create_assumptions(root, items, rejected)
        return root

    @staticmethod
    def _statement(action: Action, question: str, plan: "recipes.Plan | None") -> str:
        """The problem as the root node states it, for the user to confirm or edit.

        The model may write one, and its own words are preferred when it does. It is not
        required to, because a grammar that does not require a field will not produce it, and a
        recipe plus its filled slots is a more precise statement than a restatement anyway. The
        last resort is the question itself, which is never wrong.
        """
        written = str(action.get("statement") or "").strip()
        if written:
            return written
        if plan is not None:
            recipe = recipes.RECIPES.get(plan.recipe)
            if recipe is not None:
                return recipe.statement(plan.values)
        return question.strip()

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
        hit = lookup.reusable_by_fingerprint(self.engine.repo, fp)
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
            self._next_role = pre.next_role
            return None, "\n".join(out.lines + pre.lines)
        self._next_role = out.next_role
        return None, "\n".join(out.lines)

    # --------------------------------------------------------------- recipes
    def _run_recipe(self, root: Node, plan: recipes.Plan) -> TaskResult:
        """Run a recipe's steps and write its answer, with no further model call.

        Every step's tool call is built by code from the slot values and the results of the
        steps before it, so an easy question costs exactly the one call that chose the recipe.
        Nothing here weakens a check: each call goes through ``_execute_tool``, which applies
        the risk rules and runs the required checks, and the answer still comes from
        ``_finalize`` filling a template from node results.

        A step whose independent check fails hands over to the failure ladder, which is the
        recovery the rest of the system already uses: the retry is of the one call that failed,
        by the specialist role the ladder chose, and the recipe itself is never re-planned by
        the model. A step that cannot be built or run at all is reported as it happened.
        """
        if plan.unsourced:
            # Unreachable through _formalize, which refuses such a plan before this is called.
            # Here as the last gate in front of the tools, because this is the invariant that
            # matters: no tool is ever run on a value the question did not contain.
            return self._recipe_failed(
                root, plan, f"{', '.join(plan.unsourced)} is not in the question")
        handles: list[str] = []
        results: list[dict[str, Any]] = []
        deps = [root.id]
        for build in plan.steps:
            try:
                built = build(plan.values, results)
            except recipes.SlotError as exc:
                return self._recipe_failed(root, plan, str(exc))
            if built is None:
                continue
            tool, args, title = built
            try:
                out = self._execute_tool(root, tool, dict(args), list(deps), None, "controller", title=title)
            except StepError as exc:
                return self._recipe_failed(root, plan, f"{title} could not be computed: {exc}")
            if out.result is not None:
                return out.result
            node = self._last_tool_node
            if node is None:
                return self._recipe_failed(root, plan, "the step produced no result")
            node = self.engine.resolve(node.id)
            if node.status in TERMINAL_BAD:
                # The recipe was the right plan and its tool failed its independent check. That
                # is what the failure ladder is for, and it has already chosen the specialist
                # role for the retry, so recovery goes step by step from here. The recipe is
                # never re-planned by the model: what is re-tried is the one call that failed.
                return self._loop(root, "\n".join(out.lines), out.next_role)
            handles.append(self.engine.handle(node.id))
            results.append(node.result or {})
            # Every later step depends on every earlier one, which is both the honest lineage
            # (the drug-likeness verdict really is built on the descriptors and the logP) and
            # what makes those results count as sources for the next step's arguments.
            deps = [*deps, node.id]
        if not handles:
            return self._recipe_failed(root, plan, "no step of the recipe produced a result")
        nodes = [self.engine.resolve(h) for h in handles]
        pre = self._ensure_verified(nodes)
        if pre.result is not None:
            return pre.result
        if any(self.engine.resolve(n.id).status in TERMINAL_BAD for n in nodes):
            return self._loop(root, "\n".join(pre.lines) or "a check failed", pre.next_role)
        return self._finalize(root, plan.answer(plan.values, handles), handles)

    def _recipe_failed(self, root: Node, plan: recipes.Plan, why: str) -> TaskResult:
        self._error_node(f"the {plan.recipe} recipe did not answer the question", why, [root.id])
        return TaskResult("error", detail=f"{plan.recipe}: {why}")

    def _needs_slots(self, question: str, missing: recipes.MissingSlots) -> TaskResult:
        """A required slot the question does not contain, recorded and asked about."""
        names = ", ".join(s.name for s in missing.slots)
        node = Node(session_id="", layer=Layer.REASONING, type=NodeType.HINT,
                    title="Missing value", role="controller",
                    content=f"The {missing.recipe} recipe needs {names}, which the question does "
                            f"not give.",
                    tool_inputs={"question": question, "recipe": missing.recipe,
                                 "missing": [s.name for s in missing.slots]})
        self.engine.add_node(node)
        return TaskResult("needs_user", question=missing.question(), final_node=node.id)

    def _out_of_scope(self, question: str, statement: str) -> TaskResult:
        """The router found no recipe for the question. It says so and lists what it has."""
        node = Node(session_id="", layer=Layer.REASONING, type=NodeType.HINT,
                    title="Out of scope", role="controller",
                    content=f"No recipe fits this question: {statement}",
                    tool_inputs={"question": question, "recipe": recipes.NONE,
                                 "supported": list(recipes.RECIPES)})
        self.engine.add_node(node)
        return TaskResult("out_of_scope", detail=recipes.out_of_scope(question),
                          final_node=node.id)

    def _loop(self, root: Node, observation: str, role: str = "controller") -> TaskResult:
        """Step-by-step recovery. Reached when a recipe step or a delegated goal fails its
        check, and for the dataset and molecule routes, which plan their own steps.

        ``role`` is the role the ladder chose for the retry. It used to be hard-coded to
        "controller", so a tool call made outside the loop that failed its check lost the
        specialist the ladder had just picked; the retry then came from the general controller,
        which knows the least about the tool that failed."""
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
        from sciai.llm.actions import action_schema

        # One schema, built once: it is both the output format Ollama generates under and the
        # rules the reply is judged by, so the model is never rejected for a field the request
        # did not require of it.
        schema = action_schema(allowed)
        last_err: ActionError | None = None
        for attempt in range(2):
            prompt = user if attempt == 0 else (
                f"{user}\n\nYOUR PREVIOUS REPLY WAS INVALID: {last_err}\nReply again with one valid JSON action.")
            self.llm_calls += 1
            reply = self.llm.chat(system, prompt, schema)
            try:
                action = parse_action(reply.text, allowed, schema)
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
        elif a.kind == "ask_user":
            check_question(a["question"])
        elif a.kind == "finish":
            root = self.engine.root()
            check_answer_template(a["answer_template"], a["answer_nodes"],
                                  (root.tool_inputs or {}).get("question", "") if root else "")
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
    _pending_plan: "recipes.Plan | None" = None
    _next_role: str = "controller"

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
            hit = lookup.reusable_by_fingerprint(self.engine.repo, fp)
            if hit is not None:
                handle = self.engine.link_existing(hit.id)
                for dep in self.engine.dependencies(hit.id):  # its dataset and test assumptions
                    if self.engine.resolve(dep).type in (NodeType.ENTITY, NodeType.ASSUMPTION) \
                            and dep not in self.engine.nodes:
                        self.engine.link_existing(dep)
                self._last_tool_node = hit
                kind = "hypothesis" if hit.status == Status.HYPOTHESIS else "verified result"
                return StepOutcome([f"Reused {kind} {handle} = {hit.display_result()}"])
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
            method = outcome.runs[-1].plan.method
            if node.status == Status.HYPOTHESIS:
                # chem: the check passed, but passing is not what makes a node believable here
                return StepOutcome([f"{h}: the {method} check passed; it stays a hypothesis{why}."])
            return StepOutcome([f"{h} verified by {method}{why}."])
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
        # A chemistry node is a hypothesis however well its checks went, so an answer built on
        # one is never verified. Saying only "unverified inputs" would read as a failed check;
        # the answer says what it actually rests on instead.
        hypotheses = [n for n in nodes if n.status == Status.HYPOTHESIS]
        if hypotheses:
            content += ("\nHypothesis: this answer rests on computed properties of candidates "
                        "nobody has tested.")
        final = Node(session_id="", layer=Layer.REASONING, type=NodeType.FINAL, title="Answer",
                     content=content,
                     tool_inputs={"template": template, "answer_nodes": [n.id for n in nodes],
                                  "assumptions": assumed})
        if not all_verified:
            unverified = [n for n in nodes if n.status != Status.HYPOTHESIS]
            final.flags = (["unverified inputs"] if unverified else []) + \
                          (["hypothesis"] if hypotheses else [])
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
        hit = lookup.reusable_by_fingerprint(self.engine.repo, fp)
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

    # ----------------------------------------------------------------- decline
    def _refusal(self, operation: str | None, goal: dict[str, Any] | None,
                 asked: str) -> chem_decline.Decline | None:
        """The refusal for a request nothing can serve, or None.

        The test is capability, never wording: ``decline_for`` asks the registry whether a tool
        performs the operation, so an operation the model names and no tool provides is declined
        because nothing does it. The keyword rules inside only choose which sentence is printed.
        A goal naming a chemistry tool that is not registered is the same thing said another
        way, and is declined on the same grounds.
        """
        if operation:
            refusal = chem_decline.decline_for(operation, asked)
            if refusal is not None:
                return refusal
        tool = (goal or {}).get("tool") or ""
        if tool.startswith("chem."):
            try:
                get_tool(tool)
            except KeyError:
                return chem_decline.decline_for(tool.split(".", 1)[1], asked)
        return None

    def _decline(self, refusal: chem_decline.Decline) -> TaskResult:
        """Record what was asked, which rule chose the wording, and what was available instead."""
        node = Node(session_id="", layer=Layer.REASONING, type=NodeType.HINT,
                    title="Declined", content=refusal.node_content(), role="controller",
                    tool_inputs={"operation": refusal.operation, "wording": refusal.wording,
                                 "asked": refusal.asked, "served": list(refusal.served)})
        self.engine.add_node(node)
        return TaskResult("declined", detail=refusal.message, final_node=node.id)

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
        if r.status in ("declined", "out_of_scope"):
            return r.detail
        return f"[{r.status}] {r.detail}"
