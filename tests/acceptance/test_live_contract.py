"""The contract between the schema the router sends and the validator that reads the reply.

Every live run on 2026-10-06 failed at the same step: the request declared only ``action``
required, so Ollama's grammar let the model stop after ``{"action","recipe","slots"}``, while
the validator demanded ``statement`` as well. Every mock test passed, because a mock writes
whatever the test wrote.

These tests close that gap from both ends. ``tests/fixtures/live/`` holds recorded replies in
the format ``--trace`` writes, and each one is checked against the schema it was generated
under, then put through the validator and the controller. And for every action, the *minimal*
object the schema permits is put through the validator too, which is the test that would have
caught the bug: that minimal formalize object is exactly what qwen3:8b sent.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import jsonschema
import pytest

from sciai.llm.actions import ACTIONS, FIELDS, REQUIRED, action_schema, action_variant, parse_action
from sciai.llm.client import LLMReply
from tests.fakes.fake_llm import ScriptedLLM

FIXTURES = sorted((Path(__file__).resolve().parents[1] / "fixtures" / "live").glob("*.jsonl"))


def records() -> list[tuple[str, dict[str, Any]]]:
    out = []
    for path in FIXTURES:
        for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if line.strip():
                out.append((f"{path.name}:{i}", json.loads(line)))
    return out


RECORDS = records()
IDS = [name for name, _ in RECORDS]


def schema_of(record: dict[str, Any]) -> dict[str, Any]:
    """The schema the request sent, from the record when a real trace carries it."""
    sent = (record.get("request") or {}).get("schema")
    return sent if isinstance(sent, dict) else action_schema(("formalize",))


def test_there_are_fixtures_for_every_route():
    """A new route without a recorded reply is a route nothing has ever validated."""
    from sciai.controller import recipes

    routed = {json.loads(r["reply"])["recipe"] for _, r in RECORDS}
    assert set(recipes.ROUTES) <= routed


@pytest.mark.parametrize("name,record", RECORDS, ids=IDS)
def test_a_recorded_reply_satisfies_the_schema_it_was_generated_under(name, record):
    """Step one: the fixture is a reply a real model could have produced."""
    jsonschema.validate(json.loads(record["reply"]), schema_of(record))


@pytest.mark.parametrize("name,record", RECORDS, ids=IDS)
def test_the_validator_accepts_every_recorded_reply(name, record):
    """Step two, and the one that failed live: what the grammar allows, the validator takes."""
    action = parse_action(record["reply"], ("formalize",), schema_of(record))
    assert action.kind == "formalize"


def _is_recipe(record: dict[str, Any]) -> bool:
    from sciai.controller import recipes

    return json.loads(record["reply"])["recipe"] in recipes.RECIPES


RECIPE_RECORDS = [(n, r) for n, r in RECORDS if _is_recipe(r)]


@pytest.mark.parametrize("name,record", RECIPE_RECORDS, ids=[n for n, _ in RECIPE_RECORDS])
def test_a_recorded_recipe_reply_plans(name, record):
    """Step three: the slots a real reply carries fill the recipe they name."""
    from sciai.controller import recipes

    reply = json.loads(record["reply"])
    route = reply["recipe"]
    plan = recipes.RECIPES[route].plan(reply.get("slots") or {})
    assert plan.recipe == route
    assert plan.steps and recipes.RECIPES[route].statement(plan.values)


ANSWERABLE = [(n, r) for n, r in RECORDS if r.get("expected")]


@pytest.mark.parametrize("name,record", ANSWERABLE, ids=[n for n, _ in ANSWERABLE])
def test_the_controller_answers_a_recorded_reply_in_one_model_call(name, record, make_rt):
    """End to end on the recorded text itself: the reply is replayed verbatim, and the question
    is answered with the value the suite expects and the single routing call. A chemistry
    answer is a hypothesis and must never come back verified, which is Phase 4's whole rule."""
    rt = make_rt(ScriptedLLM([record["reply"]]))
    rt.engine.new_session(name)
    res = rt.controller.run(record["question"])

    assert res.status == "answered", res.detail
    assert res.llm_calls == 1
    assert res.verified is not record.get("hypothesis", False)
    values = _values(rt, res)
    tol = float(record.get("rel_tol") or 1e-4)
    for expected in record["expected"]:
        assert any(abs(v - expected) <= tol * max(abs(expected), 1e-12) for v in values), \
            f"expected {expected}, got {values}"


REFUSED = [(n, r) for n, r in RECORDS if r.get("refused")]


@pytest.mark.parametrize("name,record", REFUSED, ids=[n for n, _ in REFUSED])
def test_a_reply_whose_slots_are_not_in_the_question_is_refused(name, record, make_rt):
    """The chem run's wrong answers, as permanent tests. A molecule question answered
    "3.9728917e-19" was the photon-energy example's own value: the router had picked that
    recipe and the model copied its worked example. Both replies (the first and the one retry)
    are the same, because that is what the model did -- and nothing may be computed from them.
    """
    rt = make_rt(ScriptedLLM([record["reply"], record["reply"]]))
    rt.engine.new_session(name)
    res = rt.controller.run(record["question"])

    assert res.status == "out_of_scope", f"{res.status}: {res.detail or res.answer}"
    assert not nodes_of_type(rt, "tool_result"), "a tool ran on a value the question never had"


def nodes_of_type(rt, type_name: str) -> list[Any]:
    return [n for n, _ in rt.repo.session_view(rt.engine.session_id)
            if n.type.value == type_name]


def _values(rt, res) -> list[float]:
    out: list[float] = []
    final = rt.engine.resolve(res.final_node)
    for nid in (final.tool_inputs or {}).get("answer_nodes", []):
        out += _numbers(rt.engine.resolve(nid).result)
    return out


def _numbers(value: Any) -> list[float]:
    """Every number anywhere in a stored result, so one helper covers a quantity, a descriptor
    table and a neighbour list alike."""
    if isinstance(value, bool):
        return []
    if isinstance(value, (int, float)):
        return [float(value)]
    if isinstance(value, dict):
        return [x for v in value.values() for x in _numbers(v)]
    if isinstance(value, (list, tuple)):
        return [x for v in value for x in _numbers(v)]
    return []


def test_a_reply_without_a_statement_gets_one_from_the_recipe(make_rt):
    """What replaced the required field: the root the user confirms is written from the recipe
    and its slots, which is more precise than a restatement and cannot be left out."""
    from sciai.controller import recipes

    record = next(r for n, r in RECORDS if n.startswith("formalize-no-statement"))
    reply = json.loads(record["reply"])
    assert "statement" not in reply

    rt = make_rt(ScriptedLLM([record["reply"]]))
    rt.engine.new_session("no statement")
    res = rt.controller.run(record["question"])

    assert res.status == "answered", res.detail
    root = rt.engine.resolve(rt.engine.root().id)
    expected = recipes.RECIPES["definite_integral"].statement(
        recipes.RECIPES["definite_integral"].plan(reply["slots"]).values)
    assert root.content == expected
    assert "x**2*exp(-x)" in root.content and "lower = 0" in root.content
    assert root.tool_inputs["recipe"] == "definite_integral"


# ------------------------------------------------- the minimal reply the schema permits
def minimal(schema: dict[str, Any]) -> dict[str, Any]:
    """The smallest object the schema allows: its required properties and nothing else.

    A grammar-constrained model tends to produce exactly this, which is why it is the shape
    worth testing. Values come from each property's own schema, so this stays true when the
    schema changes.
    """
    out: dict[str, Any] = {}
    for name in schema["required"]:
        spec = schema["properties"][name]
        if "enum" in spec:
            out[name] = spec["enum"][0]
        elif spec.get("type") == "object":
            out[name] = {}
        elif spec.get("type") == "array":
            out[name] = []
        elif spec.get("pattern") == "^n[0-9]+$":
            out[name] = "n1"
        else:
            out[name] = "x"
    return out


@pytest.mark.parametrize("kind", ACTIONS)
def test_the_validator_accepts_the_minimal_reply_for_every_action(kind):
    schema = action_schema((kind,))
    data = minimal(schema)
    assert parse_action(json.dumps(data), (kind,), schema).kind == kind
    # And the same object inside the full contract, which is what the controller loop sends.
    whole = action_schema()
    assert parse_action(json.dumps(data), ACTIONS, whole).kind == kind


@pytest.mark.parametrize("kind", ACTIONS)
def test_an_action_requires_only_fields_it_declares(kind):
    """No required field outside FIELDS, and no required field the prompt cannot explain."""
    assert set(REQUIRED[kind]) <= set(FIELDS[kind])
    assert action_variant(kind)["required"] == ["action", *REQUIRED[kind]]


def test_the_formalize_request_does_not_require_a_statement():
    """The field that broke every live run. Code writes the statement now."""
    assert "statement" not in REQUIRED["formalize"]
    assert "statement" in FIELDS["formalize"]  # still accepted, and preferred when it is there


def test_the_schema_sent_to_the_model_is_the_schema_the_reply_is_judged_by(make_rt):
    """One schema, literally: the object handed to ``chat`` is the object ``parse_action`` uses."""
    sent: list[dict[str, Any]] = []
    judged: list[dict[str, Any]] = []

    class Recording:
        model_name = "recording"

        def chat(self, system: str, user: str, schema: dict[str, Any]) -> LLMReply:
            sent.append(schema)
            return LLMReply('{"action":"formalize","recipe":"definite_integral","slots":'
                            '{"integrand":"x**2","var":"x","lower":"0","upper":"1"}}')

    import sciai.controller.loop as loop_module

    real = loop_module.parse_action

    def spy(text, allowed=ACTIONS, schema=None):
        judged.append(schema)
        return real(text, allowed, schema)

    monkey = pytest.MonkeyPatch()
    monkey.setattr(loop_module, "parse_action", spy)
    try:
        rt = make_rt(Recording())
        rt.engine.new_session("one schema")
        res = rt.controller.run("Compute the definite integral of x**2 from x = 0 to x = 1.")
    finally:
        monkey.undo()

    assert res.status == "answered", res.detail
    assert sent and judged
    assert sent[0] is judged[0]
    assert sent[0] == action_schema(("formalize",))


def prompt_examples() -> list[tuple[str, str, dict[str, Any]]]:
    """Every worked example in the router's prompt, as (label, recipe, slots)."""
    from sciai.controller import recipes

    out = []
    for name, recipe in recipes.RECIPES.items():
        for i, text in enumerate(recipe.examples, 1):
            _, arrow, tail = text.partition("-> ")
            assert arrow, f"{name} example {i} is not '\"question\" -> {{slots}}': {text}"
            out.append((f"{name}#{i}", name, json.loads(tail)))
    return out


EXAMPLES = prompt_examples()
# A question with no digit and no number word in it ("two" would have sourced the 2 in x**2),
# so every number in an example slot is plainly not from here.
UNRELATED = "Which of these molecules is more lipophilic, and why does it matter?"


@pytest.mark.parametrize("label,route,slots", EXAMPLES, ids=[e[0] for e in EXAMPLES])
def test_a_worked_example_value_never_reaches_an_answer(label, route, slots, make_rt):
    """Yeri's item 6, in its general form: an example in the prompt is wording, never data.

    The live run answered five chemistry questions with the photon-energy example's
    3.9728917e-19 J and the circuit example's 0.06 A. Every example of every recipe is replayed
    here against a question that contains none of its values: the plan must be refused, nothing
    may be computed, and no value of the example may appear in what the user is told.
    """
    from sciai.controller import recipes
    from sciai.controller.provenance import numbers_in

    try:
        plan = recipes.RECIPES[route].plan(slots, UNRELATED)
        assert plan.unsourced, f"{label}: the example's slots passed as the question's own values"
    except recipes.MissingSlots:
        pass    # a structure recipe asks the user for one rather than ruling on the example

    reply = json.dumps({"action": "formalize", "recipe": route, "slots": slots})
    rt = make_rt(ScriptedLLM([reply, reply]))   # the first reply and the one re-ask
    rt.engine.new_session(label)
    res = rt.controller.run(UNRELATED)

    # Either answer is a refusal of the example: no recipe fits, or the question is put back to
    # the user. What may never happen is a value of the example being used or shown.
    assert res.status in ("out_of_scope", "needs_user"), \
        f"{label}: {res.status}: {res.detail or res.answer}"
    assert not nodes_of_type(rt, "tool_result"), f"{label}: a tool ran on an example value"
    told = " ".join(str(x) for x in (res.answer, res.detail, res.question) if x)
    for shown in numbers_in(slots):
        assert shown not in told, f"{label}: the example's {shown} was given to the user"
