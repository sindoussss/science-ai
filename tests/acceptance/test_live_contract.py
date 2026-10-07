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
    is answered, verified, with the value the suite expects and the single routing call."""
    rt = make_rt(ScriptedLLM([record["reply"]]))
    rt.engine.new_session(name)
    res = rt.controller.run(record["question"])

    assert res.status == "answered", res.detail
    assert res.llm_calls == 1
    assert res.verified
    values = _values(rt, res)
    for expected in record["expected"]:
        assert any(abs(v - expected) <= 1e-4 * max(abs(expected), 1e-12) for v in values), \
            f"expected {expected}, got {values}"


def _values(rt, res) -> list[float]:
    out: list[float] = []
    final = rt.engine.resolve(res.final_node)
    for nid in (final.tool_inputs or {}).get("answer_nodes", []):
        out += _numbers(rt.engine.resolve(nid).result)
    return out


def _numbers(result: Any) -> list[float]:
    if not isinstance(result, dict):
        return []
    out = []
    for key in ("value", "si_value", "numeric"):
        v = result.get(key)
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            out.append(float(v))
        elif isinstance(v, list):
            out += [x for item in v for x in _numbers(item if isinstance(item, dict) else {"value": item})]
    return out


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
