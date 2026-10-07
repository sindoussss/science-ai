"""The router end to end: one model call, code plans, and every check still applies.

These are the live-check failures as acceptance tests. Each one drives the real tool sandbox
with a scripted router reply, so what is measured is what the system does with a reply the
model can actually produce -- which was the whole problem: the old schema had no such reply.
"""
from __future__ import annotations

import math
import re

import pytest

from sciai.controller import recipes as R
from sciai.graph.model import NodeType, Status
from sciai.tools.faults import FaultyRunner
from tests.fakes.fake_llm import ScriptedLLM


def route(recipe, statement, **slots):
    return {"action": "formalize", "recipe": recipe, "statement": statement, "slots": slots}


def nodes(rt, tool=None, type_=None):
    out = [n for n in rt.engine.nodes.values() if (tool is None or n.tool_name == tool)
           and (type_ is None or n.type == type_)]
    return sorted(out, key=lambda n: n.created_at)


def methods(rt, node_id):
    return {ev.method: ev.outcome for ev in rt.repo.evidence_for(node_id)}


def values_of(result):
    """Every real number in a stored result, so a symbolic answer counts by its numeric value."""
    if not isinstance(result, dict):
        return []
    out = []
    for key in ("value", "si_value", "numeric"):
        v = result.get(key)
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            out.append(float(v))
        elif isinstance(v, list):
            for item in v:
                out += values_of(item) if isinstance(item, dict) else (
                    [float(item)] if isinstance(item, (int, float)) else [])
    return out


# Every recipe, the question it answers, the reply the router gives, and the answer expected.
CASES = [
    ("unit_convert", "Convert 60 miles per hour to meters per second.",
     route("unit_convert", "Convert 60 mph to m/s.", value="60", from_unit="mph", to_unit="m/s"),
     "That is 26.82 m/s.", (26.8224,)),
    ("unit_convert_temperature", "Convert 25 degC to kelvin.",
     route("unit_convert", "Convert 25 degC to K.", value="25", from_unit="degC", to_unit="K",
           kind="absolute_temperature"),
     "That is 298.1 K.", (298.15,)),
    ("definite_integral", "Compute the definite integral of x^2*exp(-x) from x = 0 to x = 1.",
     route("definite_integral", "Integrate x**2*exp(-x) from 0 to 1.",
           integrand="x**2*exp(-x)", var="x", lower="0", upper="1"),
     "The integral is 2 - 5*exp(-1).", (2 - 5 / math.e,)),
    ("solve_equation", "Solve x^2 - 5*x + 6 = 0 for x.",
     route("solve_equation", "Solve x**2 - 5*x + 6 = 0.", equation="x**2 - 5*x + 6 = 0", var="x"),
     "x = 2, 3.", (2.0, 3.0)),
    ("area_between_curves", "Find the area enclosed between y = x^3 - 3*x and y = x.",
     route("area_between_curves", "Area between x**3 - 3*x and x.",
           curve1="x**3 - 3*x", curve2="x", var="x"),
     "The area between the curves is 8.", (8.0,)),
    ("ode_ivp", "Solve dy/dx = 6*x^2 - 4*x with y(0) = 1, then give y(2).",
     route("ode_ivp", "Solve y' = 6*x**2 - 4*x, y(0) = 1, then evaluate y(2).",
           dy_dx="6*x**2 - 4*x", func="y", var="x", y0="1", at="0", evaluate_at="2"),
     "y(2) = 9.", (9.0,)),
    ("projectile_range", "A ball is thrown at 20 m/s at 30 degrees above level ground. How far?",
     route("projectile_range", "Range at 20 m/s, 30 deg.",
           v0="20", v0_unit="m/s", angle="30", angle_unit="deg", to_unit="m"),
     "The range is 35.32 m.", (400 * math.sin(math.radians(60)) / 9.80665,)),
    ("photon_energy", "What is the energy of a photon with a wavelength of 500 nm?",
     route("photon_energy", "Photon energy at 500 nm.",
           wavelength="500", wavelength_unit="nm", to_unit="J"),
     "The photon energy is 3.973e-19 J.", (6.62607015e-34 * 299792458 / 500e-9,)),
    ("rc_discharge", "A 1 uF capacitor charged to 5 V discharges through 1 kohm. Voltage at 2 ms?",
     route("rc_discharge", "RC discharge at 2 ms.", v0="5", v0_unit="V", resistance="1",
           resistance_unit="kohm", capacitance="1", capacitance_unit="uF", t="2", t_unit="ms",
           to_unit="V"),
     "The voltage after that time is 0.6767 V.", (5 * math.exp(-2),)),
    ("series_parallel_current",
     "A 12 V source drives a 100 ohm resistor in series with two 200 ohm resistors in parallel. "
     "What current flows from the source?",
     route("series_parallel_current", "12 V, 100 ohm in series with 200 || 200.",
           voltage="12", voltage_unit="V", series="100", parallel="200, 200"),
     None, (0.06,)),
]


@pytest.mark.parametrize("key, question, reply, answer, expected",
                         CASES, ids=[c[0] for c in CASES])
def test_a_recipe_answers_in_one_model_call(make_rt, key, question, reply, answer, expected):
    llm = ScriptedLLM([reply])
    rt = make_rt(llm)
    rt.engine.new_session(key)
    res = rt.controller.run(question)

    assert res.status == "answered", res
    assert res.verified, res
    assert res.llm_calls == 1, f"{key} cost {res.llm_calls} model calls"
    if answer is not None:
        assert res.answer == answer
    final = rt.engine.resolve(res.final_node)
    shown = [float(v) for v in re.findall(r"-?\d+\.?\d*(?:e-?\d+)?", res.answer)]
    for nid in final.tool_inputs["answer_nodes"]:
        shown += values_of(rt.engine.resolve(nid).result)
    for value in expected:
        assert any(math.isclose(abs(value), abs(v), rel_tol=1e-3) for v in shown), \
            (key, value, res.answer, shown)
    # Nothing was left unchecked, and nothing was answered by the model.
    for n in nodes(rt, type_=NodeType.TOOL_RESULT):
        assert n.status == Status.VERIFIED, (key, n.title, n.status)
        assert rt.repo.evidence_for(n.id), (key, n.title)
        assert not n.flags, (key, n.flags)


def test_a_question_no_recipe_covers_is_out_of_scope(make_rt):
    llm = ScriptedLLM([{"action": "formalize", "recipe": "none", "slots": {},
                        "statement": "The user asked for a limerick about thermodynamics."}])
    rt = make_rt(llm)
    rt.engine.new_session("scope")
    res = rt.controller.run("Write me a limerick about thermodynamics.")

    assert res.status == "out_of_scope", res
    assert res.llm_calls == 1
    for recipe in R.RECIPES:
        assert recipe in res.detail
    hint = rt.engine.resolve(res.final_node)
    assert hint.type == NodeType.HINT and hint.title == "Out of scope"
    assert not nodes(rt, type_=NodeType.TOOL_RESULT)  # nothing was computed


def test_a_missing_value_asks_the_user_once(make_rt):
    llm = ScriptedLLM([route("definite_integral", "Integrate x**2.", integrand="x**2", var="x")])
    rt = make_rt(llm)
    rt.engine.new_session("missing")
    res = rt.controller.run("Integrate x squared.")

    assert res.status == "needs_user", res
    assert res.question == "I need the lower bound and the upper bound. What are they?"
    assert res.llm_calls == 1  # asked once; no re-prompt for a value the question never had
    assert rt.engine.resolve(res.final_node).title == "Missing value"


def test_a_slot_type_error_costs_one_re_prompt_and_then_works(make_rt):
    """Item 4: a re-prompt is for a real type error only, and it says what the slot takes."""
    bad = route("unit_convert", "Convert 90 km/h.", value="ninety", from_unit="km/h",
                to_unit="m/s")
    good = route("unit_convert", "Convert 90 km/h.", value="90", from_unit="km/h", to_unit="m/s")
    llm = ScriptedLLM([bad, good])
    rt = make_rt(llm)
    rt.engine.new_session("retry")
    res = rt.controller.run("Convert 90 km/h to m/s.")

    assert res.status == "answered" and res.verified
    assert res.llm_calls == 2
    assert "must be a plain number" in llm.calls[1]["user"]


def test_the_right_number_in_the_wrong_unit_fails_the_node(make_rt, runner):
    """mistral-nemo answered 96560.64 m/h to a question that asked for m/s, and it was accepted.

    The requested unit is a slot, so this is now a failed check and not a detail: the node
    cannot be verified and the answer is never built from it.
    """
    def wrong_unit(result):
        return {**result, "value": 96560.64, "unit": "m/h"}

    faulty = FaultyRunner(runner, "phys.projectile_range", corrupt=wrong_unit, calls=None)
    llm = ScriptedLLM([route("projectile_range", "Range at 20 m/s, 30 deg.", v0="20",
                             v0_unit="m/s", angle="30", angle_unit="deg", to_unit="m"),
                       {"action": "ask_user", "question": "Which unit did you want?"}])
    rt = make_rt(llm, run=faulty)
    rt.engine.new_session("unit")
    res = rt.controller.run("A ball is thrown at 20 m/s at 30 degrees; how far, in metres?")

    assert res.status != "answered", res
    bad = nodes(rt, "phys.projectile_range")[0]
    assert bad.status in (Status.FAILED, Status.INVALIDATED)
    assert methods(rt, bad.id)["target_unit"] == "fail"
    assert "96560.64" not in res.answer


def test_a_result_in_the_wrong_dimension_fails_the_node(make_rt, runner):
    """The photon energy came back as "1 / m J", which is not an energy at all."""
    def wrong_dimension(result):
        return {**result, "unit": "J / m"}

    faulty = FaultyRunner(runner, "phys.photon_energy", corrupt=wrong_dimension, calls=None)
    llm = ScriptedLLM([route("photon_energy", "Photon energy at 500 nm.", wavelength="500",
                             wavelength_unit="nm", to_unit="J"),
                       {"action": "ask_user", "question": "Which unit did you want?"}])
    rt = make_rt(llm, run=faulty)
    rt.engine.new_session("dims")
    res = rt.controller.run("What is the energy of a photon of wavelength 500 nm, in joules?")

    assert res.status != "answered", res
    bad = nodes(rt, "phys.photon_energy")[0]
    assert bad.status in (Status.FAILED, Status.INVALIDATED)
    detail = next(ev.detail for ev in rt.repo.evidence_for(bad.id) if ev.method == "target_unit")
    assert detail["got_dims"] != detail["want_dims"]


def test_an_area_that_is_not_positive_fails_the_node(make_rt, runner):
    """An area computed without the absolute value cancels to zero; zero is not an area."""
    def cancelled(result):
        return {**result, "value": 0.0}

    faulty = FaultyRunner(runner, "calc.area_between", corrupt=cancelled, calls=None)
    llm = ScriptedLLM([route("area_between_curves", "Area between x**3 - 3*x and x.",
                             curve1="x**3 - 3*x", curve2="x", var="x"),
                       {"action": "ask_user", "question": "Did you mean a signed area?"}])
    rt = make_rt(llm, run=faulty)
    rt.engine.new_session("area")
    res = rt.controller.run("Find the area enclosed between y = x^3 - 3*x and y = x.")

    assert res.status != "answered", res
    bad = nodes(rt, "calc.area_between")[0]
    assert bad.status in (Status.FAILED, Status.INVALIDATED)
    detail = next(ev.detail for ev in rt.repo.evidence_for(bad.id) if ev.method == "quadrature")
    assert "positive area" in detail["reason"]


def test_an_ode_solution_that_breaks_its_initial_condition_fails_the_node(make_rt, runner):
    """It solves the equation, but not this problem: y(0) would be 3 and the question said 1."""
    faulty = FaultyRunner(runner, "ode.dsolve", "3*exp(-2*x)", calls=None)
    llm = ScriptedLLM([route("ode_ivp", "Solve y' = -2*y, y(0) = 1.", dy_dx="-2*y", func="y",
                             var="x", y0="1", at="0"),
                       {"action": "ask_user", "question": "Which initial condition did you mean?"}])
    rt = make_rt(llm, run=faulty)
    rt.engine.new_session("ode")
    res = rt.controller.run("Solve dy/dx = -2*y with y(0) = 1.")

    assert res.status != "answered", res
    bad = nodes(rt, "ode.dsolve")[0]
    assert bad.status in (Status.FAILED, Status.INVALIDATED)
    assert methods(rt, bad.id)["residual"] == "fail"


def test_a_recipe_answer_never_writes_a_unit_twice(make_rt):
    llm = ScriptedLLM([route("projectile_range", "Range at 20 m/s, 30 deg.", v0="20",
                             v0_unit="m/s", angle="30", angle_unit="deg", to_unit="m")])
    rt = make_rt(llm)
    rt.engine.new_session("once")
    res = rt.controller.run("A ball is thrown at 20 m/s at 30 degrees; how far does it land?")
    assert res.answer == "The range is 35.32 m."
    assert "m meters" not in res.answer and res.answer.count(" m") == 1


def test_the_answer_still_comes_from_the_tool_and_not_the_model(make_rt):
    """The safety invariant is untouched: the recipe writes the sentence, the tool the value."""
    llm = ScriptedLLM([route("projectile_range", "Range at 20 m/s, 30 deg.", v0="20",
                             v0_unit="m/s", angle="30", angle_unit="deg", to_unit="m")])
    rt = make_rt(llm)
    rt.engine.new_session("template")
    res = rt.controller.run("A ball is thrown at 20 m/s at 30 degrees; how far does it land?")
    final = rt.engine.resolve(res.final_node)
    handle = rt.engine.handle(final.tool_inputs["answer_nodes"][0])
    assert final.tool_inputs["template"] == "The range is {{%s}}." % handle
    value = rt.engine.resolve(final.tool_inputs["answer_nodes"][0])
    assert value.tool_name == "phys.projectile_range"
    assert value.display_result() in res.answer
