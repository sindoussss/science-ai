"""The five Phase 2 acceptance tests, against a scripted model and the real tool sandbox."""
from __future__ import annotations

import math

import pytest

from sciai.controller import lookup
from sciai.domains.physics.assumptions import DEFAULTS
from sciai.graph.model import Layer, LadderStage, NodeType, Status
from sciai.tools.faults import FaultyRunner
from tests.conftest import numbers, result_of
from tests.fakes.fake_llm import ScriptedLLM, last_handle

G0 = 9.80665


def q(value, unit, kind=None):
    return {"value": value, "unit": unit, **({"kind": kind} if kind else {})}


def methods(rt, node_id):
    return {ev.method: ev.outcome for ev in rt.repo.evidence_for(node_id)}


def nodes(rt, tool=None, type_=None):
    out = [n for n in rt.engine.nodes.values() if (tool is None or n.tool_name == tool)
           and (type_ is None or n.type == type_)]
    return sorted(out, key=lambda n: n.created_at)


def assert_numbers_from_tools(rt, final_id):
    final = rt.engine.resolve(final_id)
    allowed = set()
    for nid in final.tool_inputs["answer_nodes"]:
        allowed |= numbers(rt.engine.resolve(nid).display_result())
    assert numbers(final.content) <= allowed, (final.content, allowed)


# 1 ---------------------------------------------------------------- projectile
PROJECTILE_Q = "A ball is thrown at 20 m/s at 30 degrees; how far does it land?"
PROJECTILE_FORMAL = {
    "action": "formalize", "statement": "Range of a projectile launched at 20 m/s, 30 degrees above level ground.",
    "problem_type": "projectile",
    "givens": {"v0": q(20, "m/s", "speed"), "theta": q(30, "deg", "angle")},
}
CONST_G = {"action": "call_tool", "tool": "phys.constant", "args": {"name": "g"}, "title": "g"}


def range_step(prompt):
    g = last_handle(prompt, "phys.constant")
    assert result_of(prompt, g).startswith("9.80665")
    return {"action": "call_tool", "tool": "phys.evaluate", "title": "range", "depends_on": [g, "n1"],
            "args": {"expr": "v0**2*sin(2*theta)/g", "to_unit": "m", "kind": "distance",
                     "values": {"v0": q(20, "m/s"), "theta": q(30, "deg"), "g": q(G0, "m/s^2")}}}


def finish_range(prompt):
    h = last_handle(prompt, "phys.evaluate", status="verified")
    return {"action": "finish", "answer_template": f"The ball lands {{{{{h}}}}} away.", "answer_nodes": [h]}


def run_projectile(rt):
    return rt.controller.run(PROJECTILE_Q)


def test_projectile_with_units_end_to_end(make_rt):
    llm = ScriptedLLM([PROJECTILE_FORMAL, CONST_G, range_step, finish_range])
    rt = make_rt(llm)
    rt.engine.new_session("p1")
    res = run_projectile(rt)

    assert res.status == "answered" and res.verified, res
    assert res.llm_calls == 4
    expected = 400 * math.sin(math.radians(60)) / G0  # 35.32 m; sin(30 rad) would give -12.4 m
    rng = nodes(rt, "phys.evaluate")[0]
    assert rng.result["value"] == pytest.approx(expected, rel=1e-12) and rng.result["unit"] == "m"
    assert "35.3" in res.answer and res.answer.endswith("m away.")
    assert methods(rt, rng.id) == {"plausibility": "pass", "dimensional": "pass", "alt_algorithm": "pass"}
    g = nodes(rt, "phys.constant")[0]
    assert g.status == Status.VERIFIED and methods(rt, g.id) == {"plausibility": "pass", "known_value": "pass"}
    assert not rng.flags  # every value came from a given or the constant node

    # The projectile checklist became locked assumption nodes the problem depends on.
    assumed = nodes(rt, type_=NodeType.ASSUMPTION)
    assert [a.content for a in assumed] == list(DEFAULTS["projectile"])
    assert all(a.locked and a.status == Status.PROPOSED for a in assumed)
    root = rt.engine.root()
    assert {a.id for a in assumed} <= set(rt.engine.dependencies(root.id))
    assert res.assumptions == list(DEFAULTS["projectile"])
    assert rt.engine.resolve(res.final_node).tool_inputs["assumptions"] == list(DEFAULTS["projectile"])
    assert_numbers_from_tools(rt, res.final_node)


def test_unticked_assumption_is_recorded_as_rejected(make_rt, cfg):
    cfg.controller.auto_confirm_root = False
    llm = ScriptedLLM([{**PROJECTILE_FORMAL, "modelling_assumptions": ["no spin"]}, CONST_G, range_step,
                       finish_range])
    rt = make_rt(llm)
    seen = []

    def confirm(root):
        seen.append([i["text"] for i in root.tool_inputs["checklist"]])
        return {"statement": None, "rejected": ["launch and landing at the same height"]}

    rt.controller.confirm_root = confirm
    rt.engine.new_session("p1b")
    res = run_projectile(rt)

    assert seen == [[*DEFAULTS["projectile"], "no spin"]]
    assert res.status == "answered" and res.verified
    assert res.assumptions == ["no air resistance", "constant g = 9.80665 m/s^2", "no spin"]
    dropped = next(a for a in nodes(rt, type_=NodeType.ASSUMPTION) if a.content.startswith("launch"))
    assert dropped.status == Status.FAILED and not dropped.locked
    assert dropped.id not in rt.engine.dependencies(rt.engine.root().id)


# 2 ---------------------------------------------------------------- RC discharge
RC_Q = "A 1 uF capacitor charged to 5 V discharges through a 1 kohm resistor. What is its voltage at t = 2 ms?"
RC_FORMAL = {
    "action": "formalize", "statement": "RC discharge: C = 1 uF, R = 1 kohm, V0 = 5 V; find v(2 ms).",
    "problem_type": "rc_rl_transient",
    "givens": {"C": q(1, "uF", "capacitance"), "R": q(1, "kohm", "resistance"), "V0": q(5, "V", "voltage"),
               "t1": q(2, "ms", "time")},
}
DSOLVE = {"action": "call_tool", "tool": "ode.dsolve", "title": "v(t)",
          "args": {"equation": "Derivative(v(t), t) = -v(t)/(1000*1e-6)", "func": "v", "var": "t",
                   "ics": [{"at": "0", "value": "5"}], "t_span": ["0", "0.002"]}}
IVP = {"action": "call_tool", "tool": "ode.solve_ivp", "title": "v(t) numeric",
       "args": {"rhs": ["-v/(1000*1e-6)"], "funcs": ["v"], "var": "t", "y0": ["5"], "t_span": ["0", "0.002"]}}


def last_result(prompt, handle):
    """The result on a digest line whose args may themselves contain " = "."""
    line = next(ln for ln in prompt.splitlines() if ln.startswith(handle + " "))
    return line.split(" <- ")[0].rsplit(" = ", 1)[1].strip()


def subs_rc(prompt):
    h = last_handle(prompt, "ode.dsolve", status="verified")
    return {"action": "call_tool", "tool": "sympy.subs", "title": "v(2 ms)", "depends_on": [h],
            "args": {"expr": last_result(prompt, h), "values": {"t": "0.002"}}}


def plot_rc(prompt):
    return {"action": "call_tool", "tool": "plot.series", "title": "v(t) plot",
            "args": {"node": last_handle(prompt, "ode.solve_ivp"), "component": "v"}}


def finish_rc(prompt):
    d, s, i = (last_handle(prompt, t) for t in ("ode.dsolve", "sympy.subs", "ode.solve_ivp"))
    return {"action": "finish", "answer_template": f"v(t) = {{{{{d}}}}} volts, so at the given time it is "
                                                   f"{{{{{s}}}}} volts (numeric run: {{{{{i}}}}}).",
            "answer_nodes": [d, s, i]}


def test_rc_discharge(make_rt):
    llm = ScriptedLLM([RC_FORMAL, DSOLVE, IVP, plot_rc, subs_rc, finish_rc])
    rt = make_rt(llm)
    rt.engine.new_session("rc")
    res = rt.controller.run(RC_Q)

    assert res.status == "answered" and res.verified, res
    assert res.llm_calls == 6
    dsolve, ivp, subs = nodes(rt, "ode.dsolve")[0], nodes(rt, "ode.solve_ivp")[0], nodes(rt, "sympy.subs")[0]
    assert dsolve.result["value"] == "5*exp(-1000.0*t)"
    assert methods(rt, dsolve.id) == {"residual": "pass", "symbolic_vs_numeric": "pass"}
    assert methods(rt, ivp.id) == {"alt_algorithm": "pass", "symbolic_vs_numeric": "pass"}
    assert float(subs.result["numeric"]) == pytest.approx(5 * math.exp(-2), rel=1e-12)  # 0.677 V
    assert ivp.result["value"][0] == pytest.approx(5 * math.exp(-2), rel=1e-6)
    assert "0.6766" in res.answer
    plot = nodes(rt, "plot.series")[0]
    assert plot.type == NodeType.PLOT and plot.tool_inputs["node"] == ivp.id and not plot.flags
    assert ivp.id in rt.engine.dependencies(plot.id)
    assert plot.result["value"]["y"][-1] == pytest.approx(ivp.result["value"][0])
    for n in (dsolve, ivp, subs):
        assert n.status == Status.VERIFIED and not n.flags, n  # every number sourced from the givens
    assert_numbers_from_tools(rt, res.final_node)


# 3 ---------------------------------------------------------------- injected unit error
CAR_Q = "A car travels at 72 km/h for 15 s. How far does it go?"
CAR_FORMAL = {"action": "formalize", "statement": "Distance at 72 km/h for 15 s.", "problem_type": "kinematics",
              "givens": {"v": q(72, "km/h", "speed"), "t": q(15, "s", "time")}}
CAR = {"action": "call_tool", "tool": "phys.evaluate", "title": "distance",
       "args": {"expr": "v*t", "values": {"v": q(72, "km/h"), "t": q(15, "s")}, "to_unit": "m"}}


def wrong_unit(res):
    return {**res, "unit": "s", "si_unit": "second", "dims": {"[time]": 1}}


def retry_car(prompt):
    bad = last_handle(prompt, "phys.evaluate", status="failed")
    return {**CAR, "args": {**CAR["args"], "method": "si_first"}, "replaces": bad, "title": "distance (SI first)"}


def finish_car(prompt):
    h = last_handle(prompt, "phys.evaluate", status="verified")
    return {"action": "finish", "answer_template": f"The car travels {{{{{h}}}}}.", "answer_nodes": [h]}


def test_injected_unit_error_is_caught_and_recovered(make_rt, runner):
    faulty = FaultyRunner(runner, "phys.evaluate", corrupt=wrong_unit, calls={1})
    llm = ScriptedLLM([CAR_FORMAL, CAR, retry_car, finish_car])
    rt = make_rt(llm, run=faulty)
    rt.engine.new_session("unit")
    res = rt.controller.run(CAR_Q)

    assert faulty.corrupted == [1]
    assert res.status == "answered" and res.verified, res
    assert res.answer == "The car travels 300 m."
    assert "physics worker" in llm.calls[2]["system"]  # the retry went to the physics role
    bad, good = nodes(rt, "phys.evaluate")
    assert bad.status == Status.FAILED and bad.ladder_stage == LadderStage.RETRIED
    assert methods(rt, bad.id)["dimensional"] == "fail"
    assert good.lineage_id == bad.lineage_id and good.status == Status.VERIFIED
    assert good.tool_inputs["method"] == "si_first"
    assert bad.id in rt.engine.nodes  # the failed node is kept and shown


# 4 ---------------------------------------------------------------- rejecting an assumption
def test_rejecting_an_assumption_invalidates_everything_built_on_it(make_rt):
    llm = ScriptedLLM([PROJECTILE_FORMAL, CONST_G, range_step, finish_range])
    rt = make_rt(llm)
    first = rt.engine.new_session("a")
    res = run_projectile(rt)
    assert res.status == "answered" and res.verified
    rng = nodes(rt, "phys.evaluate")[0]
    fp = rng.fingerprint

    # Another session reuses the verified answer.
    rt.engine.new_session("b")
    other = rt.engine.session_id
    again = rt.controller.run(PROJECTILE_Q)
    assert again.status == "reused" and again.llm_calls == 0
    assert again.assumptions == list(DEFAULTS["projectile"])

    rt.engine.open_session(first)
    air = next(a for a in nodes(rt, type_=NodeType.ASSUMPTION) if a.content == "no air resistance")
    report = rt.engine.reject_assumption(air.id)

    assert rt.engine.resolve(air.id).status == Status.FAILED
    for nid in (rt.engine.root().id, rng.id, res.final_node):
        assert rt.repo.get_node(nid).status == Status.INVALIDATED, nid
    assert res.final_node in report.invalidated and other in report.affected_sessions
    assert any("invalidated" in n["text"] for n in rt.repo.notices(other))
    # Every result of the problem depends on its problem node, the constant included.
    g = nodes(rt, "phys.constant")[0]
    assert rt.repo.get_node(g.id).status == Status.INVALIDATED

    # Nothing is silently reused: the question and the tool call both recompute.
    assert lookup.verified_answer_for_question(rt.repo, PROJECTILE_Q) is None
    assert lookup.verified_by_fingerprint(rt.repo, fp) is None
    llm.steps = [PROJECTILE_FORMAL, CONST_G, range_step, finish_range]
    rt.engine.new_session("c")
    third = rt.controller.run(PROJECTILE_Q)
    assert third.status == "answered" and third.verified and third.llm_calls == 4
    assert nodes(rt, "phys.evaluate")[-1].id != rng.id


# 5 ---------------------------------------------------------------- reuse across units
def convert(question, name, value, unit):
    return {"action": "formalize", "statement": question, "problem_type": "general",
            "givens": {name: q(value, unit, "speed")},
            "goal": {"tool": "phys.evaluate", "args": {"expr": name, "values": {name: q(value, unit)},
                                                         "to_unit": "m/s"}}}


def test_reuse_across_units(make_rt):
    q1, q2, q3 = ("Convert 60 mph to m/s", "How many m/s is 96.56064 km/h?", "How many m/s is 96.56 km/h?")
    llm = ScriptedLLM([convert(q1, "v", 60, "mph")])
    rt = make_rt(llm)
    rt.engine.new_session("r1")
    first = rt.controller.run(q1)
    assert first.status == "answered" and first.verified and first.llm_calls == 1
    original = nodes(rt, "phys.evaluate")[0]
    assert original.result["value"] == pytest.approx(26.8224)

    # Exactly 60 mph (a mile is exactly 1.609344 km): one formalize call, no tool run.
    llm.steps = [convert(q2, "speed", 96.56064, "km/h")]
    rt.engine.new_session("r2")
    calls_before = len(rt.runner.calls) if hasattr(rt.runner, "calls") else None
    second = rt.controller.run(q2)
    assert second.status == "answered" and second.verified and second.llm_calls == 1
    assert [n.id for n in nodes(rt, "phys.evaluate")] == [original.id]  # the same node, linked
    if calls_before is not None:
        assert "phys.evaluate" not in rt.runner.calls[calls_before:]

    # 96.56 km/h is 26.8222 m/s: a different input, so it is computed afresh.
    llm.steps = [convert(q3, "speed", 96.56, "km/h")]
    rt.engine.new_session("r3")
    third = rt.controller.run(q3)
    assert third.status == "answered" and third.verified
    fresh = nodes(rt, "phys.evaluate")
    assert len(fresh) == 1 and fresh[0].id != original.id
    assert fresh[0].result["value"] == pytest.approx(26.82222222, rel=1e-8)


# extra ------------------------------------------------------------------ circuits
DIV_Q = "A 10 V source drives 100 ohm and 150 ohm resistors in series. What is the voltage across the 150 ohm one?"
NETLIST = {"elements": [{"name": "V1", "type": "V", "n1": "1", "n2": "0", "value": q(10, "V")},
                        {"name": "R1", "type": "R", "n1": "1", "n2": "2", "value": q(100, "ohm")},
                        {"name": "R2", "type": "R", "n1": "2", "n2": "0", "value": q(150, "ohm")}], "ground": "0"}
DIV_FORMAL = {"action": "formalize", "statement": DIV_Q, "problem_type": "dc_circuit",
              "givens": {"V1": q(10, "V", "voltage"), "R1": q(100, "ohm", "resistance"),
                         "R2": q(150, "ohm", "resistance")}}


def dc(outputs):
    return {"action": "call_tool", "tool": "circuit.dc", "args": {"netlist": NETLIST, "outputs": outputs}}


def finish_dc(prompt):
    a = [h for h in (last_handle(prompt, "circuit.dc"),)][0]
    return {"action": "finish", "answer_template": f"Across the resistor: {{{{{a}}}}}.", "answer_nodes": [a]}


def test_circuit_netlist_is_an_entity(make_rt):
    llm = ScriptedLLM([DIV_FORMAL, dc(["I(R1)"]), dc(["V(2)"]), finish_dc])
    rt = make_rt(llm)
    rt.engine.new_session("dc")
    res = rt.controller.run(DIV_Q)

    assert res.status == "answered" and res.verified, res
    assert res.answer == "Across the resistor: V(2) = 6 V."
    entities = nodes(rt, type_=NodeType.ENTITY)
    assert len(entities) == 1 and entities[0].layer == Layer.DOMAIN  # one netlist, reused by both calls
    assert "R2 2-0 150 ohm" in entities[0].content
    for n in nodes(rt, "circuit.dc"):
        assert entities[0].id in rt.engine.dependencies(n.id)
        assert n.status == Status.VERIFIED and not n.flags
        assert methods(rt, n.id) == {"kirchhoff": "pass", "power_balance": "pass", "series_parallel": "pass"}
    assert "netlist=n" in llm.calls[-1]["user"]  # the digest points at the entity, not the JSON
