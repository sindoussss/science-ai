"""The Phase 2 acceptance tests, against a scripted model and the real tool sandbox.

Phase 2's subject is physics and engineering: quantities with units, the default-assumptions
checklist, reuse across equivalent units, the circuit netlist as an entity, and recovery from a
unit error. The router replaced open-ended formalization, so a physics question now enters
through a recipe whose formula lives in code and whose problem type names its own checklist.
"""
from __future__ import annotations

import math

import pytest

from sciai.controller import lookup
from sciai.domains.physics.assumptions import DEFAULTS
from sciai.graph.model import Layer, LadderStage, NodeType, Status
from sciai.tools.faults import FaultyRunner
from tests.conftest import numbers
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
    allowed |= numbers((rt.engine.root().tool_inputs or {}).get("question", ""))
    assert numbers(final.content) <= allowed, (final.content, allowed)


# 1 ---------------------------------------------------------------- projectile
PROJECTILE_Q = "A ball is thrown at 20 m/s at 30 degrees; how far does it land?"
PROJECTILE_ROUTE = {
    "action": "formalize", "recipe": "projectile_range",
    "statement": "Range of a projectile launched at 20 m/s, 30 degrees above level ground.",
    "slots": {"v0": "20", "v0_unit": "m/s", "angle": "30", "angle_unit": "deg", "to_unit": "m"},
}


def run_projectile(rt):
    return rt.controller.run(PROJECTILE_Q)


def test_projectile_with_units_end_to_end(make_rt):
    llm = ScriptedLLM([PROJECTILE_ROUTE])
    rt = make_rt(llm)
    rt.engine.new_session("p1")
    res = run_projectile(rt)

    assert res.status == "answered" and res.verified, res
    assert res.llm_calls == 1  # the router; the formula is the recipe's, not the model's
    expected = 400 * math.sin(math.radians(60)) / G0  # 35.32 m; sin(30 rad) would give -12.4 m
    rng = nodes(rt, "phys.projectile_range")[0]
    assert rng.result["value"] == pytest.approx(expected, rel=1e-12) and rng.result["unit"] == "m"
    assert res.answer == "The range is 35.32 m."
    # The range is checked three ways, and the simulation is the independent one: it integrates
    # the motion instead of re-using the closed form.
    assert methods(rt, rng.id) == {"target_unit": "pass", "plausibility": "pass",
                                   "simulation": "pass"}
    assert rng.result["quantity_kind"] == "distance"
    assert not rng.flags  # every value came from a slot the question gave

    # The projectile checklist became locked assumption nodes the problem depends on. The
    # recipe names the problem type, so this no longer depends on the model saying so.
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
    llm = ScriptedLLM([{**PROJECTILE_ROUTE, "modelling_assumptions": ["no spin"]}])
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
RC_ROUTE = {
    "action": "formalize", "recipe": "rc_discharge",
    "statement": "RC discharge: C = 1 uF, R = 1 kohm, V0 = 5 V; find v(2 ms).",
    "slots": {"v0": "5", "v0_unit": "V", "resistance": "1", "resistance_unit": "kohm",
              "capacitance": "1", "capacitance_unit": "uF", "t": "2", "t_unit": "ms",
              "to_unit": "V"},
}


def test_rc_discharge(make_rt):
    llm = ScriptedLLM([RC_ROUTE])
    rt = make_rt(llm)
    rt.engine.new_session("rc")
    res = rt.controller.run(RC_Q)

    assert res.status == "answered" and res.verified, res
    assert res.llm_calls == 1
    v = nodes(rt, "phys.rc_discharge")[0]
    assert v.result["value"] == pytest.approx(5 * math.exp(-2), rel=1e-12)  # 0.6767 V
    assert v.result["unit"] == "V"
    # The check solves dV/dt = -V/(R*C) numerically instead of evaluating the closed form.
    assert methods(rt, v.id) == {"target_unit": "pass", "plausibility": "pass",
                                 "symbolic_vs_numeric": "pass"}
    assert res.answer == "The voltage after that time is 0.6767 V."
    assert v.status == Status.VERIFIED and not v.flags
    assert [a.content for a in nodes(rt, type_=NodeType.ASSUMPTION)] == \
        list(DEFAULTS["rc_rl_transient"])
    assert_numbers_from_tools(rt, res.final_node)


def test_a_failed_recipe_step_recovers_with_a_numeric_solve_and_a_plot(make_rt, runner):
    """A recipe's closed form fails its check, and the physics specialist recovers.

    The ladder picks the role for the tool that failed, so the retry comes from the physics
    worker, which reaches the same voltage by integrating the circuit's differential equation
    and then plots the run. This is also what keeps ``ode.solve_ivp`` and ``plot.series``
    exercised end to end now that the happy path is one closed-form call.
    """
    def wrong_voltage(res):
        return {**res, "value": 4.0, "si_value": 4.0}

    ivp = {"action": "call_tool", "tool": "ode.solve_ivp", "title": "v(t) numeric",
           "args": {"rhs": ["-v/(1000*1e-6)"], "funcs": ["v"], "var": "t", "y0": ["5"],
                    "t_span": ["0", "0.002"]}}

    def plot_rc(prompt):
        return {"action": "call_tool", "tool": "plot.series", "title": "v(t) plot",
                "args": {"node": last_handle(prompt, "ode.solve_ivp"), "component": "v"}}

    def finish_rc(prompt):
        h = last_handle(prompt, "ode.solve_ivp")
        return {"action": "finish", "answer_template": f"The voltage is {{{{{h}}}}}.",
                "answer_nodes": [h]}

    faulty = FaultyRunner(runner, "phys.rc_discharge", corrupt=wrong_voltage, calls={1})
    llm = ScriptedLLM([RC_ROUTE, ivp, plot_rc, finish_rc])
    rt = make_rt(llm, run=faulty)
    rt.engine.new_session("rc2")
    res = rt.controller.run(RC_Q)

    assert faulty.corrupted == [1]
    assert res.status == "answered" and res.verified, res
    assert "physics worker" in llm.calls[1]["system"]  # the retry went to the physics role
    bad = nodes(rt, "phys.rc_discharge")[0]
    assert bad.status == Status.FAILED and bad.ladder_stage == LadderStage.RETRIED
    assert methods(rt, bad.id)["symbolic_vs_numeric"] == "fail"
    solved = nodes(rt, "ode.solve_ivp")[0]
    assert solved.status == Status.VERIFIED
    assert solved.result["value"][0] == pytest.approx(5 * math.exp(-2), rel=1e-6)
    assert methods(rt, solved.id) == {"alt_algorithm": "pass", "symbolic_vs_numeric": "pass"}
    plot = nodes(rt, "plot.series")[0]
    assert plot.type == NodeType.PLOT and plot.tool_inputs["node"] == solved.id
    assert solved.id in rt.engine.dependencies(plot.id)
    assert plot.result["value"]["y"][-1] == pytest.approx(solved.result["value"][0])


# 3 ---------------------------------------------------------------- injected unit error
def test_injected_unit_error_is_caught_and_recovered(make_rt, runner):
    """A range returned in the wrong unit fails the node, and the retry gets it right.

    The requested unit is a slot of the recipe, so "the right number in another unit" is a
    failure and not a detail: the answer sentence does not name the unit, so a result in the
    wrong one would read as the answer to the question that was asked.
    """
    def wrong_unit(res):
        return {**res, "unit": "m/h", "si_unit": "meter / hour", "dims": {"[length]": 1, "[time]": -1}}

    def retry_range(prompt):
        bad = last_handle(prompt, "phys.projectile_range", status="failed")
        return {"action": "call_tool", "tool": "phys.evaluate", "replaces": bad,
                "title": "range by the formula",
                "args": {"expr": "v0**2*sin(2*theta)/g", "to_unit": "m", "kind": "distance",
                         "values": {"v0": q(20, "m/s"), "theta": q(30, "deg"),
                                    "g": q(G0, "m/s^2")}}}

    def finish_range(prompt):
        h = last_handle(prompt, "phys.evaluate", status="verified")
        return {"action": "finish", "answer_template": f"The ball lands {{{{{h}}}}} away.",
                "answer_nodes": [h]}

    faulty = FaultyRunner(runner, "phys.projectile_range", corrupt=wrong_unit, calls={1})
    llm = ScriptedLLM([PROJECTILE_ROUTE, retry_range, finish_range])
    rt = make_rt(llm, run=faulty)
    rt.engine.new_session("unit")
    res = run_projectile(rt)

    assert faulty.corrupted == [1]
    assert res.status == "answered" and res.verified, res
    assert res.answer == "The ball lands 35.32 m away."
    assert "physics worker" in llm.calls[1]["system"]
    bad = nodes(rt, "phys.projectile_range")[0]
    assert bad.status == Status.FAILED and bad.ladder_stage == LadderStage.RETRIED
    assert methods(rt, bad.id)["target_unit"] == "fail"
    good = nodes(rt, "phys.evaluate")[0]
    assert good.lineage_id == bad.lineage_id and good.status == Status.VERIFIED
    assert bad.id in rt.engine.nodes  # the failed node is kept and shown


# 4 ---------------------------------------------------------------- rejecting an assumption
def test_rejecting_an_assumption_invalidates_everything_built_on_it(make_rt):
    llm = ScriptedLLM([PROJECTILE_ROUTE])
    rt = make_rt(llm)
    first = rt.engine.new_session("a")
    res = run_projectile(rt)
    assert res.status == "answered" and res.verified
    rng = nodes(rt, "phys.projectile_range")[0]
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

    # Nothing is silently reused: the question and the tool call both recompute.
    assert lookup.verified_answer_for_question(rt.repo, PROJECTILE_Q) is None
    assert lookup.reusable_by_fingerprint(rt.repo, fp) is None
    llm.steps = [PROJECTILE_ROUTE]
    rt.engine.new_session("c")
    third = rt.controller.run(PROJECTILE_Q)
    assert third.status == "answered" and third.verified and third.llm_calls == 1
    assert nodes(rt, "phys.projectile_range")[-1].id != rng.id


# 5 ---------------------------------------------------------------- reuse across units
def photon(question, value, unit):
    return {"action": "formalize", "recipe": "photon_energy", "statement": question,
            "slots": {"wavelength": value, "wavelength_unit": unit, "to_unit": "J"}}


def test_reuse_across_units(make_rt):
    """A slot's number and unit are canonicalized to SI before fingerprinting, so the same
    wavelength written two ways is the same tool call and the second one runs nothing."""
    q1 = "What is the energy of a photon of wavelength 500 nm, in joules?"
    q2 = "A photon has a wavelength of 0.5 um. What is its energy in joules?"
    q3 = "A photon has a wavelength of 0.501 um. What is its energy in joules?"
    llm = ScriptedLLM([photon(q1, "500", "nm")])
    rt = make_rt(llm)
    rt.engine.new_session("r1")
    first = rt.controller.run(q1)
    assert first.status == "answered" and first.verified and first.llm_calls == 1
    original = nodes(rt, "phys.photon_energy")[0]

    llm.steps = [photon(q2, "0.5", "um")]
    rt.engine.new_session("r2")
    calls_before = len(rt.runner.calls)
    second = rt.controller.run(q2)
    assert second.status == "answered" and second.verified and second.llm_calls == 1
    assert [n.id for n in nodes(rt, "phys.photon_energy")] == [original.id]  # the same node, linked
    assert "phys.photon_energy" not in rt.runner.calls[calls_before:]

    # 0.501 um is a different wavelength, so it is computed afresh.
    llm.steps = [photon(q3, "0.501", "um")]
    rt.engine.new_session("r3")
    third = rt.controller.run(q3)
    assert third.status == "answered" and third.verified
    fresh = nodes(rt, "phys.photon_energy")
    assert len(fresh) == 1 and fresh[0].id != original.id


# extra ------------------------------------------------------------------ circuits
SERIES_Q = "What current flows when 12 V is across a 100 ohm and a 220 ohm resistor in series?"
SERIES_ROUTE = {"action": "formalize", "recipe": "series_parallel_current", "statement": SERIES_Q,
                "slots": {"voltage": "12", "voltage_unit": "V", "series": "100, 220",
                          "series_unit": "ohm"}}


def test_circuit_netlist_is_an_entity(make_rt):
    """The recipe builds the netlist, and it still becomes the domain entity the result
    depends on, so a new version of the circuit cascades to everything computed from it."""
    llm = ScriptedLLM([SERIES_ROUTE])
    rt = make_rt(llm)
    rt.engine.new_session("dc")
    res = rt.controller.run(SERIES_Q)

    assert res.status == "answered" and res.verified, res
    assert res.answer == "The current through the circuit is I(Rs1) = 0.0375 A, I(Rs2) = 0.0375 A."
    entities = nodes(rt, type_=NodeType.ENTITY)
    assert len(entities) == 1 and entities[0].layer == Layer.DOMAIN
    assert "Rs2 m1-0 220 ohm" in entities[0].content
    dc = nodes(rt, "circuit.dc")[0]
    assert entities[0].id in rt.engine.dependencies(dc.id)
    assert dc.status == Status.VERIFIED and not dc.flags
    assert methods(rt, dc.id) == {"kirchhoff": "pass", "power_balance": "pass",
                                  "series_parallel": "pass"}
    assert [a.content for a in nodes(rt, type_=NodeType.ASSUMPTION)] == list(DEFAULTS["dc_circuit"])
