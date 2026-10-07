"""The Phase 1 acceptance tests, against a scripted model and the real tool sandbox.

Phase 1's subject is the controller: one tool call at a time, an independent check on every
result, and the failure ladder (retry with a different method, backtrack, escalate). Since the
router replaced open-ended formalization, a question enters through a recipe, so these tests
route one and then exercise what happens after. A recipe whose tool fails its check hands over
to the ladder, which is why the recovery tests below still drive the step-by-step loop.
"""
from __future__ import annotations

from sciai.graph.model import EdgeKind, LadderStage, NodeType, Status
from sciai.tools.faults import FaultyRunner
from tests.conftest import numbers
from tests.fakes.fake_llm import ScriptedLLM, last_handle

# One model call, two tool calls: the recipe solves the initial value problem and then
# evaluates the solution at the point the question asked about.
QUESTION = "Solve dy/dx = -2*y with y(0) = 5, then find y(3)."
ROUTE = {"action": "formalize", "recipe": "ode_ivp",
         "statement": "Solve y' = -2*y with y(0) = 5, then evaluate y(3).",
         "slots": {"dy_dx": "-2*y", "func": "y", "var": "x", "y0": "5", "at": "0",
                   "evaluate_at": "3"}}

INTEGRAL_Q = "Compute the definite integral of x^2*sin(x) from x = 0 to x = pi."
INTEGRAL_ROUTE = {"action": "formalize", "recipe": "definite_integral",
                  "statement": "Compute the definite integral of x**2*sin(x) for x from 0 to pi.",
                  "slots": {"integrand": "x**2*sin(x)", "var": "x", "lower": "0", "upper": "pi"}}


def retry_integral(method: str):
    """What the ladder asks for: the same integral by a method the failed attempt did not use."""
    def step(prompt: str) -> dict:
        bad = last_handle(prompt, "sympy.integrate")
        return {"action": "call_tool", "tool": "sympy.integrate", "replaces": bad,
                "args": {"expr": "x**2*sin(x)", "var": "x", "lower": "0", "upper": "pi",
                         "method": method},
                "title": f"the integral via {method}"}
    return step


def finish_integral(prompt: str) -> dict:
    h = last_handle(prompt, "sympy.integrate", status="proposed")
    return {"action": "finish", "answer_template": f"The integral is {{{{{h}}}}}.",
            "answer_nodes": [h]}


def assert_numbers_from_tools(rt, final_id: str) -> None:
    final = rt.engine.resolve(final_id)
    allowed = set()
    for nid in final.tool_inputs["answer_nodes"]:
        allowed |= numbers(rt.engine.resolve(nid).display_result())
    allowed |= numbers((rt.engine.root().tool_inputs or {}).get("question", ""))
    assert numbers(final.content) <= allowed, (final.content, allowed)


# 1 ----------------------------------------------------------------------------
def test_multistep_problem_end_to_end(make_rt):
    llm = ScriptedLLM([ROUTE])
    rt = make_rt(llm)
    rt.engine.new_session("t1")
    res = rt.controller.run(QUESTION)

    assert res.status == "answered", res
    assert res.verified
    assert res.answer.startswith("y(3) = ")
    assert "0.01239" in res.answer
    # The recipe is the plan, so the whole question costs the one call that chose it.
    assert res.llm_calls == 1
    types = [n.type for n in rt.engine.nodes.values()]
    assert types.count(NodeType.PROBLEM) == 1
    assert types.count(NodeType.TOOL_RESULT) == 2
    assert types.count(NodeType.CHECK) == 3
    assert types.count(NodeType.FINAL) == 1
    for n in rt.engine.nodes.values():
        if n.type in (NodeType.TOOL_RESULT, NodeType.FINAL):
            assert n.status == Status.VERIFIED, n
    assert_numbers_from_tools(rt, res.final_node)


# 2 ----------------------------------------------------------------------------
def test_injected_wrong_intermediate_is_caught_and_recovered(make_rt, runner):
    faulty = FaultyRunner(runner, "sympy.integrate", "pi**2 - 2", calls={1})
    llm = ScriptedLLM([INTEGRAL_ROUTE, retry_integral("meijerg"), finish_integral])
    rt = make_rt(llm, run=faulty)
    rt.engine.new_session("t2")
    res = rt.controller.run(INTEGRAL_Q)

    assert faulty.corrupted == [1]
    assert res.status == "answered" and res.verified, res
    assert "-4 + pi**2" in res.answer
    integrals = [n for n in rt.engine.nodes.values() if n.tool_name == "sympy.integrate"]
    lineage = rt.repo.lineage(integrals[0].lineage_id)
    assert [v.status for v in lineage] == [Status.FAILED, Status.VERIFIED]
    assert lineage[0].ladder_stage == LadderStage.RETRIED
    assert_numbers_from_tools(rt, res.final_node)


def test_a_failed_recipe_step_retries_with_the_specialist_role(make_rt, runner):
    """The ladder picks the specialist for the failed tool, and the retry comes from it.

    This used to be lost: a tool call made outside the step-by-step loop that failed its check
    dropped back to the general controller, which knows the least about the tool that failed.
    """
    faulty = FaultyRunner(runner, "sympy.integrate", "pi**2 - 2", calls={1})
    llm = ScriptedLLM([INTEGRAL_ROUTE, retry_integral("meijerg"), finish_integral])
    rt = make_rt(llm, run=faulty)
    rt.engine.new_session("t2r")
    assert rt.controller.run(INTEGRAL_Q).status == "answered"
    assert "You are the algebra worker" in llm.calls[1]["system"]


def test_backtrack_when_retry_also_fails(make_rt, runner):
    faulty = FaultyRunner(runner, "sympy.integrate", "pi**2 - 2", calls={1, 2})
    llm = ScriptedLLM([INTEGRAL_ROUTE, retry_integral("meijerg"), finish_integral,
                       retry_integral("manual"), finish_integral])
    rt = make_rt(llm, run=faulty)
    rt.engine.new_session("t2b")
    res = rt.controller.run(INTEGRAL_Q)

    assert res.status == "answered" and res.verified, res
    integrals = [n for n in rt.engine.nodes.values() if n.tool_name == "sympy.integrate"]
    lineage = rt.repo.lineage(integrals[0].lineage_id)
    assert len(lineage) == 3
    assert lineage[1].status == Status.INVALIDATED and lineage[1].ladder_stage == LadderStage.BACKTRACKED
    assert lineage[2].status == Status.VERIFIED


def test_escalates_when_backtrack_fails(make_rt, runner):
    faulty = FaultyRunner(runner, "sympy.integrate", "pi**2 - 2", calls=None)
    llm = ScriptedLLM([INTEGRAL_ROUTE, retry_integral("meijerg"), finish_integral,
                       retry_integral("manual"), finish_integral,
                       retry_integral("heurisch"), finish_integral])
    rt = make_rt(llm, run=faulty)
    rt.engine.new_session("t2c")
    res = rt.controller.run(INTEGRAL_Q)

    assert res.status == "escalated", res
    assert len(res.conflict) >= 2
    kinds = {d["kind"] for _, _, d in rt.engine.g.edges(data=True)}
    assert EdgeKind.CONFLICTS_WITH in kinds
    assert llm.steps  # stopped instead of using every scripted retry


# 3 ----------------------------------------------------------------------------
def test_reuses_verified_answer_without_model(make_rt):
    rt = make_rt(ScriptedLLM([ROUTE]))
    rt.engine.new_session("first")
    first = rt.controller.run(QUESTION)
    assert first.status == "answered" and first.verified

    rt.llm.steps = []  # any model call would now fail the test
    rt.controller.llm = rt.llm
    rt.engine.new_session("second")
    tool_calls_before = len(rt.runner.calls)
    again = rt.controller.run("solve  dy/dx = -2*y with y(0) = 5, then find y(3).")
    assert again.status == "reused"
    assert again.llm_calls == 0
    assert len(rt.runner.calls) == tool_calls_before
    assert again.answer == first.answer


def test_reuses_a_verified_tool_result_a_paraphrase_routes_to(make_rt):
    """Two questions that fill the same slots make the same tool call, so the second reuses the
    first's verified result: the recipe is what makes the two comparable."""
    route = {"action": "formalize", "recipe": "definite_integral",
             "statement": "The antiderivative of x**2*sin(x) between 0 and pi.",
             "slots": {"integrand": "x**2*sin(x)", "var": "x", "lower": "0", "upper": "pi"}}
    rt = make_rt(ScriptedLLM([INTEGRAL_ROUTE]))
    rt.engine.new_session("a")
    first = rt.controller.run(INTEGRAL_Q)
    assert first.status == "answered" and first.verified and first.llm_calls == 1

    rt.controller.llm = ScriptedLLM([{**route, "slots": {"integrand": "sin(x) * x^2", "var": "x",
                                                         "lower": "0", "upper": "pi"}}])
    rt.engine.new_session("b")
    before = list(rt.runner.calls)
    second = rt.controller.run("What is the integral of x squared times sine x from 0 to pi?")
    assert second.status == "answered" and second.verified
    assert second.llm_calls == 1  # the router only; no model call for the maths
    assert "sympy.integrate" not in rt.runner.calls[len(before):]
    assert first.answer == second.answer


# 4 ----------------------------------------------------------------------------
def test_persistence_verified_kept_unverified_flagged(make_rt, tmp_path, cfg, runner):
    db = tmp_path / "persist.db"
    rt = make_rt(ScriptedLLM([ROUTE]), db_path=db)
    sid = rt.engine.new_session("persist")
    res = rt.controller.run(QUESTION)
    assert res.verified

    # A recipe step that fails its check hands over to the loop, and this session stops there
    # with an unfinished, unchecked result still on the graph.
    faulty = FaultyRunner(runner, "sympy.integrate", "pi**2 - 2", calls={1})
    rt2 = make_rt(ScriptedLLM([INTEGRAL_ROUTE,
                               {"action": "call_tool", "tool": "sympy.expand",
                                "args": {"expr": "(x+1)**3"}},
                               {"action": "ask_user", "question": "Which bounds did you mean?"}]),
                  db_path=db, run=faulty)
    sid2 = rt2.engine.new_session("unfinished")
    pending = rt2.controller.run(INTEGRAL_Q)
    assert pending.status == "needs_user"
    verified_ids = {n.id for n in rt2.engine.repo.get_nodes([x.id for x, _ in rt2.repo.session_view(sid)])
                    if n.status == Status.VERIFIED}
    assert verified_ids
    rt2.close()

    rt3 = make_rt(ScriptedLLM([]), db_path=db)
    assert rt3.flagged_on_start >= 1
    rt3.engine.open_session(sid)
    assert {n.id for n in rt3.engine.nodes.values() if n.status == Status.VERIFIED} == verified_ids
    rt3.engine.open_session(sid2)
    expand = next(n for n in rt3.engine.nodes.values() if n.tool_name == "sympy.expand")
    assert expand.status == Status.PROPOSED and expand.needs_recheck and expand.warning
    rt3.engine.new_session("after reopen")
    again = rt3.controller.run(QUESTION)
    assert again.status == "reused" and again.llm_calls == 0


# 5 ----------------------------------------------------------------------------
def test_model_cannot_supply_numeric_results(make_rt, runner):
    """In the loop the model still may not type a value, nor use a number from nowhere."""
    def bad_finish(prompt: str) -> dict:
        h = last_handle(prompt, "sympy.integrate", status="proposed")
        return {"action": "finish", "answer_template": "The integral is 42.", "answer_nodes": [h]}

    def unsourced_subs(prompt: str) -> dict:
        h = last_handle(prompt, "sympy.integrate", status="proposed")
        return {"action": "call_tool", "tool": "sympy.subs",
                "args": {"expr": "x**2*sin(x)", "values": {"x": "3.7"}}, "depends_on": [h]}

    faulty = FaultyRunner(runner, "sympy.integrate", "pi**2 - 2", calls={1})
    llm = ScriptedLLM([INTEGRAL_ROUTE, retry_integral("meijerg"), unsourced_subs, bad_finish,
                       finish_integral])
    rt = make_rt(llm, run=faulty)
    rt.engine.new_session("t5")
    res = rt.controller.run(INTEGRAL_Q)
    assert res.status == "answered", res
    assert "42" not in res.answer
    assert any("may not contain the number(s) 42" in c["user"] for c in llm.calls)
    sub = next(n for n in rt.engine.nodes.values() if n.tool_name == "sympy.subs")
    assert any(f.startswith("unsourced_numbers: 3.7") for f in sub.flags)
    assert rt.repo.evidence_for(sub.id), "flagged node must be checked by the verifier"
    assert_numbers_from_tools(rt, res.final_node)
