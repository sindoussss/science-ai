"""The five Phase 1 acceptance tests, against a scripted model and the real tool sandbox."""
from __future__ import annotations

from sciai.graph.model import EdgeKind, LadderStage, NodeType, Status
from sciai.tools.faults import FaultyRunner
from tests.conftest import numbers, result_of
from tests.fakes.fake_llm import ScriptedLLM, last_handle

QUESTION = "Differentiate x^2*sin(x) and evaluate the derivative at x = pi"
FORMAL = {"action": "formalize", "statement": "Compute f'(x) for f(x) = x**2*sin(x), then evaluate f'(pi)."}
DIFF = {"action": "call_tool", "thought": "differentiate", "tool": "sympy.diff",
        "args": {"expr": "x**2*sin(x)", "var": "x"}, "depends_on": ["n1"], "title": "f'(x)"}


def subs_step(prompt: str) -> dict:
    h = last_handle(prompt, "sympy.diff", status="proposed")
    return {"action": "call_tool", "tool": "sympy.subs", "args": {"expr": result_of(prompt, h), "values": {"x": "pi"}},
            "depends_on": [h], "title": "f'(pi)"}


def finish_both(prompt: str) -> dict:
    d, s = last_handle(prompt, "sympy.diff"), last_handle(prompt, "sympy.subs")
    return {"action": "finish", "answer_template": f"f'(x) = {{{{{d}}}}}, so f'(pi) = {{{{{s}}}}}.",
            "answer_nodes": [d, s]}


def finish_value(prompt: str) -> dict:
    s = last_handle(prompt, "sympy.subs")
    return {"action": "finish", "answer_template": f"f'(pi) = {{{{{s}}}}}.", "answer_nodes": [s]}


def retry_diff(method: str):
    def step(prompt: str) -> dict:
        bad = last_handle(prompt, "sympy.diff")
        return {**DIFF, "args": {**DIFF["args"], "method": method}, "replaces": bad, "title": f"f'(x) via {method}"}
    return step


def assert_numbers_from_tools(rt, final_id: str) -> None:
    final = rt.engine.resolve(final_id)
    allowed = set()
    for nid in final.tool_inputs["answer_nodes"]:
        allowed |= numbers(rt.engine.resolve(nid).display_result())
    assert numbers(final.content) <= allowed, (final.content, allowed)


# 1 ----------------------------------------------------------------------------
def test_multistep_problem_end_to_end(make_rt):
    llm = ScriptedLLM([FORMAL, DIFF, subs_step, finish_both])
    rt = make_rt(llm)
    rt.engine.new_session("t1")
    res = rt.controller.run(QUESTION)

    assert res.status == "answered", res
    assert res.verified
    assert "-pi**2" in res.answer and "x**2*cos(x) + 2*x*sin(x)" in res.answer
    assert res.llm_calls == 4
    types = [n.type for n in rt.engine.nodes.values()]
    assert types.count(NodeType.PROBLEM) == 1
    assert types.count(NodeType.TOOL_RESULT) == 2
    assert types.count(NodeType.CHECK) == 2
    assert types.count(NodeType.FINAL) == 1
    for n in rt.engine.nodes.values():
        if n.type in (NodeType.TOOL_RESULT, NodeType.FINAL):
            assert n.status == Status.VERIFIED, n
    assert_numbers_from_tools(rt, res.final_node)


# 2 ----------------------------------------------------------------------------
def test_injected_wrong_intermediate_is_caught_and_recovered(make_rt, runner):
    faulty = FaultyRunner(runner, "sympy.diff", "2*x*sin(x)", calls={1})
    llm = ScriptedLLM([FORMAL, DIFF, subs_step, finish_value, retry_diff("first_principles"), subs_step,
                       finish_value])
    rt = make_rt(llm, run=faulty)
    rt.engine.new_session("t2")
    res = rt.controller.run(QUESTION)

    assert faulty.corrupted == [1]
    assert res.status == "answered" and res.verified, res
    assert "-pi**2" in res.answer
    diffs = [n for n in rt.engine.nodes.values() if n.tool_name == "sympy.diff"]
    lineage = rt.repo.lineage(diffs[0].lineage_id)
    assert [v.status for v in lineage] == [Status.FAILED, Status.VERIFIED]
    assert lineage[0].ladder_stage == LadderStage.RETRIED
    subs = sorted((n for n in rt.engine.nodes.values() if n.tool_name == "sympy.subs"), key=lambda n: n.created_at)
    assert subs[0].status == Status.INVALIDATED  # built on the bad value
    assert subs[1].status == Status.VERIFIED
    assert_numbers_from_tools(rt, res.final_node)


def test_backtrack_when_retry_also_fails(make_rt, runner):
    faulty = FaultyRunner(runner, "sympy.diff", "2*x*sin(x)", calls={1, 2})
    llm = ScriptedLLM([FORMAL, DIFF, subs_step, finish_value,
                       retry_diff("first_principles"), subs_step, finish_value,
                       retry_diff("expand_first"), subs_step, finish_value])
    rt = make_rt(llm, run=faulty)
    rt.engine.new_session("t2b")
    res = rt.controller.run(QUESTION)

    assert res.status == "answered" and res.verified, res
    diffs = [n for n in rt.engine.nodes.values() if n.tool_name == "sympy.diff"]
    lineage = rt.repo.lineage(diffs[0].lineage_id)
    assert len(lineage) == 3
    assert lineage[1].status == Status.INVALIDATED and lineage[1].ladder_stage == LadderStage.BACKTRACKED
    assert lineage[2].status == Status.VERIFIED


def test_escalates_when_backtrack_fails(make_rt, runner):
    faulty = FaultyRunner(runner, "sympy.diff", "2*x*sin(x)", calls=None)
    llm = ScriptedLLM([FORMAL, DIFF, subs_step, finish_value,
                       retry_diff("first_principles"), subs_step, finish_value,
                       retry_diff("expand_first"), subs_step, finish_value])
    rt = make_rt(llm, run=faulty)
    rt.engine.new_session("t2c")
    res = rt.controller.run(QUESTION)

    assert res.status == "escalated", res
    assert len(res.conflict) >= 2
    kinds = {d["kind"] for _, _, d in rt.engine.g.edges(data=True)}
    assert EdgeKind.CONFLICTS_WITH in kinds
    assert not llm.steps  # stopped instead of looping


# 3 ----------------------------------------------------------------------------
def test_reuses_verified_answer_without_model(make_rt):
    rt = make_rt(ScriptedLLM([FORMAL, DIFF, subs_step, finish_both]))
    rt.engine.new_session("first")
    first = rt.controller.run(QUESTION)
    assert first.status == "answered" and first.verified

    rt.llm.steps = []  # any model call would now fail the test
    rt.controller.llm = rt.llm
    rt.engine.new_session("second")
    tool_calls_before = len(rt.runner.calls)
    again = rt.controller.run("differentiate x^2*sin(x)   and evaluate the derivative at x = pi.")
    assert again.status == "reused"
    assert again.llm_calls == 0
    assert len(rt.runner.calls) == tool_calls_before
    assert again.answer == first.answer


def test_reuses_verified_tool_result_by_goal(make_rt):
    goal = {"tool": "sympy.integrate", "args": {"expr": "x**2*sin(x)", "var": "x"}}
    rt = make_rt(ScriptedLLM([{"action": "formalize", "statement": "Integrate x**2*sin(x) dx", "goal": goal}]))
    rt.engine.new_session("a")
    first = rt.controller.run("Integrate x^2 sin(x) with respect to x")
    assert first.status == "answered" and first.verified and first.llm_calls == 1

    paraphrase_goal = {"tool": "sympy.integrate", "args": {"expr": "sin(x) * x^2", "var": "x"}}
    rt.controller.llm = ScriptedLLM([{"action": "formalize", "statement": "Antiderivative of sin(x)*x**2",
                                      "goal": paraphrase_goal}])
    rt.engine.new_session("b")
    before = list(rt.runner.calls)
    second = rt.controller.run("What is the antiderivative of x squared times sine x?")
    assert second.status == "answered" and second.verified
    assert second.llm_calls == 1  # formalization only; no model call for the math
    assert "sympy.integrate" not in rt.runner.calls[len(before):]
    assert first.answer == second.answer


# 4 ----------------------------------------------------------------------------
def test_persistence_verified_kept_unverified_flagged(make_rt, tmp_path, cfg, runner):
    db = tmp_path / "persist.db"
    rt = make_rt(ScriptedLLM([FORMAL, DIFF, subs_step, finish_both]), db_path=db)
    sid = rt.engine.new_session("persist")
    res = rt.controller.run(QUESTION)
    assert res.verified
    rt.controller.llm = ScriptedLLM([
        {"action": "formalize", "statement": "Expand (x+1)**3"},
        {"action": "call_tool", "tool": "sympy.expand", "args": {"expr": "(x+1)**3"}, "depends_on": ["n1"]},
        {"action": "ask_user", "question": "Do you also want it factored?"},
    ])
    sid2 = rt.engine.new_session("unfinished")
    pending = rt.controller.run("Expand (x+1)^3")
    assert pending.status == "needs_user"
    verified_ids = {n.id for n in rt.engine.repo.get_nodes([x.id for x, _ in rt.repo.session_view(sid)])
                    if n.status == Status.VERIFIED}
    rt.close()

    rt2 = make_rt(ScriptedLLM([]), db_path=db)
    assert rt2.flagged_on_start >= 1
    rt2.engine.open_session(sid)
    assert {n.id for n in rt2.engine.nodes.values() if n.status == Status.VERIFIED} == verified_ids
    rt2.engine.open_session(sid2)
    expand = next(n for n in rt2.engine.nodes.values() if n.tool_name == "sympy.expand")
    assert expand.status == Status.PROPOSED and expand.needs_recheck and expand.warning
    rt2.engine.new_session("after reopen")
    again = rt2.controller.run(QUESTION)
    assert again.status == "reused" and again.llm_calls == 0


# 5 ----------------------------------------------------------------------------
def test_model_cannot_supply_numeric_results(make_rt):
    def bad_finish(prompt: str) -> dict:
        return {"action": "finish", "answer_template": "The value at pi is 42.", "answer_nodes": ["n2"]}

    def subs_unsourced(prompt: str) -> dict:
        h = last_handle(prompt, "sympy.diff")
        return {"action": "call_tool", "tool": "sympy.subs",
                "args": {"expr": result_of(prompt, h), "values": {"x": "3.7"}}, "depends_on": [h]}

    llm = ScriptedLLM([FORMAL, DIFF, subs_unsourced, bad_finish, finish_both])
    rt = make_rt(llm)
    rt.engine.new_session("t5")
    res = rt.controller.run(QUESTION)
    assert res.status == "answered"
    assert "42" not in res.answer
    assert any("must not contain digits" in c["user"] for c in llm.calls)
    sub = next(n for n in rt.engine.nodes.values() if n.tool_name == "sympy.subs")
    assert any(f.startswith("unsourced_numbers: 3.7") for f in sub.flags)
    assert rt.repo.evidence_for(sub.id), "flagged node must be checked by the verifier"
    assert_numbers_from_tools(rt, res.final_node)
