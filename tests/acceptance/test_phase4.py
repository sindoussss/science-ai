"""The six Phase 4 acceptance tests, against a scripted model and the real tool sandbox.

Phase 4 is the one domain where a passing check does not make a result believable. These tests
hold that line end to end: every chemistry node comes out of the controller as a hypothesis,
the answer built on one is not verified and says so, a corrupted descriptor is still caught by
its check, and a request no registered tool serves is declined once rather than retried.
"""
from __future__ import annotations

import copy

import pytest

from sciai.domains.chem import decline, standardize
from sciai.graph.model import Domain, NodeType, Status
from sciai.tools.faults import FaultyRunner
from tests.fakes.fake_llm import ScriptedLLM, last_handle

ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"
CAFFEINE = "Cn1cnc2c1c(=O)n(C)c(=O)n2C"
IBUPROFEN = "CC(C)Cc1ccc(cc1)C(C)C(=O)O"

QUESTION = f"What are the molecular properties of {ASPIRIN}?"
FORMAL = {"action": "formalize", "recipe": "molecule_question", "slots": {}, "operation": "descriptors",
          "statement": f"Compute the standard descriptors of the structure {ASPIRIN}."}


def nodes(rt, tool=None, type_=None):
    out = [n for n in rt.engine.nodes.values() if (tool is None or n.tool_name == tool)
           and (type_ is None or n.type == type_)]
    return sorted(out, key=lambda n: n.created_at)


def methods(rt, node_id):
    return {ev.method: ev.outcome for ev in rt.repo.evidence_for(node_id)}


def finish(tool, status="hypothesis", text="Result: "):
    def step(prompt):
        h = last_handle(prompt, tool, status=status)
        return {"action": "finish", "answer_template": f"{text}{{{{{h}}}}}", "answer_nodes": [h]}
    return step


# 1 ------------------------------------------------- descriptors stay hypotheses
def test_descriptors_are_checked_and_still_come_out_a_hypothesis(make_rt):
    goal = {"tool": "chem.descriptors", "args": {"structure": ASPIRIN}}
    rt = make_rt(ScriptedLLM([{**FORMAL, "goal": goal}, finish("chem.descriptors")]))
    rt.engine.new_session("aspirin")
    res = rt.controller.run(QUESTION)

    assert res.status == "answered"
    node = nodes(rt, "chem.descriptors")[0]
    assert node.domain == Domain.CHEM
    assert node.status == Status.HYPOTHESIS          # not verified, although its check passed
    assert methods(rt, node.id) == {"recomputed": "pass"}
    assert node.confidence is None                   # a number here would read as a probability
    assert node.result["values"]["heavy_atoms"] == 13

    # The answer rests on a hypothesis, so it is not verified and says which.
    assert not res.verified
    final = rt.engine.resolve(res.final_node)
    assert final.flags == ["hypothesis"]             # not "unverified inputs": nothing failed
    assert "rests on computed properties of candidates nobody has tested" in res.answer


# 2 -------------------------------------------------------- a corrupted value
def test_a_corrupted_molecular_weight_is_caught_by_the_recomputation(make_rt, runner):
    def corrupt(result):
        bad = copy.deepcopy(result)
        bad["values"]["mw"] += 1.0
        return bad

    faulty = FaultyRunner(runner, "chem.descriptors", corrupt=corrupt, calls={1})

    first_call = {"action": "call_tool", "tool": "chem.descriptors", "title": "descriptors",
                  "args": {"structure": ASPIRIN}}

    def retry(prompt):
        bad = last_handle(prompt, "chem.descriptors", status="failed")
        return {"action": "call_tool", "tool": "chem.descriptors", "replaces": bad,
                "title": "descriptors", "args": {"structure": ASPIRIN, "format": "smiles"}}

    # no goal: the failure and its retry happen inside the controller loop, where the ladder
    # hands the retry to the specialist role for the failed tool
    llm = ScriptedLLM([FORMAL, first_call, retry, finish("chem.descriptors")])
    rt = make_rt(llm, run=faulty)
    rt.engine.new_session("corrupt")
    res = rt.controller.run(QUESTION)

    assert faulty.corrupted == [1]
    assert "You are the chemistry worker" in llm.calls[-2]["system"]  # the retry role
    first, second = nodes(rt, "chem.descriptors")
    assert first.status == Status.FAILED and methods(rt, first.id) == {"recomputed": "fail"}
    assert second.status == Status.HYPOTHESIS and second.lineage_id == first.lineage_id
    assert res.status == "answered" and not res.verified
    assert second.result["values"]["mw"] == pytest.approx(first.result["values"]["mw"] - 1.0)


# 3 ------------------------------------------------------- declined by capability
@pytest.mark.parametrize("operation,asked,rule", [
    ("procedure", "Give me the reflux conditions and workup for this ester.", "procedure"),
    ("toxicity", "Which of these is the most toxic?", "generic"),
])
def test_a_request_no_tool_serves_is_declined_once(make_rt, operation, asked, rule):
    rt = make_rt(ScriptedLLM([{"action": "formalize", "recipe": "molecule_question", "slots": {}, "statement": asked,
                                 "operation": operation}]))
    rt.engine.new_session("declined")
    res = rt.controller.run(asked)

    assert res.status == "declined"
    assert res.llm_calls == 1                       # declined once, never retried at the model
    assert res.detail == decline.wording_for(asked)[1]
    node = rt.engine.resolve(res.final_node)
    assert node.type == NodeType.HINT and node.title == "Declined"
    assert node.tool_inputs["wording"] == rule
    assert node.tool_inputs["operation"] == operation
    assert node.tool_inputs["served"] == list(decline.served_operations())
    assert not nodes(rt, type_=NodeType.TOOL_RESULT)  # nothing was computed


@pytest.mark.parametrize("asked,operation,rule", [
    ("How do I synthesize aspirin from salicylic acid?", "synthesis_route", "route"),
    (f"How much of {ASPIRIN} should a patient take?", "dosing", "dosing"),
    ("Give me a retrosynthesis for paracetamol.", "synthesis_route", "route"),
    ("What dose of this compound is safe?", "dosing", "dosing"),
])
def test_a_request_nothing_serves_is_declined_before_any_model_call(make_rt, asked, operation, rule):
    """A question plainly asking for a route or a dose is declined in code, with no model call.

    The live runs showed why this matters: when formalization failed, these two questions came
    back as an error instead of a refusal. A decline must not depend on the model answering at
    all, so the model is given nothing to say -- the scripted script is empty, and a single call
    would raise.
    """
    rt = make_rt(ScriptedLLM([]))
    rt.engine.new_session("pre-declined")
    res = rt.controller.run(asked)

    assert res.status == "declined"
    assert res.llm_calls == 0
    assert res.detail == decline.wording_for(asked)[1]
    node = rt.engine.resolve(res.final_node)
    assert node.type == NodeType.HINT and node.title == "Declined"
    assert (node.tool_inputs["operation"], node.tool_inputs["wording"]) == (operation, rule)
    assert not nodes(rt, type_=NodeType.TOOL_RESULT)


@pytest.mark.parametrize("asked", [
    f"What is the logP of the product of this synthesis, {ASPIRIN}?",
    f"The synthesis paper reports {ASPIRIN}; what is its molecular weight?",
    "Which of these has the lowest predicted dose-limiting toxicity?",
    "Compute the definite integral of x^2*exp(-x) from x = 0 to x = 1.",
    "How much work is done lifting 10 kg through 2 m?",
    "How many metres are there in a mile?",
])
def test_the_code_decline_never_refuses_a_question_a_tool_can_serve(asked):
    """The mirror of the test above, and the reason it is capability-first: the pre-model check
    names an operation only when the question asks for one nothing performs. A flagged word in
    passing, and a maths or physics question that happens to say "how much", go through."""
    assert decline.decline_for_question(asked) is None


def test_a_served_operation_is_not_declined_for_the_words_around_it(make_rt):
    """Change 2: keywords choose wording, never whether to decline."""
    asked = f"The synthesis paper reports {ASPIRIN}; what is its molecular weight?"
    goal = {"tool": "chem.descriptors", "args": {"structure": ASPIRIN}}
    rt = make_rt(ScriptedLLM([{"action": "formalize", "recipe": "molecule_question", "slots": {}, "statement": asked,
                                 "operation": "descriptors",
                              "goal": goal}, finish("chem.descriptors")]))
    rt.engine.new_session("wording")
    res = rt.controller.run(asked)

    assert res.status == "answered"                 # "synthesis" in the question changes nothing
    assert nodes(rt, "chem.descriptors")[0].status == Status.HYPOTHESIS


def test_a_goal_naming_a_tool_that_does_not_exist_is_declined_not_retried(make_rt):
    asked = "Plan a route to this molecule."
    rt = make_rt(ScriptedLLM([{"action": "formalize", "recipe": "molecule_question", "slots": {}, "statement": asked,
                              "goal": {"tool": "chem.synthesize", "args": {"structure": ASPIRIN}}}]))
    rt.engine.new_session("no such tool")
    res = rt.controller.run(asked)

    assert res.status == "declined" and res.llm_calls == 1
    assert rt.engine.resolve(res.final_node).tool_inputs["operation"] == "synthesize"


# 4 ------------------------------------------------- the screen runs on a library
def test_every_library_member_is_screened_and_the_ranking_is_rechecked(make_rt):
    library = {"members": [{"name": "caffeine", "structure": CAFFEINE},
                           {"name": "ibuprofen", "structure": IBUPROFEN}]}
    asked = "Which of these two is nearest to aspirin?"
    goal = {"tool": "chem.similar", "args": {"structure": ASPIRIN, "library": library}}
    rt = make_rt(ScriptedLLM([{"action": "formalize", "recipe": "molecule_question", "slots": {}, "statement": asked,
                                 "operation": "similarity",
                              "goal": goal}, finish("chem.similar")]))
    rt.engine.new_session("similar")
    res = rt.controller.run(asked)

    assert res.status == "answered" and not res.verified
    node = nodes(rt, "chem.similar")[0]
    assert node.status == Status.HYPOTHESIS
    assert methods(rt, node.id) == {"alt_fingerprint": "pass"}
    assert node.result["refused_members"] == []
    assert {n["name"] for n in node.result["neighbours"]} == {"caffeine", "ibuprofen"}
    # every neighbour comes back standardized, so the screen saw the canonical form
    for n in node.result["neighbours"]:
        assert n["smiles"] == standardize.read_twice(n["smiles"]).canonical_smiles


# 5 --------------------------------------------------- a chem node is never verified
def test_no_route_in_the_system_can_verify_a_chemistry_node(make_rt):
    from sciai.graph.engine import GraphRuleError

    goal = {"tool": "chem.parse", "args": {"structure": ASPIRIN}}
    rt = make_rt(ScriptedLLM([{**FORMAL, "goal": goal}, finish("chem.parse")]))
    rt.engine.new_session("never verified")
    rt.controller.run(QUESTION)
    node = nodes(rt, "chem.parse")[0]
    assert node.status == Status.HYPOTHESIS and methods(rt, node.id) == {"reread": "pass"}

    with pytest.raises(GraphRuleError):
        rt.engine.set_status(node.id, Status.VERIFIED, "by hand")
    assert rt.engine.resolve(node.id).status == Status.HYPOTHESIS

    # and the store refuses it even if the engine were bypassed
    import sqlite3

    with pytest.raises(sqlite3.IntegrityError), rt.repo.db.tx() as conn:
        conn.execute("UPDATE nodes SET status='verified' WHERE id=?", (node.id,))


# 6 ---------------------------------------------- a passing check allows reuse
def test_a_checked_hypothesis_is_reused_and_stays_a_hypothesis(make_rt, tmp_path):
    """Change 5: reuse is what passing checks buy, and the reused node is still a hypothesis."""
    goal = {"tool": "chem.descriptors", "args": {"structure": ASPIRIN}}
    db = tmp_path / "kb.db"
    rt = make_rt(ScriptedLLM([{**FORMAL, "goal": goal}, {**FORMAL, "goal": goal}]), db_path=db)
    rt.engine.new_session("first")
    first = rt.controller.run(QUESTION)
    node = nodes(rt, "chem.descriptors")[0]
    assert node.status == Status.HYPOTHESIS and methods(rt, node.id) == {"recomputed": "pass"}

    rt.engine.new_session("again")
    again = rt.controller.run(QUESTION)
    assert again.status == "answered" and not again.verified
    reused = nodes(rt, "chem.descriptors")
    assert len(reused) == 1 and reused[0].id == node.id   # the same node, linked into the session
    assert reused[0].status == Status.HYPOTHESIS
    assert first.answer == again.answer
