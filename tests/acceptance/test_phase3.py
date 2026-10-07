"""The seven Phase 3 acceptance tests, against a scripted model and the real tool sandbox."""
from __future__ import annotations

import copy
import re

import numpy as np
import pytest

from sciai.graph.model import NodeType, Status
from sciai.tools.faults import FaultyRunner
from tests.conftest import numbers
from tests.fakes.fake_llm import ScriptedLLM, last_handle

QUESTION = "In trial.csv, is the score different between groups A and B?"


def write_trial(path, shift=6.0, seed=7):
    """40 patients: a score that is higher in group B, a dose and its response, site and sex."""
    rng = np.random.default_rng(seed)
    lines = ["id,group,score [points],dose [mg],response,site,sex"]
    for i in range(40):
        g = "A" if i < 20 else "B"
        score = rng.normal(50 + (shift if g == "B" else 0), 6)
        dose = 10 + (i % 10) * 5
        lines.append(f"{i + 1},{g},{score:.2f},{dose},{0.8 * dose + rng.normal(0, 4):.2f},"
                     f"{['north', 'south', 'east'][i % 3]},{'F' if rng.random() < 0.5 else 'M'}")
    path.write_text("\n".join(lines) + "\n")
    return path


@pytest.fixture
def cfg(cfg, tmp_path):
    cfg.data.data_dir = str(tmp_path / "datasets")
    return cfg


@pytest.fixture
def trial(tmp_path):
    return write_trial(tmp_path / "trial.csv")


def nodes(rt, tool=None, type_=None):
    out = [n for n in rt.engine.nodes.values() if (tool is None or n.tool_name == tool)
           and (type_ is None or n.type == type_)]
    return sorted(out, key=lambda n: n.created_at)


def methods(rt, node_id):
    return {ev.method: ev.outcome for ev in rt.repo.evidence_for(node_id)}


def assert_numbers_from_tools(rt, final_id):
    """Every number in the answer appears in a tool result behind it: an answer node, or a
    diagnostic of a test it reports."""
    final = rt.engine.resolve(final_id)
    allowed = set()
    for nid in final.tool_inputs["answer_nodes"]:
        node = rt.engine.resolve(nid)
        allowed |= numbers(node.display_result())
        for dep in rt.engine.dependencies(nid):
            d = rt.engine.resolve(dep)
            if d.type == NodeType.ASSUMPTION:
                allowed |= numbers(d.content)
    assert numbers(final.content) <= allowed, (final.content, numbers(final.content) - allowed)


TTEST_GOAL = {"tool": "stats.ttest", "args": {"dataset": "trial.csv", "column": "score", "by": "group"}}
FORMAL_GOAL = {"action": "formalize", "recipe": "dataset_question",
               "statement": "Compare mean score between groups A and B in trial.csv.",
               "problem_type": "data", "goal": TTEST_GOAL}
FORMAL = {"action": "formalize", "recipe": "dataset_question",
          "statement": "Describe and test the trial.csv data.", "problem_type": "data"}


def finish(tool, status=None, text="Result: "):
    def step(prompt):
        h = last_handle(prompt, tool, status=status)
        return {"action": "finish", "answer_template": f"{text}{{{{{h}}}}}", "answer_nodes": [h]}
    return step


# 1 ---------------------------------------------------------------- Welch t-test
def test_welch_answer_with_effect_ci_permutation_and_diagnostics(make_rt, trial):
    llm = ScriptedLLM([FORMAL_GOAL])
    rt = make_rt(llm)
    rt.import_file(trial)
    rt.engine.new_session("welch")
    res = rt.controller.run(QUESTION)

    assert res.status == "answered" and res.verified, res
    assert res.llm_calls == 1  # the goal is one tool call: no further model turns
    test = nodes(rt, "stats.ttest")[0]
    r = test.result
    assert r["test"] == "welch_t" and r["significant"] and r["alpha"] == 0.05
    assert r["effect"]["name"] == "mean difference (A - B)" and len(r["effect"]["ci"]) == 2
    assert methods(rt, test.id) == {"formula": "pass", "permutation": "pass"}
    assert "Welch t-test of score [points] by group (A vs B)" in res.answer
    assert "95% CI [" in res.answer and "alpha 0.05" in res.answer

    dataset = rt.engine.resolve(test.tool_inputs["dataset"])
    assert dataset.type == NodeType.ENTITY and dataset.status == Status.VERIFIED
    assert methods(rt, dataset.id) == {"reread": "pass"}
    assert dataset.id in rt.engine.dependencies(test.id)
    assert rt.engine.dependencies(dataset.id) == []  # shared across questions: no root

    assumed = sorted((rt.engine.resolve(d) for d in rt.engine.dependencies(test.id)
                      if rt.engine.resolve(d).type == NodeType.ASSUMPTION), key=lambda a: a.created_at)
    assert [a.tool_inputs["diagnostic"]["assumption"] for a in assumed] == ["normality", "normality",
                                                                            "independence"]
    assert all(a.locked for a in assumed)
    assert assumed[0].content.startswith("score [points] is roughly normal in A (Shapiro-Wilk p = ")
    assert res.assumptions == [a.content for a in assumed]
    assert_numbers_from_tools(rt, res.final_node)


def test_a_doubtful_diagnostic_is_flagged_not_failed(make_rt, tmp_path):
    path = tmp_path / "skew.csv"
    rng = np.random.default_rng(3)
    rows = [f"{g},{v:.4f}" for g, v in [("A", x) for x in rng.exponential(1, 12) ** 3] +
            [("B", x) for x in rng.exponential(1, 12) ** 3 + 2]]
    path.write_text("group,score\n" + "\n".join(rows) + "\n")
    goal = {"tool": "stats.ttest", "args": {"dataset": "skew.csv", "column": "score", "by": "group"}}
    rt = make_rt(ScriptedLLM([{**FORMAL_GOAL, "goal": goal}]))
    rt.import_file(path)
    rt.engine.new_session("skew")
    res = rt.controller.run("Is the score different between groups in skew.csv?")
    assert res.status == "answered"
    doubtful = [a for a in nodes(rt, type_=NodeType.ASSUMPTION) if any(f.startswith("doubtful") for f in a.flags)]
    assert doubtful and all(a.status == Status.PROPOSED for a in doubtful)  # flagged, not failed
    assert "Caution: score is roughly normal in" in res.answer


# 2 ---------------------------------------------------------------- injected wrong mean
def test_wrong_mean_is_caught_and_the_ladder_recovers(make_rt, runner, trial):
    def corrupt(result):
        bad = copy.deepcopy(result)
        bad["rows"][0][3] += 1.0
        return bad

    faulty = FaultyRunner(runner, "data.describe", corrupt=corrupt, calls={1})

    def describe(prompt):
        assert "trial.csv: 40 rows; columns: id (integer), group (text; levels A, B), score [points] (number)" \
            in prompt
        return {"action": "call_tool", "tool": "data.describe", "title": "summary",
                "args": {"dataset": "trial.csv", "columns": ["score"]}}

    def retry(prompt):
        bad = last_handle(prompt, "data.describe", status="failed")
        assert "data.describe and data.group take method" in llm.calls[-1]["system"]  # the data role
        return {"action": "call_tool", "tool": "data.describe", "replaces": bad, "title": "summary (numpy)",
                "args": {"dataset": "trial.csv", "columns": ["score"], "method": "numpy"}}

    llm = ScriptedLLM([FORMAL, describe, finish("data.describe"), retry, finish("data.describe", "proposed")])
    rt = make_rt(llm, run=faulty)
    rt.import_file(trial)
    rt.engine.new_session("ladder")
    res = rt.controller.run("Summarize the score in trial.csv.")

    assert faulty.corrupted == [1]
    assert res.status == "answered" and res.verified, res
    first, second = nodes(rt, "data.describe")
    assert first.status == Status.FAILED and methods(rt, first.id) == {"alt_algorithm": "fail"}
    assert second.status == Status.VERIFIED and second.lineage_id == first.lineage_id
    assert second.tool_inputs["method"] == "numpy"
    real_mean = second.result["rows"][0][3]
    assert first.result["rows"][0][3] == pytest.approx(real_mean + 1.0)
    assert_numbers_from_tools(rt, res.final_node)


# 3 ---------------------------------------------------------------- re-import
def test_reimport_is_reused_and_a_changed_cell_is_a_new_dataset(make_rt, trial):
    rt = make_rt(ScriptedLLM([FORMAL_GOAL, FORMAL_GOAL]))
    first_import = rt.import_file(trial)
    rt.engine.new_session("first")
    first = rt.controller.run(QUESTION)
    assert first.status == "answered" and first.verified and first.llm_calls == 1
    test1 = nodes(rt, "stats.ttest")[0]

    again = rt.import_file(trial)
    assert again.reused and again.record.id == first_import.record.id
    rt.engine.new_session("same file")
    reused = rt.controller.run(QUESTION)
    assert reused.status == "reused" and reused.llm_calls == 0 and reused.final_node == first.final_node

    text = trial.read_text().splitlines()
    cells = text[5].split(",")
    cells[2] = "99.99"  # one score changes
    text[5] = ",".join(cells)
    trial.write_text("\n".join(text) + "\n")
    changed = rt.import_file(trial)
    assert not changed.reused and changed.record.id != first_import.record.id
    rt.engine.new_session("changed file")
    fresh = rt.controller.run(QUESTION)
    assert fresh.status == "answered" and fresh.llm_calls == 1 and fresh.final_node != first.final_node
    test2 = nodes(rt, "stats.ttest")[0]
    assert test2.id != test1.id and test2.fingerprint != test1.fingerprint
    assert test2.result["dataset"]["key"] != test1.result["dataset"]["key"]
    assert test2.result["p"] != test1.result["p"]
    reused_ids = {n.id for n in rt.engine.nodes.values()} & {test1.id, test1.tool_inputs["dataset"]}
    assert not reused_ids  # nothing from the old file was brought into this session


# 4 ---------------------------------------------------------------- family rule
FIVE = [
    ("stats.ttest", {"column": "score", "by": "group"}),
    ("stats.mannwhitney", {"column": "score", "by": "group"}),
    ("stats.correlation", {"x": "dose", "y": "response"}),
    ("stats.chi2", {"row": "site", "column": "sex"}),
    ("stats.anova", {"column": "score", "by": "site"}),
]


def test_five_tests_trigger_the_family_rule_and_holm(make_rt, trial):
    steps = [FORMAL] + [{"action": "call_tool", "tool": t, "args": {"dataset": "trial.csv", **a}, "title": t}
                        for t, a in FIVE]

    def finish_all(prompt):
        hs = [last_handle(prompt, t) for t, _ in FIVE]
        return {"action": "finish", "answer_nodes": hs,
                "answer_template": "Tests: " + "; ".join(f"{{{{{h}}}}}" for h in hs)}

    rt = make_rt(ScriptedLLM([*steps, finish_all]))
    rt.import_file(trial)
    rt.engine.new_session("family")
    res = rt.controller.run("Run the standard tests on trial.csv.")
    assert res.status == "answered" and res.verified, res

    tests = [nodes(rt, t)[0] for t, _ in FIVE]
    assert all(t.status == Status.VERIFIED for t in tests)
    assert all("family" in {f["rule"] for f in t.risk["fired"]} for t in tests)
    adj = nodes(rt, "stats.adjust")[0]
    assert adj.status == Status.VERIFIED and methods(rt, adj.id) == {"alt_algorithm": "pass"}
    p = [t.result["p"] for t in tests]
    assert adj.result["p"] == p
    order = np.argsort(p)
    holm = np.maximum.accumulate([min(1.0, (5 - k) * p[i]) for k, i in enumerate(order)])
    expected = np.empty(5)
    expected[order] = holm
    assert adj.result["p_adjusted"] == pytest.approx(expected.tolist())
    assert set(rt.engine.dependencies(adj.id)) == {t.id for t in tests}
    assert "Holm-adjusted p for 5 tests on trial.csv" in res.answer
    assert "significant after adjustment:" in res.answer
    final = rt.engine.resolve(res.final_node)
    assert adj.id in final.tool_inputs["answer_nodes"]
    assert_numbers_from_tools(rt, res.final_node)


# 5 ---------------------------------------------------------------- regression and fit plot
def test_regression_with_fit_plot_and_deleting_the_dataset_invalidates_both(make_rt, trial):
    from sciai.graph import plotspec

    regress = {"action": "call_tool", "tool": "stats.regression", "title": "response on dose",
               "args": {"dataset": "trial.csv", "y": "response", "x": ["dose"]}}

    def plot(prompt):
        h = last_handle(prompt, "stats.regression")
        return {"action": "call_tool", "tool": "plot.fit", "title": "fit", "args": {"dataset": "trial.csv", "node": h}}

    rt = make_rt(ScriptedLLM([FORMAL, regress, plot, finish("stats.regression", text="Fit: ")]))
    rt.import_file(trial)
    rt.engine.new_session("regression")
    res = rt.controller.run("How does response depend on dose in trial.csv?")
    assert res.status == "answered" and res.verified, res
    (reg,) = nodes(rt, "stats.regression")
    (fig,) = nodes(rt, "plot.fit")
    assert methods(rt, reg.id) == {"formula": "pass", "statsmodels": "pass"}
    slope = reg.result["coefficients"][1]
    assert slope["ci"][0] < 0.8 < slope["ci"][1] and "slope dose [mg] = " in res.answer
    assert fig.type == NodeType.PLOT and {reg.id, reg.tool_inputs["dataset"]} <= set(rt.engine.dependencies(fig.id))
    spec = fig.result["value"]
    assert spec["version"] == 2 and [s["type"] for s in spec["series"]] == ["scatter", "line"]
    line = spec["series"][1]
    assert line["y"][0] == pytest.approx(reg.result["coefficients"][0]["estimate"] + slope["estimate"] * line["x"][0])
    assert plotspec.render_png(spec, 600, 400)[:8] == b"\x89PNG\r\n\x1a\n"

    removed = rt.remove_dataset("trial.csv")
    assert removed == [reg.tool_inputs["dataset"]]
    for nid in (reg.id, fig.id, res.final_node):
        assert rt.engine.resolve(nid).status == Status.INVALIDATED
    assert rt.repo.dataset_by_name("trial.csv") is None
    assert rt.controller._datasets_text("trial.csv") == ""  # the model is no longer offered the file


# 6 ---------------------------------------------------------------- unknown column
def test_unknown_column_is_an_error_node_and_a_retry(make_rt, trial):
    def wrong(prompt):
        return {"action": "call_tool", "tool": "stats.ttest", "title": "weight by group",
                "args": {"dataset": "trial.csv", "column": "weight", "by": "group"}}

    def right(prompt):
        assert "column 'weight' is not in trial.csv; its columns are: id, group, score [points]" in prompt
        return {"action": "call_tool", "tool": "stats.ttest", "title": "score by group",
                "args": {"dataset": "trial.csv", "column": "score", "by": "group"}}

    rt = make_rt(ScriptedLLM([FORMAL, wrong, right, finish("stats.ttest")]))
    rt.import_file(trial)
    rt.engine.new_session("columns")
    res = rt.controller.run("Is weight different between groups in trial.csv?")
    assert res.status == "answered" and res.verified
    errors = nodes(rt, type_=NodeType.ERROR)
    assert len(errors) == 1 and "column 'weight' is not in trial.csv" in errors[0].content
    tests = nodes(rt, "stats.ttest")
    assert [t.tool_inputs["column"] for t in tests] == ["score"]  # no node was made for a made-up column


# 7 ---------------------------------------------------------------- no number without a tool
def test_numbers_come_only_from_tools(make_rt, trial):
    def digits_in_template(prompt):
        h = last_handle(prompt, "stats.ttest")
        return {"action": "finish", "answer_template": f"Yes, p < 0.05: {{{{{h}}}}}", "answer_nodes": [h]}

    def alpha_from_nowhere(prompt):
        return {"action": "call_tool", "tool": "stats.ttest",
                "args": {"dataset": "trial.csv", "column": "score", "by": "group", "alpha": 0.01}}

    def significant_without_test(prompt):
        h = last_handle(prompt, "data.describe")
        return {"action": "finish", "answer_template": f"The difference is significant: {{{{{h}}}}}",
                "answer_nodes": [h]}

    describe = {"action": "call_tool", "tool": "data.describe", "args": {"dataset": "trial.csv", "columns": ["score"]}}
    ttest = {"action": "call_tool", "tool": "stats.ttest", "args": {"dataset": "trial.csv", "column": "score",
                                                                     "by": "group"}}
    llm = ScriptedLLM([FORMAL, alpha_from_nowhere, ttest, describe, significant_without_test, digits_in_template,
                       finish("stats.ttest", text="Groups compared: ")])
    rt = make_rt(llm)
    rt.import_file(trial)
    rt.engine.new_session("numbers")
    res = rt.controller.run("Is the score different between the groups in trial.csv?")
    assert res.status == "answered" and res.verified
    prompts = "\n".join(c["user"] for c in llm.calls)
    assert "alpha 0.01 is not in the question; leave alpha out to use 0.05" in prompts
    assert 'the word "significant" needs the test result node in answer_nodes' in prompts
    assert "answer_template may not contain the number(s) 0.05" in prompts
    (test,) = nodes(rt, "stats.ttest")
    assert test.tool_inputs["alpha"] == 0.05 and not test.flags  # the configured alpha is a source
    assert_numbers_from_tools(rt, res.final_node)
    assert re.search(r"\d", res.answer)  # the numbers are there, all from the test node
