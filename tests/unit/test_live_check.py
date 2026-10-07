"""scripts/live_check.py, driven by a scripted model (the real script needs Ollama)."""
from __future__ import annotations

import importlib.util
import inspect
import json
import sys
from pathlib import Path

import pytest

from sciai.graph.model import NodeType
from sciai.llm.actions import action_schema
from tests.fakes.fake_llm import ScriptedLLM

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "live_check.py"


@pytest.fixture(scope="module")
def lc():
    spec = importlib.util.spec_from_file_location("live_check", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["live_check"] = mod  # dataclasses look their module up while the class is built
    spec.loader.exec_module(mod)
    yield mod
    sys.modules.pop("live_check", None)


INTEGRAL_ROUTE = {"action": "formalize", "recipe": "definite_integral",
                  "statement": "Compute the integral of x**2*exp(-x) over [0, 1].",
                  "slots": {"integrand": "x**2*exp(-x)", "var": "x", "lower": "0", "upper": "1"}}


def test_counts_retries_calls_and_reuse(lc, make_rt):
    llm = lc.CountingLLM(ScriptedLLM(["not json at all", INTEGRAL_ROUTE]))
    rt = make_rt(llm)
    first, repeat = lc.PROBLEMS[0], lc.PROBLEMS[-1]
    assert repeat.question == first.question and repeat.reuse_of == first.key

    row = lc.run_problem(rt, llm, first)
    # Two calls, one of them the re-ask after invalid JSON: the planning budget of 1 counts
    # the routing call only, which is what the recipe library was meant to bring down.
    assert row.passed, row
    assert (row.status, row.calls, row.retries, row.ladder) == ("answered", 2, 1, "none")

    row = lc.run_problem(rt, llm, repeat)  # same store: answered from it, no model call
    assert row.passed, row
    assert (row.status, row.calls, row.retries) == ("reused", 0, 0)


def test_a_wrong_value_fails_with_the_reason(lc, make_rt):
    wrong = {**INTEGRAL_ROUTE, "slots": {**INTEGRAL_ROUTE["slots"], "upper": "2"}}
    llm = lc.CountingLLM(ScriptedLLM([wrong]))
    row = lc.run_problem(make_rt(llm), llm, lc.PROBLEMS[0])
    assert not row.passed and "expected" in row.why


def test_too_many_planning_calls_fails_even_with_the_right_value(lc, make_rt, runner):
    """Item 4 of the consolidated fix, as the live check measures it: an easy problem costs one
    model call. A right answer that took a second round of planning is still a failure."""
    from sciai.tools.faults import FaultyRunner

    def retry(prompt):
        from tests.fakes.fake_llm import last_handle
        bad = last_handle(prompt, "sympy.integrate")
        return {"action": "call_tool", "tool": "sympy.integrate", "replaces": bad,
                "args": {"expr": "x**2*exp(-x)", "var": "x", "lower": "0", "upper": "1",
                         "method": "meijerg"}}

    def finish(prompt):
        from tests.fakes.fake_llm import last_handle
        h = last_handle(prompt, "sympy.integrate", status="proposed")
        return {"action": "finish", "answer_template": f"The integral is {{{{{h}}}}}.",
                "answer_nodes": [h]}

    faulty = FaultyRunner(runner, "sympy.integrate", "1/3", calls={1})
    llm = lc.CountingLLM(ScriptedLLM([INTEGRAL_ROUTE, retry, finish]))
    row = lc.run_problem(make_rt(llm, run=faulty), llm, lc.PROBLEMS[0])
    assert not row.passed, row
    assert row.why == "right value, but it took 3 planning call(s) and should take 1"


def test_trace_records_the_request_the_schema_and_the_raw_reply(lc, make_rt, tmp_path):
    """Item 2 of the consolidated fix. The trace is the only place the prompt, the schema sent
    as the output format and the reply sit side by side, which is what a formalization failure
    needs: the bug it exists to expose was the schema and the validator disagreeing."""
    llm = lc.CountingLLM(ScriptedLLM([INTEGRAL_ROUTE]))
    path = tmp_path / "trace-math-1.jsonl"
    llm.trace_to(path, "integral")
    rt = make_rt(llm)
    row = lc.run_problem(rt, llm, lc.PROBLEMS[0])
    assert row.passed, row

    lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(lines) == 1                      # one routing call, so one record
    rec = lines[0]
    assert rec["problem"] == "integral" and rec["call"] == 1 and rec["retry"] is False
    assert "QUESTION:" in rec["request"]["user"]
    assert "recipe" in rec["request"]["system"] or "RECIPES" in rec["request"]["system"]
    sent = rec["request"]["schema"]
    assert sent == action_schema(("formalize",))          # the schema the model generated under
    assert sent["required"] == ["action", "recipe", "slots"]
    assert json.loads(rec["reply"])["recipe"] == "definite_integral"

    llm.trace_to(None)                          # tracing off again: nothing more is written
    llm.inner.steps.append(INTEGRAL_ROUTE)
    llm.chat("sys", "QUESTION:\nanything", sent)
    assert len([ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]) == 1


def test_trace_is_off_unless_asked_for(lc):
    assert lc.parse_args(["--model", "m"]).trace is False
    assert lc.parse_args(["--model", "m", "--trace"]).trace is True


def test_a_fresh_store_is_the_default(lc):
    """Yeri's item 3: a suite must not inherit, or leave behind, a stored answer."""
    assert lc.parse_args(["--model", "m"]).fresh_store is True
    assert lc.parse_args(["--model", "m", "--no-fresh-store"]).fresh_store is False


def test_retry_markers_match_the_controller(lc):
    from sciai.controller import loop

    src = inspect.getsource(loop)
    assert all(m in src for m in lc.RETRY_MARKERS)


def test_report_table_and_file_name(lc):
    rows = [lc.Row(lc.PROBLEMS[0], status="answered", passed=True, retries=1, ladder="none", seconds=3.25,
                   calls=2, answer="Answer: 2 - 5*exp(-1)"),
            lc.Row(lc.PROBLEMS[4], status="escalated", retries=0, ladder="escalated", seconds=40, calls=9,
                   why="expected 8, got [4.0]")]
    text = lc.report(rows, {"model": "qwen2.5:7b", "think": "false"})
    assert "- passed: 1/2" in text
    assert "| 1 | integral | 2 - 5/e ≈ 0.160603 | answered | PASS | 1 | none | 3.2 | 2 |" in text
    assert "| 2 | hard | 8 | escalated | FAIL | 0 | escalated | 40.0 | 9 | expected 8, got [4.0] |" in text
    assert lc.safe_name("qwen2.5:7b-instruct-q4_K_M") == "qwen2.5-7b-instruct-q4_K_M"
    assert lc.safe_name("hf.co/org/Model:Q4") == "hf.co-org-Model-Q4"


def test_six_problems_of_the_requested_kinds(lc):
    assert [p.key for p in lc.PROBLEMS] == ["integral", "equation", "units", "ode", "hard", "repeat"]
    # Every one of them is a recipe, so each should cost the single routing call.
    assert [p.max_calls for p in lc.PROBLEMS] == [1, 1, 1, 1, 1, 0]
    assert [p.max_calls for p in lc.PHYSICS_PROBLEMS] == [1, 1, 1, 1, 1, 0]
    # The five chemistry questions are recipes too; the two declines and the repeat cost nothing.
    assert [p.max_calls for p in lc.CHEM_PROBLEMS] == [1, 1, 1, 1, 1, 0, 0, 0]
    args = lc.parse_args(["--model", "llama3.1:8b", "--no-think"])
    assert (args.model, args.think) == ("llama3.1:8b", False)


def test_physics_suite(lc, make_rt):
    assert [p.key for p in lc.PHYSICS_PROBLEMS] == ["projectile", "temperature", "photon", "rc", "circuit", "repeat"]
    assert lc.SUITES["physics"] is lc.PHYSICS_PROBLEMS and lc.parse_args(["--model", "m"]).suite == "math"
    assert lc.parse_args(["--model", "m", "--suite", "physics"]).suite == "physics"
    # A quantity answer counts in its shown unit or in SI; tiny values are not "close" to zero.
    assert lc._numbers({"kind": "quantity", "value": 25.0, "unit": "degC", "si_value": 298.15}) == [25.0, 298.15]
    assert not lc._close(3.97e-19, 0.0) and lc._close(-0.06, 0.06, signed=False) and not lc._close(-0.06, 0.06)

    temp = {"action": "formalize", "recipe": "unit_convert",
            "statement": "Convert 25 degC to kelvin.",
            "slots": {"value": "25", "from_unit": "degC", "to_unit": "K",
                      "kind": "absolute_temperature"}}
    llm = lc.CountingLLM(ScriptedLLM([temp]))
    row = lc.run_problem(make_rt(llm), llm, lc.PHYSICS_PROBLEMS[1])
    assert row.passed, row
    assert (row.calls, row.retries) == (1, 0)

    # The assumptions a physics recipe carries are its problem type's, and the physics suite
    # accepts them automatically because nobody is there to tick the checklist.
    projectile = {"action": "formalize", "recipe": "projectile_range",
                  "statement": "Range at 20 m/s, 30 degrees.",
                  "slots": {"v0": "20", "v0_unit": "m/s", "angle": "30", "angle_unit": "deg",
                            "g": "9.80665", "g_unit": "m/s^2", "to_unit": "m"}}
    llm = lc.CountingLLM(ScriptedLLM([projectile]))
    row = lc.run_problem(make_rt(llm), llm, lc.PHYSICS_PROBLEMS[0])
    assert row.passed, row
    assert row.assumptions == ["no air resistance", "constant g = 9.80665 m/s^2",
                               "launch and landing at the same height"]
    text = lc.report([row], {"model": "m", "suite": "physics"})
    assert "## Assumptions accepted automatically" in text and "no air resistance" in text


def test_chem_suite_declines_cost_no_model_call(lc, make_rt):
    """The chem suite scored 0/8 live, both declines included, because formalization failed
    before the capability check ran. They are decided in code now, so the scripted model is
    given nothing at all: a single call would raise."""
    assert [p.key for p in lc.CHEM_PROBLEMS] == ["identity", "descriptors", "logp", "druglike",
                                                 "similarity", "declined", "declined_dose",
                                                 "repeat"]
    assert lc.SUITES["chem"] is lc.CHEM_PROBLEMS
    assert lc.parse_args(["--model", "m", "--suite", "chem"]).suite == "chem"

    for problem in [p for p in lc.CHEM_PROBLEMS if p.declined]:
        llm = lc.CountingLLM(ScriptedLLM([]))
        row = lc.run_problem(make_rt(llm), llm, problem)
        assert row.passed, row
        assert (row.status, row.calls, row.retries) == ("declined", 0, 0)


def test_the_whole_chem_suite_answers_through_recipes_in_one_call_each(lc, make_rt):
    """Yeri's item 4: the five chemistry questions are recipes now, so each costs the single
    routing call, and each answer is still a hypothesis. This is the suite that came back 2/8
    with physics answers in the chemistry rows."""
    problems = {p.key: p for p in lc.CHEM_PROBLEMS}
    routes = {
        "identity": ("chem_identity", {"structure": lc.ASPIRIN}),
        "descriptors": ("chem_descriptors", {"structure": lc.CAFFEINE}),
        "logp": ("chem_logp", {"structure": lc.IBUPROFEN}),
        "druglike": ("chem_druglike", {"structure": lc.PARACETAMOL}),
        "similarity": ("chem_similarity", {"structure": lc.ASPIRIN, "library": lc.CHEM_LIBRARY}),
    }
    for key, (recipe, slots) in routes.items():
        llm = lc.CountingLLM(ScriptedLLM([{"action": "formalize", "recipe": recipe, "slots": slots}]))
        row = lc.run_problem(make_rt(llm), llm, problems[key])
        assert row.passed, f"{key}: {row.why}"
        assert (row.status, row.calls, row.retries) == ("answered", 1, 0), key


def test_a_physics_recipe_cannot_answer_a_molecule_question(lc, make_rt):
    """Yeri's items 1 and 2, as the live check scores them. The model is given exactly the reply
    the real run produced -- photon_energy with the worked example's 500 nm -- and the suite must
    record an out-of-scope answer, not caffeine's molecular weight reported as 3.97e-19 J."""
    wrong = {"action": "formalize", "recipe": "photon_energy",
             "slots": {"wavelength": "500", "wavelength_unit": "nm", "to_unit": "J"}}
    llm = lc.CountingLLM(ScriptedLLM([wrong, wrong]))
    rt = make_rt(llm)
    row = lc.run_problem(rt, llm, [p for p in lc.CHEM_PROBLEMS if p.key == "descriptors"][0])
    assert not row.passed and row.status == "out_of_scope", row
    computed = [n for n, _ in rt.repo.session_view(rt.engine.session_id or "")
                if n.type is NodeType.TOOL_RESULT]
    assert not computed, "an unsourced plan reached the tools"


def test_chem_identity_answers_in_one_call_and_stays_a_hypothesis(lc, make_rt):
    from tests.fakes.fake_llm import last_handle

    aspirin = "CC(=O)Oc1ccccc1C(=O)O"
    route = {"action": "formalize", "recipe": "molecule_question", "slots": {},
             "operation": "identity",
             "goal": {"tool": "chem.parse", "args": {"structure": aspirin}}}

    def finish(prompt):
        h = last_handle(prompt, "chem.parse")
        return {"action": "finish", "answer_template": f"It is {{{{{h}}}}}.", "answer_nodes": [h]}

    llm = lc.CountingLLM(ScriptedLLM([route, finish]))
    row = lc.run_problem(make_rt(llm), llm, lc.CHEM_PROBLEMS[0])
    assert row.passed, row
    assert row.status == "answered"


def test_a_failing_row_says_why_and_quotes_the_model(lc, make_rt):
    """Two chem runs in a row came back 6/8 with a row that said only "I can only answer
    questions that fit one of the recipes I have", which is the sentence every out-of-scope
    question gets. The cause -- the route, or the slot and the value the model put in it -- is
    in the graph, and the raw reply is what it has to be read against, so the table and the
    saved report now carry both and a failure no longer needs the trace file."""
    problem = [p for p in lc.CHEM_PROBLEMS if p.key == "descriptors"][0]
    wrong_route = {"action": "formalize", "recipe": "photon_energy",
                   "slots": {"wavelength": "500", "wavelength_unit": "nm", "to_unit": "J"}}
    llm = lc.CountingLLM(ScriptedLLM([wrong_route, wrong_route]))
    row = lc.run_problem(make_rt(llm), llm, problem)

    assert not row.passed and row.status == "out_of_scope"
    assert "photon_energy" in row.why, row.why
    assert row.replies and all("photon_energy" in r for r in row.replies)
    # The console says it too, because a pasted console is usually all anyone reads.
    assert any("photon_energy" in line for line in lc.explain(row))
    text = lc.report([row], {"model": "m", "suite": "chem"})
    assert "## What the model replied, for each failing problem" in text
    assert "photon_energy" in text

    # A slot refusal names the slot and the value the model put in it.
    nameless = lc.Problem("nameless", "What is the logP of caffeine?", (), "logP", max_calls=1)
    bad = {"action": "formalize", "recipe": "chem_logp",
           "slots": {"structure": "CN1C=NC2=C1C(=O)N(C)C(=O)N2C"}}
    llm = lc.CountingLLM(ScriptedLLM([bad, bad]))
    row = lc.run_problem(make_rt(llm), llm, nameless)
    assert "structure = CN1C=NC2=C1C(=O)N(C)C(=O)N2C" in row.why, row.why


def test_data_suite(lc, make_rt, cfg, tmp_path):
    assert [p.key for p in lc.DATA_PROBLEMS] == ["welch", "anova", "regression", "excel", "means", "repeat"]
    assert lc.parse_args(["--model", "m", "--suite", "data"]).suite == "data"
    assert all((lc.DATA_DIR / name).is_file() for name in lc.DATA_FILES)
    cfg.data.data_dir = str(tmp_path / "datasets")
    welch, anova, regression, excel, means, repeat = lc.DATA_PROBLEMS

    def ask(problem, *script):
        llm = lc.CountingLLM(ScriptedLLM(list(script)))
        rt.controller.llm = llm
        return lc.run_problem(rt, llm, problem)

    rt = make_rt(ScriptedLLM([]))
    assert lc.import_files(rt, lc.DATA_FILES) == \
        "trial.csv (40 x 7), plantgrowth.tsv (30 x 2), sleep.xlsx (20 x 3)"

    def goal(tool, **args):
        return {"action": "formalize", "recipe": "dataset_question", "slots": {}, "statement": "data question",
                "problem_type": "data", "goal": {"tool": tool, "args": args}}

    row = ask(welch, goal("stats.ttest", dataset="trial.csv", column="score", by="group"))
    assert row.passed, row
    assert ask(anova, goal("stats.anova", dataset="plantgrowth.tsv", column="weight", by="group")).passed
    assert ask(regression, goal("stats.regression", dataset="trial.csv", y="response", x=["dose"])).passed
    row = ask(excel, goal("stats.ttest", dataset="sleep.xlsx", column="extra", by="drug"))
    assert row.passed, row
    row = ask(means, goal("data.group", dataset="trial.csv", by="group", column="score", agg="mean"))
    assert row.passed, row
    row = ask(repeat)  # answered from the store: no model call
    assert row.passed and (row.status, row.calls) == ("reused", 0), row

    rt = make_rt(ScriptedLLM([]), db_path=tmp_path / "fresh.db")  # a new store, so nothing is reused
    lc.import_files(rt, ("plantgrowth.tsv",))
    wrong = ask(anova, goal("stats.kruskal", dataset="plantgrowth.tsv", column="weight", by="group"))
    assert not wrong.passed and wrong.why.startswith("expected F = 4.846")
