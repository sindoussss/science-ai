"""scripts/live_check.py, driven by a scripted model (the real script needs Ollama)."""
from __future__ import annotations

import importlib.util
import inspect
import sys
from pathlib import Path

import pytest

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


INTEGRAL_GOAL = {"action": "formalize", "statement": "Compute the integral of x**2*exp(-x) over [0, 1].",
                 "goal": {"tool": "sympy.integrate",
                          "args": {"expr": "x**2*exp(-x)", "var": "x", "lower": "0", "upper": "1"}}}


def test_counts_retries_calls_and_reuse(lc, make_rt):
    llm = lc.CountingLLM(ScriptedLLM(["not json at all", INTEGRAL_GOAL]))
    rt = make_rt(llm)
    first, repeat = lc.PROBLEMS[0], lc.PROBLEMS[-1]
    assert repeat.question == first.question and repeat.reuse_of == first.key

    row = lc.run_problem(rt, llm, first)
    assert row.passed, row
    assert (row.status, row.calls, row.retries, row.ladder) == ("answered", 2, 1, "none")

    row = lc.run_problem(rt, llm, repeat)  # same store: answered from it, no model call
    assert row.passed, row
    assert (row.status, row.calls, row.retries) == ("reused", 0, 0)


def test_a_wrong_value_fails_with_the_reason(lc, make_rt):
    wrong = {**INTEGRAL_GOAL, "goal": {"tool": "sympy.integrate",
                                       "args": {"expr": "x**2*exp(-x)", "var": "x", "lower": "0", "upper": "2"}}}
    llm = lc.CountingLLM(ScriptedLLM([wrong]))
    row = lc.run_problem(make_rt(llm), llm, lc.PROBLEMS[0])
    assert not row.passed and "expected" in row.why


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
    args = lc.parse_args(["--model", "llama3.1:8b", "--no-think"])
    assert (args.model, args.think) == ("llama3.1:8b", False)
