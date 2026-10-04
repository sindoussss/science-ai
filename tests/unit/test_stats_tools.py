"""Statistical tests: reference values, and each independent check catching a wrong result."""
from __future__ import annotations

import copy
import math

import pytest

from sciai.config import DataConfig
from sciai.domains.data.datasets import import_file
from sciai.domains.data.report import adjusted_text, stats_text
from sciai.store.db import Database
from sciai.store.repository import Repository
from sciai.tools import stats_tools as S
from sciai.tools.sandbox import InlineRunner

R = InlineRunner()

# R's datasets::sleep (extra hours of sleep, two drugs, ten patients) and datasets::PlantGrowth.
SLEEP_1 = [0.7, -1.6, -0.2, -1.2, -0.1, 3.4, 3.7, 0.8, 0.0, 2.0]
SLEEP_2 = [1.9, 0.8, 1.1, 0.1, -0.1, 4.4, 5.5, 1.6, 4.6, 3.4]
PLANTS = {"ctrl": [4.17, 5.58, 5.18, 6.11, 4.50, 4.61, 5.17, 4.53, 5.33, 5.14],
          "trt1": [4.81, 4.17, 4.41, 3.59, 5.87, 3.83, 6.03, 4.89, 4.32, 4.69],
          "trt2": [6.31, 5.12, 5.54, 5.50, 5.37, 5.29, 4.92, 6.15, 5.80, 5.26]}
CHECKS = ("stats.check_formula", "stats.check_permutation", "stats.check_statsmodels")


def run(tool, **args):
    out = R.run(tool, args)
    assert out.ok, out.error
    return out.value["result"]


def fails(tool, **args):
    out = R.run(tool, args)
    assert not out.ok
    return out.error


def check(name, ds, args, result):
    return run(name, dataset=ds, test_args=args, result=result)


@pytest.fixture(scope="module")
def data(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("stats")
    db = Database(tmp / "k.db")
    repo, cfg = Repository(db), DataConfig(data_dir=str(tmp / "datasets"))
    lines = ["patient,drug,extra [h],drug1 [h],drug2 [h]"]
    lines += [f"{i + 1},1,{a},{a},{b}" for i, (a, b) in enumerate(zip(SLEEP_1, SLEEP_2))]
    lines += [f"{i + 1},2,{b},," for i, b in enumerate(SLEEP_2)]
    (tmp / "sleep.csv").write_text("\n".join(lines) + "\n")
    lines = ["weight,group"] + [f"{w},{g}" for g, ws in PLANTS.items() for w in ws]
    (tmp / "plants.csv").write_text("\n".join(lines) + "\n")
    lines = ["x,y,smoker,outcome"]
    for i in range(60):
        x = i / 6
        y = 2.0 * x + 1.0 + math.sin(7 * i) * 0.8
        lines.append(f"{x},{y:.6f},{'yes' if i % 3 == 0 else 'no'},{'ill' if (i % 3 == 0) == (i % 4 != 0) else 'well'}")
    (tmp / "obs.csv").write_text("\n".join(lines) + "\n")
    out = {n: import_file(tmp / f"{n}.csv", repo, R, cfg).descriptor for n in ("sleep", "plants", "obs")}
    yield out
    db.close()


# ------------------------------------------------------------------ reference values (R)
def test_welch_matches_r_sleep(data):
    a = {"column": "extra", "by": "drug"}
    r = run("stats.ttest", dataset=data["sleep"], **a)
    assert r["test"] == "welch_t" and r["spec"]["levels"] == ["1", "2"]
    assert r["statistic"]["value"] == pytest.approx(-1.8608, abs=1e-4)
    assert r["df"] == pytest.approx(17.776, abs=1e-3)
    assert r["p"] == pytest.approx(0.07939, abs=1e-5)
    assert r["effect"]["ci"] == pytest.approx([-3.3654832, 0.2054832], abs=1e-6)
    assert r["significant"] is False and r["alpha"] == 0.05
    kinds = [d["assumption"] for d in r["diagnostics"]]
    assert kinds == ["normality", "normality", "independence"]  # Welch does not assume equal variances
    for name in CHECKS:
        assert check(name, data["sleep"], a, r)["outcome"] == "pass", name
    text = stats_text(r)
    assert text.startswith("Welch t-test of extra [h] by drug (1 vs 2): t(17.78) = -1.861, p = 0.0794, alpha 0.05")
    assert "mean difference (1 - 2) = -1.58, 95% CI [-3.365, 0.2055]" in text


def test_paired_matches_r_sleep(data):
    a = {"column": "drug1", "paired_with": "drug2"}
    r = run("stats.ttest", dataset=data["sleep"], **a)
    assert r["n"] == 10 and r["spec"]["dropped"] == 10  # the drug-2 rows have no pair columns
    assert r["statistic"]["value"] == pytest.approx(-4.0621, abs=1e-4)
    assert r["p"] == pytest.approx(0.002833, abs=1e-6)
    assert r["effect"]["ci"] == pytest.approx([-2.4598858, -0.7001142], abs=1e-6)
    for name in CHECKS:
        assert check(name, data["sleep"], a, r)["outcome"] == "pass", name


def test_mann_whitney_matches_r_sleep(data):
    a = {"column": "extra", "by": "drug"}
    r = run("stats.mannwhitney", dataset=data["sleep"], **a)
    assert r["spec"]["method"] == "asymptotic"  # ties: normal approximation with continuity correction
    assert r["statistic"]["value"] == 25.5
    assert r["p"] == pytest.approx(0.06933, abs=1e-5)
    assert r["effect"]["value"] == pytest.approx(2 * 25.5 / 100 - 1)
    lo, hi = r["effect"]["ci"]
    assert lo < r["effect"]["value"] < hi
    assert check("stats.check_formula", data["sleep"], a, r)["outcome"] == "pass"
    assert check("stats.check_permutation", data["sleep"], a, r)["outcome"] == "pass"


def test_anova_matches_r_plant_growth(data):
    a = {"column": "weight", "by": "group"}
    r = run("stats.anova", dataset=data["plants"], **a)
    assert r["statistic"]["value"] == pytest.approx(4.846, abs=1e-3)
    assert r["df"] == [2, 27]
    assert r["p"] == pytest.approx(0.01591, abs=1e-5)
    assert r["effect"]["value"] == pytest.approx(0.2641, abs=1e-4)
    assert [d["assumption"] for d in r["diagnostics"]] == ["normality"] * 3 + ["equal_variance", "independence"]
    for name in CHECKS:
        assert check(name, data["plants"], a, r)["outcome"] == "pass", name


def test_kruskal_matches_r_plant_growth(data):
    a = {"column": "weight", "by": "group"}
    r = run("stats.kruskal", dataset=data["plants"], **a)
    assert r["statistic"]["value"] == pytest.approx(7.9882, abs=1e-4)
    assert r["df"] == 2 and r["p"] == pytest.approx(0.01842, abs=1e-5)
    assert check("stats.check_formula", data["plants"], a, r)["outcome"] == "pass"
    assert check("stats.check_permutation", data["plants"], a, r)["outcome"] == "pass"
    bad = _corrupt(r, ("statistic", "value"), 6.5)
    assert check("stats.check_formula", data["plants"], a, bad)["outcome"] == "fail"


def test_mann_whitney_exact_distribution():
    assert S._mw_counts(2, 2) == [1, 1, 2, 1, 1]
    assert sum(S._mw_counts(4, 6)) == math.comb(10, 4)


def test_holm():
    r = run("stats.adjust", p=[0.01, 0.04, 0.03, 0.005], labels=["a", "b", "c", "d"], alpha=0.05)
    assert r["p_adjusted"] == pytest.approx([0.03, 0.06, 0.06, 0.02])
    assert r["significant"] == [True, False, False, True]
    assert run("stats.check_adjust", result=r)["outcome"] == "pass"
    wrong = copy.deepcopy(r)
    wrong["p_adjusted"][1] = 0.04  # Bonferroni-free, step-down forgotten
    assert run("stats.check_adjust", result=wrong)["outcome"] == "fail"
    assert adjusted_text(r).startswith("Holm-adjusted p for 4 tests on the same data: a 0.01 to 0.03")


# ------------------------------------------------------------------ the other tests pass their checks
@pytest.mark.parametrize("tool,args", [
    ("stats.ttest", {"column": "extra", "by": "drug", "equal_var": True, "alternative": "less"}),
    ("stats.ttest", {"column": "extra", "mu": 0.5}),
    ("stats.mannwhitney", {"column": "weight", "by": "group", "levels": ["trt2", "CTRL"]}),
    ("stats.correlation", {"x": "x", "y": "y"}),
    ("stats.correlation", {"x": "x", "y": "y", "method": "spearman", "alternative": "greater"}),
    ("stats.chi2", {"row": "smoker", "column": "outcome"}),
    ("stats.regression", {"y": "y", "x": ["x"]}),
])
def test_checks_agree(data, tool, args):
    ds = data["obs"] if "x" in args or "row" in args else data["plants"] if args.get("column") == "weight" \
        else data["sleep"]
    r = run(tool, dataset=ds, **args)
    outcomes = {name: check(name, ds, args, r)["outcome"] for name in CHECKS}
    assert outcomes["stats.check_formula"] == "pass"
    assert "fail" not in outcomes.values()
    assert "pass" in (outcomes["stats.check_permutation"], outcomes["stats.check_statsmodels"])
    assert stats_text(r)


def test_regression_fields(data):
    r = run("stats.regression", dataset=data["obs"], y="y", x=["x"])
    slope = r["coefficients"][1]
    assert slope["name"] == "x" and slope["estimate"] == pytest.approx(2.0, abs=0.05)
    assert slope["ci"][0] < 2.0 < slope["ci"][1]
    assert r["r2"] > 0.95 and r["formula"] == "y ~ x"
    assert {d["assumption"] for d in r["diagnostics"]} == {"normality", "constant_variance", "linearity",
                                                           "independence"}


# ------------------------------------------------------------------ wrong results are caught
def _corrupt(r, path, value):
    bad = copy.deepcopy(r)
    target = bad
    for k in path[:-1]:
        target = target[k]
    target[path[-1]] = value
    return bad


@pytest.mark.parametrize("path,value,also_permutation", [
    (("statistic", "value"), -2.5, False),
    (("p",), 0.004, True),           # a p-value that flips the decision
    (("df",), 18.0, False),
    (("effect", "ci"), [-3.0, 0.1], False),
    (("spec", "levels"), ["2", "1"], False),
    (("significant",), True, False),
])
def test_formula_check_catches(data, path, value, also_permutation):
    a = {"column": "extra", "by": "drug"}
    r = run("stats.ttest", dataset=data["sleep"], **a)
    bad = _corrupt(r, path, value)
    assert check("stats.check_formula", data["sleep"], a, bad)["outcome"] == "fail"
    if also_permutation:
        assert check("stats.check_permutation", data["sleep"], a, bad)["outcome"] == "fail"
        assert check("stats.check_statsmodels", data["sleep"], a, bad)["outcome"] == "fail"


def test_checks_reject_a_result_from_another_dataset(data):
    a = {"column": "weight", "by": "group", "levels": ["ctrl", "trt1"]}
    r = run("stats.ttest", dataset=data["plants"], **a)
    other = _corrupt(r, ("dataset", "key"), data["sleep"]["key"])
    assert check("stats.check_formula", data["plants"], a, other)["outcome"] == "fail"


def test_permutation_tolerance():
    # an exact test on 4 + 4 values can't go below 1/35: Welch's 0.005 against 0.0286 agrees
    assert abs(0.0286 - 0.005) <= S._perm_p_tolerance(0.005, 0.0286, 35, True)
    # Monte Carlo: a p-value off by half is not within tolerance
    assert abs(0.40 - 0.20) > S._perm_p_tolerance(0.20, 0.40, 10_000, False)


# ------------------------------------------------------------------ refusals the model sees
def test_errors_name_the_problem(data):
    plants = data["plants"]
    assert "has 3 levels (ctrl, trt1, trt2); this test compares exactly 2" in \
        fails("stats.ttest", dataset=plants, column="weight", by="group")
    assert "level 'trt9' is not in column 'group'; its levels are: ctrl, trt1, trt2" in \
        fails("stats.mannwhitney", dataset=plants, column="weight", by="group", levels=["ctrl", "trt9"])
    assert "column 'yield' is not in plants.csv" in fails("stats.anova", dataset=plants, column="yield", by="group")
    assert "is text, not numeric" in fails("stats.correlation", dataset=plants, x="weight", y="group")
    assert "a column is used twice" in fails("stats.regression", dataset=data["obs"], y="y", x=["x", "X"])
