import pytest

from sciai.tools.sandbox import InlineRunner
from sciai.verify.checks import plans_for

R = InlineRunner()
RC = {"equation": "Derivative(v(t), t) = -v(t)/(1000*1e-6)", "func": "v", "var": "t",
      "ics": [{"at": "0", "value": "5"}]}


def run(tool, **args):
    out = R.run(tool, args)
    assert out.ok, out.error
    return out.value["result"]


def outcome(tool, **args):
    return run(tool, **args)["outcome"]


def test_rc_discharge_closed_form_and_checks():
    r = run("ode.dsolve", **RC)
    assert r["value"] == "5*exp(-1000.0*t)"
    plans = plans_for("ode.dsolve", RC, r)
    assert {p.method for p, _ in plans} == {"residual", "symbolic_vs_numeric"}
    for plan, args in plans:
        assert outcome(plan.checker, **args) == "pass", plan.method


def test_residual_catches_wrong_decaying_solution():
    # Regression: an absolute floor of 1.0 let a wrong solution of small size pass.
    wrong = {**RC, "solution": "5*exp(-900*t)", "t_span": ["0", "0.005"]}
    assert outcome("ode.check_residual", **wrong) == "fail"
    assert outcome("ode.check_numeric", **wrong) == "fail"
    wrong_ic = {**RC, "solution": "4*exp(-1000*t)"}
    assert outcome("ode.check_residual", **wrong_ic) == "fail"


def test_second_order_with_derivative_ic():
    args = {"equation": "Derivative(x(t), t, 2) = -4*x(t)", "func": "x", "var": "t",
            "ics": [{"at": "0", "value": "1"}, {"at": "0", "value": "0", "order": 1}]}
    r = run("ode.dsolve", **args)
    assert r["value"] == "cos(2*t)"
    for plan, a in plans_for("ode.dsolve", args, r):
        assert outcome(plan.checker, **a) == "pass"


@pytest.mark.parametrize("equation", [
    "Derivative(v(t), t) = __import__('os').system('true')",
    "Derivative(v(t), t) = v(t).__class__",
    "Derivative(v(t), t) = eval('1')",
    "Derivative(v(t), t) = lambda: 1",
])
def test_unsafe_rhs_refused(equation):
    out = R.run("ode.dsolve", {**RC, "equation": equation})
    assert not out.ok


@pytest.mark.parametrize("rhs", ["__import__('os').getcwd()", "open('x')", "y.__dict__"])
def test_unsafe_ivp_rhs_refused(rhs):
    out = R.run("ode.solve_ivp", {"rhs": [rhs], "funcs": ["y"], "var": "t", "y0": ["1"], "t_span": ["0", "1"]})
    assert not out.ok


def test_ode_shape_errors():
    assert not R.run("ode.dsolve", {**RC, "equation": "v(t) = 3"}).ok  # no derivative
    assert not R.run("ode.dsolve", {**RC, "equation": "Derivative(v(t), t) = w(t)"}).ok  # other function
    assert not R.run("ode.dsolve", {**RC, "ics": [{"at": "0", "value": "1", "order": 1}]}).ok  # IC too high
    assert not R.run("ode.solve_ivp", {"rhs": ["k*y"], "funcs": ["y"], "var": "t", "y0": ["1"],
                                       "t_span": ["0", "1"]}).ok  # unbound k


def test_projectile_ivp_and_checks():
    args = {"rhs": ["vx", "vy", "0", "-9.81"], "funcs": ["x", "y", "vx", "vy"], "var": "t",
            "y0": ["0", "0", "10", "10"], "t_span": ["0", "2"]}
    r = run("ode.solve_ivp", **args)
    assert r["kind"] == "series" and r["value"][0] == pytest.approx(20.0)
    assert r["value"][1] == pytest.approx(20 - 9.81 * 2, rel=1e-8)
    plans = dict((p.method, (p, a)) for p, a in plans_for("ode.solve_ivp", args, r))
    p, a = plans["alt_algorithm"]
    assert outcome(p.checker, **a) == "pass"
    p, a = plans["symbolic_vs_numeric"]
    assert outcome(p.checker, **a) == "inconclusive"  # a system: closed form only for one equation


def test_rc_ivp_matches_closed_form_and_catches_tampering():
    args = {"rhs": ["-v/(1000*1e-6)"], "funcs": ["v"], "var": "t", "y0": ["5"], "t_span": ["0", "0.002"]}
    r = run("ode.solve_ivp", **args)
    assert r["value"][0] == pytest.approx(5 * 2.718281828459045 ** -2, rel=1e-6)  # 0.677 V
    for plan, a in plans_for("ode.solve_ivp", args, r):
        assert outcome(plan.checker, **a) == "pass", plan.method
    tampered = {**r, "y": [[v * 1.01 for v in r["y"][0]]]}
    for plan, a in plans_for("ode.solve_ivp", args, tampered):
        assert outcome(plan.checker, **a) == "fail", plan.method


def test_linalg_exact_and_float():
    r = run("linalg.solve", A=[["2", "1"], ["1", "3"]], b=["3", "5"])
    assert [v["value"] for v in r["value"]] == ["4/5", "7/5"] and r["labels"] == ["x1", "x2"]
    r2 = run("linalg.solve", A=[["2.5", "1"], ["1", "3"]], b=["3", "5"])
    args = {"A": [["2.5", "1"], ["1", "3"]], "b": ["3", "5"]}
    for plan, a in plans_for("linalg.solve", args, r2):
        assert outcome(plan.checker, **a) == "pass"
    assert outcome("linalg.check_residual", **args, solution=["1", "1"]) == "fail"
    assert not R.run("linalg.solve", {"A": [["1", "2"], ["2", "4"]], "b": ["1", "2"]}).ok  # singular
