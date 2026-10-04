import pytest

from sciai.tools.registry import all_tools, get
from sciai.tools.sandbox import InlineRunner

R = InlineRunner()


def run(tool, **args):
    out = R.run(tool, args)
    assert out.ok, out.error
    return out.value["result"]


def test_registry_specs_are_complete():
    names = {t.name for t in all_tools()}
    assert {"sympy.diff", "sympy.integrate", "numeric.quad", "units.convert", "numeric.check_identity"} <= names
    for t in all_tools():
        assert t.kind in ("solver", "checker") and t.schema["type"] == "object" and t.domain


@pytest.mark.parametrize("tool,args,expected", [
    ("sympy.diff", {"expr": "x**3", "var": "x"}, "3*x**2"),
    ("sympy.diff", {"expr": "x**3", "var": "x", "method": "first_principles"}, "3*x**2"),
    ("sympy.diff", {"expr": "x**3", "var": "x", "method": "expand_first"}, "3*x**2"),
    ("sympy.integrate", {"expr": "2*x", "var": "x", "lower": "0", "upper": "3"}, "9"),
    ("sympy.limit", {"expr": "sin(x)/x", "var": "x", "point": "0"}, "1"),
    ("sympy.limit", {"expr": "sin(x)/x", "var": "x", "point": "0", "method": "series"}, "1"),
    ("sympy.simplify", {"expr": "sin(x)**2 + cos(x)**2"}, "1"),
    ("sympy.factor", {"expr": "x**2 - 1"}, "(x - 1)*(x + 1)"),
    ("sympy.subs", {"expr": "x**2", "values": {"x": "3"}}, "9"),
])
def test_solvers(tool, args, expected):
    assert run(tool, **args)["value"] == expected


def test_solve_methods_agree():
    a = run("sympy.solve", equation="x**2 - 5*x + 6 = 0", var="x")
    b = run("sympy.solve", equation="x**2 - 5*x + 6 = 0", var="x", method="roots")
    assert [v["value"] for v in a["value"]] == [v["value"] for v in b["value"]] == ["2", "3"]


def test_numeric_solvers():
    q = run("numeric.quad", expr="exp(-x**2)", var="x", lower="0", upper="1")
    assert abs(q["value"] - 0.746824132812427) < 1e-12
    r = run("numeric.root", equation="cos(x) = x", var="x", bracket=["0", "1"])
    assert abs(r["value"] - 0.7390851332151607) < 1e-12
    assert run("units.convert", value=36.0, from_unit="km/h", to_unit="m/s")["value"] == pytest.approx(10.0)


@pytest.mark.parametrize("tool,good,bad", [
    ("numeric.check_identity", {"a": "(x+1)**2", "b": "x**2+2*x+1"}, {"a": "(x+1)**2", "b": "x**2+1"}),
    ("numeric.check_derivative", {"f": "sin(x)", "df": "cos(x)", "var": "x"},
     {"f": "sin(x)", "df": "-cos(x)", "var": "x"}),
    ("numeric.check_antiderivative", {"f": "cos(x)", "F": "sin(x)", "var": "x"},
     {"f": "cos(x)", "F": "-sin(x)", "var": "x"}),
    ("numeric.check_antiderivative", {"f": "2*x", "var": "x", "lower": "0", "upper": "3", "value": "9"},
     {"f": "2*x", "var": "x", "lower": "0", "upper": "3", "value": "8"}),
    ("numeric.check_solution", {"equation": "x**2 = 4", "var": "x", "solutions": ["-2", "2"]},
     {"equation": "x**2 = 4", "var": "x", "solutions": ["2"]}),
    ("numeric.check_limit", {"expr": "(1+1/x)**x", "var": "x", "point": "oo", "value": "E"},
     {"expr": "(1+1/x)**x", "var": "x", "point": "oo", "value": "3"}),
    ("numeric.check_series", {"expr": "sin(x)", "series": "x - x**3/6", "var": "x", "order": 4},
     {"expr": "sin(x)", "series": "x - x**3/3", "var": "x", "order": 4}),
    ("numeric.check_value", {"expr": "sqrt(2)", "value": "1.4142135623730951"},
     {"expr": "sqrt(2)", "value": "1.41"}),
    ("numeric.check_quad", {"expr": "exp(-x**2)", "var": "x", "lower": "0", "upper": "1",
                            "value": "0.746824132812427"},
     {"expr": "exp(-x**2)", "var": "x", "lower": "0", "upper": "1", "value": "0.75"}),
    ("numeric.check_root", {"equation": "cos(x) = x", "var": "x", "root": 0.7390851332151607},
     {"equation": "cos(x) = x", "var": "x", "root": 0.8}),
    ("units.check_convert", {"value": 36.0, "from_unit": "km/h", "to_unit": "m/s", "converted": 10.0},
     {"value": 36.0, "from_unit": "km/h", "to_unit": "m/s", "converted": 36.0}),
    ("sympy.equals", {"a": "sin(2*x)", "b": "2*sin(x)*cos(x)"}, {"a": "sin(2*x)", "b": "sin(x)"}),
])
def test_checkers_pass_and_fail(tool, good, bad):
    assert run(tool, **good)["outcome"] == "pass"
    assert run(tool, **bad)["outcome"] == "fail"


def test_checker_respects_assumptions():
    # sqrt(x**2) == x only for x >= 0
    assert run("numeric.check_identity", a="sqrt(x**2)", b="x")["outcome"] == "fail"
    assert run("numeric.check_identity", a="sqrt(x**2)", b="x", assumptions={"x": "positive"})["outcome"] == "pass"


def test_scripts_are_generated():
    spec = get("sympy.integrate")
    script = spec.script({"expr": "x**2", "var": "x"})
    assert "integrate" in script and "import sympy" in script


def test_schema_rejects_bad_args():
    out = R.run("sympy.diff", {"expr": "x", "var": "x", "evil": 1})
    assert not out.ok
