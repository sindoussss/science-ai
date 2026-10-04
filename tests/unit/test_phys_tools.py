import math

import pytest
import scipy

from sciai.domains.physics.constants import CONSTANTS, codata_release
from sciai.tools.registry import get
from sciai.tools.sandbox import InlineRunner
from sciai.verify.checks import plans_for

R = InlineRunner()


def run(tool, **args):
    out = R.run(tool, args)
    assert out.ok, out.error
    return out.value


def fails(tool, **args):
    out = R.run(tool, args)
    assert not out.ok
    return out.error


def q(value, unit, kind=None):
    return {"value": value, "unit": unit, **({"kind": kind} if kind else {})}


PROJECTILE = {"expr": "v**2*sin(2*theta)/g",
              "values": {"v": q(20, "m/s"), "theta": q(30, "deg"), "g": q(9.81, "m/s^2")},
              "to_unit": "m", "kind": "distance"}


@pytest.mark.parametrize("method", ["default", "si_first"])
def test_projectile_range(method):
    r = run("phys.evaluate", **PROJECTILE, method=method)["result"]
    assert r["value"] == pytest.approx(400 * math.sin(math.radians(60)) / 9.81, rel=1e-12)
    assert r["unit"] == "m" and r["quantity_kind"] == "distance" and r["dims"] == {"[length]": 1}


def test_degrees_converted_before_trig():
    r = run("phys.evaluate", expr="sin(theta)", values={"theta": q(30, "deg")})["result"]
    assert r["value"] == pytest.approx(0.5, abs=1e-15)
    r = run("phys.evaluate", expr="sin(pi/6)", values={})["result"]
    assert r["value"] == pytest.approx(0.5, abs=1e-15)
    r = run("phys.evaluate", expr="cos(theta)", values={"theta": q(0.5, "rad")})["result"]
    assert r["value"] == pytest.approx(0.8775825618903728)


def test_bare_number_in_trig_refused():
    # sin(30) means 30 radians (-0.988), never 30 degrees; an angle must carry its unit.
    assert "angle" in fails("phys.evaluate", expr="sin(30)", values={}).lower()
    assert "angle" in fails("phys.evaluate", expr="sin(theta)", values={"theta": 30}).lower()


def test_trig_of_a_length_refused():
    assert fails("phys.evaluate", expr="sin(x)", values={"x": q(2, "m")})


def test_unit_mismatch_in_sum_refused():
    assert fails("phys.evaluate", expr="a + b", values={"a": q(1, "m"), "b": q(1, "s")})


def test_unbound_symbol_refused():
    assert fails("phys.evaluate", expr="a*b", values={"a": q(1, "m")})


def test_to_unit_must_match():
    assert fails("phys.evaluate", expr="a", values={"a": q(1, "m")}, to_unit="s")


def test_reuse_fingerprint_matches_equal_si_inputs():
    canon = get("phys.evaluate").canonical
    a = canon({"expr": "v*t", "values": {"v": q(60, "mph"), "t": q(2, "s")}})
    b = canon({"expr": "v*t", "values": {"v": q(96.56064, "km/h"), "t": q(2, "s")}})
    c = canon({"expr": "v*t", "values": {"v": q(96.56, "km/h"), "t": q(2, "s")}})
    assert a == b != c


def test_constants_are_codata_2022_from_pinned_scipy():
    assert scipy.__version__ == "1.17.1"
    assert codata_release() == "2022"
    out = run("phys.constant", name="m_e")
    assert out["meta"]["codata"] == "2022" and out["meta"]["scipy"] == "1.17.1"
    assert out["result"]["source"] == "CODATA 2022 via SciPy 1.17.1"


@pytest.mark.parametrize("name", list(CONSTANTS))
def test_every_constant_agrees_with_pint(name):
    r = run("phys.constant", name=name)["result"]
    check = run("phys.check_constant", name=name, si_value=r["si_value"], dims=r["dims"])["result"]
    assert check["outcome"] == "pass", check


def test_constant_check_catches_wrong_value_and_dims():
    r = run("phys.constant", name="G")["result"]
    assert run("phys.check_constant", name="G", si_value=r["si_value"] * 1.001,
               dims=r["dims"])["result"]["outcome"] == "fail"
    assert run("phys.check_constant", name="G", si_value=r["si_value"],
               dims={"[length]": 1})["result"]["outcome"] == "fail"


def test_checks_pass_on_good_result_and_catch_unit_error():
    r = run("phys.evaluate", **PROJECTILE)["result"]
    for plan, args in plans_for("phys.evaluate", PROJECTILE, r):
        assert run(plan.checker, **args)["result"]["outcome"] == "pass", plan.method
    # A unit error: the stored result claims seconds for a length formula.
    bad = {**r, "dims": {"[time]": 1}}
    dims = next(a for p, a in plans_for("phys.evaluate", PROJECTILE, bad) if p.method == "dimensional")
    assert run("phys.check_dimensions", **dims)["result"]["outcome"] == "fail"
    wrong = {**r, "si_value": r["si_value"] * 1.01}
    val = next(a for p, a in plans_for("phys.evaluate", PROJECTILE, wrong) if p.method == "alt_algorithm")
    assert run("phys.check_value", **val)["result"]["outcome"] == "fail"


def test_plausibility_check():
    r = run("phys.evaluate", expr="a", values={"a": q(-3, "K")}, kind="absolute_temperature")["result"]
    assert run("phys.check_plausibility", result=r)["result"]["outcome"] == "fail"
    r = run("phys.evaluate", expr="a", values={"a": q(-3, "K")}, kind="temperature_difference")["result"]
    assert run("phys.check_plausibility", result=r)["result"]["outcome"] == "pass"


def test_dimensional_check_rejects_inconsistent_formula():
    out = run("phys.check_dimensions", expr="a + b", values={"a": q(1, "m"), "b": q(1, "s")},
              dims={"[length]": 1})
    assert out["result"]["outcome"] == "fail"
