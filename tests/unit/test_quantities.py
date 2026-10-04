import pytest

from sciai.domains.physics import quantities as Q


def test_si_key_rounds_to_nine_digits():
    a = Q.to_si(Q.make(60, "mph")).magnitude
    b = Q.to_si(Q.make(96.56064, "km/h")).magnitude
    assert a != b  # float noise: 26.8224 vs 26.822400000000002
    assert Q.si_key(a) == Q.si_key(b) == "2.68224000e+01"
    assert Q.si_key(Q.to_si(Q.make(96.56, "km/h")).magnitude) != Q.si_key(a)
    assert Q.si_key(-0.0) == Q.si_key(0.0)


def test_canonical_object_matches_across_units():
    assert Q.canonical_object({"value": 60, "unit": "mph"}) == \
        Q.canonical_object({"value": 96.56064, "unit": "km/h"})
    assert Q.canonical_object({"value": 60, "unit": "mph", "kind": "speed"}) != \
        Q.canonical_object({"value": 60, "unit": "mph"})


def test_temperatures_are_kelvin_and_offset_needs_a_kind():
    absolute = Q.to_si(Q.make(10, "degC", "absolute_temperature"))
    diff = Q.to_si(Q.make(10, "degC", "temperature_difference"))
    assert absolute.magnitude == pytest.approx(283.15) and str(absolute.units) == "kelvin"
    assert diff.magnitude == pytest.approx(10.0)
    with pytest.raises(Q.QuantityError, match="needs kind"):
        Q.make(10, "degC")
    with pytest.raises(Q.QuantityError, match="needs kind"):
        Q.make(10, "degC", "speed")


@pytest.mark.parametrize("value,unit,kind", [
    (1, "kg", "speed"), (3, "m", "time"), (2, "V", "current"), (1, "K", "efficiency"),
])
def test_kind_must_match_dimension(value, unit, kind):
    with pytest.raises(Q.QuantityError, match="kind"):
        Q.make(value, unit, kind)


@pytest.mark.parametrize("bad", ["__import__('os')", "m; rm", "a" * 101, "frobnicate", None])
def test_bad_units_refused(bad):
    with pytest.raises(Q.QuantityError):
        Q.make(1, bad)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), True, "3"])
def test_value_must_be_finite_number(value):
    with pytest.raises(Q.QuantityError):
        Q.make(value, "m")


def test_range_rules_key_on_kind_not_dimension():
    def res(v, unit, kind):
        return Q.result(Q.make(v, unit, kind), unit, kind)

    assert Q.range_issues(res(-5, "K", "absolute_temperature"))
    assert not Q.range_issues(res(-5, "K", "temperature_difference"))  # same dimension, different kind
    assert Q.range_issues(res(1.2, "", "efficiency"))
    assert not Q.range_issues(res(1.2, "", "ratio"))
    assert Q.range_issues(res(4e8, "m/s", "speed"))
    assert Q.range_issues(res(-3, "m/s", "speed"))
    assert not Q.range_issues(res(-3, "m/s", "velocity"))
    assert not Q.range_issues(res(-3, "m/s", None))
    assert not Q.range_issues(res(0.9, "1", "efficiency"))


def test_result_shape_and_kind_issues():
    r = Q.result(Q.make(60, "mph", "speed"), "mph", "speed")
    assert r["kind"] == "quantity" and r["quantity_kind"] == "speed"
    assert r["si_unit"] == "meter / second" and r["dims"] == {"[length]": 1, "[time]": -1}
    assert Q.kind_issues(r) == []
    assert Q.kind_issues({**r, "dims": {"[length]": 1}})
    assert Q.kind_issues({**r, "quantity_kind": "nonsense"})
