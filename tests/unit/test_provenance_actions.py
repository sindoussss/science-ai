import pytest

from sciai.controller.provenance import numbers_in, unsourced
from sciai.llm.actions import ActionError, check_answer_template, parse_action


def test_numbers_in():
    assert numbers_in("x**2 + 3.50*y - 1e3") == {"2", "3.5", "1000"}
    assert numbers_in({"a": ["x1", "2.0"]}) == {"2"}


def test_unsourced_allows_ancestor_and_structural_numbers():
    args = {"expr": "x**2 + 7.25", "values": {"x": "3.7"}}
    assert unsourced(args, ["f(x) = x**2 + 7.25"], 10) == ["3.7"]
    assert unsourced(args, ["7.25", {"result": "3.7"}], 10) == []
    assert unsourced({"expr": "x**2 + 11"}, [], 10) == ["11"]


def test_template_rules():
    assert check_answer_template("It is {{n3}}.", ["n3"]) == ["n3"]
    with pytest.raises(ActionError):
        check_answer_template("It is {{n3}} or 4.", ["n3"])
    with pytest.raises(ActionError):
        check_answer_template("It is {{n4}}.", ["n3"])
    with pytest.raises(ActionError):
        check_answer_template("No refs.", ["n3"])


def test_parse_action():
    a = parse_action('```json\n{"action":"ask_user","question":"which x?"}\n```')
    assert a.kind == "ask_user"
    with pytest.raises(ActionError):
        parse_action("not json")
    with pytest.raises(ActionError):
        parse_action('{"action":"call_tool","tool":"sympy.diff"}')
    with pytest.raises(ActionError):
        parse_action('{"action":"finish","answer_template":"x","answer_nodes":["bad handle"]}')


def test_quantity_provenance_needs_matching_si_value_and_dimension():
    from sciai.controller.provenance import unsourced

    givens = {"givens": {"v": {"value": 60, "unit": "mph"}, "t": {"value": 2, "unit": "s"}}}
    ok = {"expr": "v*t", "values": {"v": {"value": 96.56064, "unit": "km/h"}, "t": {"value": 2, "unit": "s"}}}
    assert unsourced(ok, [givens], 10) == []
    wrong_unit = {"expr": "v*t", "values": {"v": {"value": 60, "unit": "km/h"}, "t": {"value": 2, "unit": "s"}}}
    assert unsourced(wrong_unit, [givens], 10) == ["60 km/h"]
    wrong_dim = {"expr": "v", "values": {"v": {"value": 2, "unit": "m"}}}
    assert unsourced(wrong_dim, [givens], 10) == ["2 m"]
    # A quantity result of an ancestor sources the same value in other units.
    result = {"kind": "quantity", "value": 26.8224, "unit": "m/s", "si_value": 26.8224,
              "si_unit": "meter / second", "dims": {"[length]": 1, "[time]": -1}}
    assert unsourced({"expr": "v", "values": {"v": {"value": 60, "unit": "mph"}}}, [result], 10) == []
    # Plain numbers keep the Phase 1 rule.
    assert unsourced({"expr": "x*37"}, ["question with 37"], 10) == []
    assert unsourced({"expr": "x*37"}, ["question"], 10) == ["37"]


def test_si_value_of_a_given_sources_plain_si_numbers():
    from sciai.controller.provenance import unsourced

    givens = {"givens": {"R": {"value": 1, "unit": "kohm"}, "C": {"value": 1, "unit": "uF"},
                         "t": {"value": 2, "unit": "ms"}}}
    args = {"equation": "Derivative(v(t), t) = -v(t)/(1000*1e-6)", "t_span": ["0", "0.002"]}
    assert unsourced(args, [givens], 10) == []
    assert unsourced({"t_span": ["0", "0.003"]}, [givens], 10) == ["0.003"]
