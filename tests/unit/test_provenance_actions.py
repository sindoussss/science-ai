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
