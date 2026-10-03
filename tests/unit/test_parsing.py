import pytest
import sympy as sp

from sciai.tools.parsing import ParseError, parse, parse_relation


@pytest.mark.parametrize("text", [
    "__import__('os')", "x.__class__", "lambda: 1", "open('f')", "eval('1')", "x[0]", "{1: 2}",
    "'s'", "x if y else z", "a; b", "exec", "Symbol('x')", "Integer(3)", "sin", "9**9**9",
    "2^(10^7)", "factorial(100000)", "factorial(10**6)", "1" * 80, "x @ y", "_x", "x = 1",
])
def test_rejects_unsafe_or_explosive(text):
    with pytest.raises(ParseError):
        parse(text)


def test_parses_ordinary_math():
    x = sp.Symbol("x")
    assert parse("x^2 + 2*x + 1") == x**2 + 2 * x + 1
    assert parse("sin(x)/x") == sp.sin(x) / x
    assert parse("E**(I*pi)") == -1
    assert parse("ln(x)") == sp.log(x)


def test_assumptions_apply():
    e = parse("sqrt(x**2)", {"x": "positive"})
    assert e == sp.Symbol("x", positive=True)


def test_relation():
    x = sp.Symbol("x")
    assert parse_relation("x**2 = 4") == sp.Eq(x**2, 4)
    assert parse_relation("x - 1") == sp.Eq(x - 1, 0)
    with pytest.raises(ParseError):
        parse_relation("x <= 1")
