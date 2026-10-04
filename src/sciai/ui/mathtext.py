"""One-line Unicode math for display: -pi**2 -> −π², x**2*cos(x) -> x²·cos(x).

Display only. Values come from stored node results through the safe parser;
nothing here feeds back into the graph.
"""
from __future__ import annotations

import re
from typing import Any, Callable

import sympy as sp
from sympy.printing.precedence import PRECEDENCE
from sympy.printing.str import StrPrinter

from sciai.graph.model import Node, NodeType
from sciai.llm.actions import PLACEHOLDER
from sciai.tools.parsing import ParseError, parse

_SUP = str.maketrans("0123456789-+()n", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁺⁽⁾ⁿ")
_NAMES = {"pi": "π", "oo": "∞", "inf": "∞", "theta": "θ", "alpha": "α", "beta": "β", "gamma": "γ",
          "delta": "δ", "lambda": "λ", "mu": "μ", "sigma": "σ", "omega": "ω", "phi": "φ", "tau": "τ"}


class _UnicodePrinter(StrPrinter):
    def _print_Pi(self, expr: Any) -> str:  # noqa: N802
        return "π"

    def _print_Exp1(self, expr: Any) -> str:  # noqa: N802
        return "e"

    def _print_ImaginaryUnit(self, expr: Any) -> str:  # noqa: N802
        return "i"

    def _print_Infinity(self, expr: Any) -> str:  # noqa: N802
        return "∞"

    def _print_NegativeInfinity(self, expr: Any) -> str:  # noqa: N802
        return "-∞"

    def _print_Symbol(self, expr: Any) -> str:  # noqa: N802
        return _NAMES.get(expr.name, expr.name)

    def _print_Float(self, expr: Any) -> str:  # noqa: N802
        return f"{float(expr):.10g}"

    def _print_Pow(self, expr: Any, rational: bool = False) -> str:  # noqa: N802
        base, exp = expr.args
        if exp == sp.Rational(1, 2):
            inner = self._print(base)
            return f"√{inner}" if base.is_Atom else f"√({inner})"
        if exp.is_Integer and exp > 0:
            b = self.parenthesize(base, PRECEDENCE["Pow"], strict=True)
            return b + str(exp).translate(_SUP)
        if exp.is_Integer and exp < 0:
            return super()._print_Pow(expr, rational)
        b = self.parenthesize(base, PRECEDENCE["Pow"], strict=True)
        e = self._print(exp)
        return f"{b}^{e}" if (exp.is_Integer or exp.is_Symbol) else f"{b}^({e})"

    def _print_Mul(self, expr: Any) -> str:  # noqa: N802
        s = super()._print_Mul(expr)
        return s.replace("*", "·")


def _finish(s: str) -> str:
    s = re.sub(r"(?<![\w.])(\d+)·(?=[A-Za-zπθαβγδλμσωφτe√(])", r"\1", s)  # 2·x -> 2x
    return s.replace("-", "−")


def expr_text(expr: sp.Basic) -> str:
    return _finish(_UnicodePrinter().doprint(expr))


def value_text(raw: str) -> str:
    """Pretty-print a stored value; fall back to light prose cleanup if it doesn't parse."""
    try:
        return expr_text(parse(raw))
    except (ParseError, ValueError, TypeError):
        return prose_text(raw)


def prose_text(s: str) -> str:
    """Model-written prose around values: pi -> π, x**2 / x^2 -> x², minus signs."""
    s = re.sub(r"\bpi\b", "π", s)
    s = re.sub(r"\*\*(\d+)|\^(\d+)", lambda m: (m.group(1) or m.group(2)).translate(_SUP), s)
    s = re.sub(r"(?<=\s)-(?=\s)", "−", s)
    s = re.sub(r"(?<=[=(\s])-(?=[\wπ√(])", "−", s)
    return s


def answer_text(final: Node, lookup: Callable[[str], Node | None]) -> str:
    """Math text for a final answer: re-render its template with pretty values.

    ``lookup`` maps a handle (n6) to a node. Any placeholder it can't resolve
    falls back to the stored rendered answer, so the display never invents values.
    """
    template = (final.tool_inputs or {}).get("template") if final.type == NodeType.FINAL else None
    if not template:
        return prose_text(final.content)
    parts: list[str] = []
    pos = 0
    for m in PLACEHOLDER.finditer(template):
        node = lookup(m.group(1))
        if node is None or not node.result:
            return prose_text(final.content)
        parts.append(prose_text(template[pos:m.start()]))
        parts.append(_result_text(node))
        pos = m.end()
    parts.append(prose_text(template[pos:]))
    return "".join(parts).strip()


def _result_text(node: Node) -> str:
    r = node.result or {}
    kind = r.get("kind")
    if kind == "expr":
        text = value_text(str(r.get("value", "")))
    elif kind == "number":
        text = _finish(f"{float(r['value']):.10g}")
    elif kind == "list":
        text = ", ".join(value_text(str(v.get("value", v)) if isinstance(v, dict) else str(v))
                         for v in r.get("value", []))
    else:
        text = prose_text(node.display_result())
    units = r.get("units")
    return f"{text} {units}" if units else text

