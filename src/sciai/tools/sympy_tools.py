"""Symbolic solvers (SymPy). Each one has alternative ``method``s so a retry can
use a genuinely different algorithm instead of re-running the same one."""
from __future__ import annotations

from typing import Any

import sympy as sp

from sciai.graph.model import Domain
from sciai.tools.parsing import ParseError, expr_result, parse, parse_relation, var
from sciai.tools.registry import ASSUMPTIONS_SCHEMA, ToolSpec, register, schema

EXPR = {"type": "string", "maxLength": 2000}
VAR = {"type": "string", "pattern": "^[A-Za-z][A-Za-z0-9_]{0,29}$"}
BOUND = {"type": "string", "maxLength": 200}


def _a(args: dict[str, Any]) -> dict[str, str]:
    return args.get("assumptions") or {}


def _long_algebra(args: dict[str, Any]) -> bool:
    """Step-type rule: long algebraic manipulation always gets checked."""
    expr = args.get("expr", "")
    return len(expr) > 60 or expr.count("(") > 6


def _script(header: str, body: str) -> str:
    return f"import sympy as sp\nfrom sympy import *\n\n# {header}\n{body}\n"


# ------------------------------------------------------------------- simplify
def simplify_fn(args: dict[str, Any]) -> dict[str, Any]:
    e = parse(args["expr"], _a(args))
    method = args.get("method", "default")
    fn = {"default": sp.simplify, "ratsimp": sp.ratsimp, "trigsimp": sp.trigsimp,
          "powsimp": sp.powsimp, "radsimp": sp.radsimp}[method]
    return {"result": expr_result(fn(e)), "confidence": None, "meta": {"method": method}}


register(ToolSpec(
    name="sympy.simplify", domain=Domain.MATH, kind="solver",
    description="simplify an expression",
    schema=schema({"expr": EXPR, "method": {"enum": ["default", "ratsimp", "trigsimp", "powsimp", "radsimp"]},
                   "assumptions": ASSUMPTIONS_SCHEMA}, ["expr"]),
    fn=simplify_fn, expr_args=("expr",), always_check=_long_algebra,
    methods=("default", "ratsimp", "trigsimp", "powsimp", "radsimp"),
    script=lambda a: _script("simplify", f"print(simplify(sympify({a['expr']!r})))"),
))


def expand_fn(args: dict[str, Any]) -> dict[str, Any]:
    e = parse(args["expr"], _a(args))
    method = args.get("method", "default")
    out = sp.expand(e) if method == "default" else sp.expand_mul(sp.expand_power_base(e))
    return {"result": expr_result(out), "confidence": None, "meta": {"method": method}}


register(ToolSpec(
    name="sympy.expand", domain=Domain.MATH, kind="solver", description="expand products and powers",
    schema=schema({"expr": EXPR, "method": {"enum": ["default", "mul"]}, "assumptions": ASSUMPTIONS_SCHEMA},
                  ["expr"]),
    fn=expand_fn, expr_args=("expr",), always_check=_long_algebra, methods=("default", "mul"),
    script=lambda a: _script("expand", f"print(expand(sympify({a['expr']!r})))"),
))


def factor_fn(args: dict[str, Any]) -> dict[str, Any]:
    e = parse(args["expr"], _a(args))
    method = args.get("method", "default")
    out = sp.factor(e) if method == "default" else sp.factor_terms(e)
    return {"result": expr_result(out), "confidence": None, "meta": {"method": method}}


register(ToolSpec(
    name="sympy.factor", domain=Domain.MATH, kind="solver", description="factor an expression",
    schema=schema({"expr": EXPR, "method": {"enum": ["default", "terms"]}, "assumptions": ASSUMPTIONS_SCHEMA},
                  ["expr"]),
    fn=factor_fn, expr_args=("expr",), always_check=_long_algebra, methods=("default", "terms"),
    script=lambda a: _script("factor", f"print(factor(sympify({a['expr']!r})))"),
))


# ---------------------------------------------------------------------- solve
def solve_fn(args: dict[str, Any]) -> dict[str, Any]:
    rel = parse_relation(args["equation"], _a(args))
    x = var(args["var"], _a(args))
    method = args.get("method", "default")
    expr = sp.simplify(rel.lhs - rel.rhs)
    if method == "default":
        sols = sp.solve(expr, x, dict=False)
    elif method == "solveset":
        s = sp.solveset(expr, x, domain=sp.S.Reals if x.is_real else sp.S.Complexes)
        if not isinstance(s, sp.FiniteSet):
            raise ValueError(f"solution set is not finite: {s}")
        sols = list(s)
    else:  # roots: polynomial-only algorithm
        poly = sp.Poly(expr, x)
        sols = list(sp.roots(poly).keys())
        if sum(sp.roots(poly).values()) != poly.degree():
            raise ValueError("roots() could not find all roots")
    sols = sorted({sp.nsimplify(s) if s.is_Float else s for s in sols}, key=sp.default_sort_key)
    return {
        "result": {"kind": "list", "value": [expr_result(s) for s in sols]},
        "confidence": None,
        "meta": {"method": method, "count": len(sols)},
    }


register(ToolSpec(
    name="sympy.solve", domain=Domain.MATH, kind="solver",
    description="solve an equation 'lhs = rhs' (or expr = 0) for var",
    schema=schema({"equation": EXPR, "var": VAR, "method": {"enum": ["default", "solveset", "roots"]},
                   "assumptions": ASSUMPTIONS_SCHEMA}, ["equation", "var"]),
    fn=solve_fn, expr_args=("equation",), methods=("default", "solveset", "roots"),
    script=lambda a: _script("solve", f"x = Symbol({a['var']!r})\nlhs, _, rhs = {a['equation']!r}.partition('=')\n"
                             f"print(solve(sympify(lhs) - sympify(rhs or '0'), x))"),
))


# ----------------------------------------------------------------------- diff
def diff_fn(args: dict[str, Any]) -> dict[str, Any]:
    e = parse(args["expr"], _a(args))
    x = var(args["var"], _a(args))
    order = int(args.get("order", 1))
    method = args.get("method", "default")
    if method == "default":
        out = sp.diff(e, x, order)
    elif method == "expand_first":
        out = sp.simplify(sp.diff(sp.expand(e), x, order))
    else:  # first principles: repeated limit of the difference quotient
        h = sp.Dummy("h")
        out = e
        for _ in range(order):
            out = sp.limit((out.subs(x, x + h) - out) / h, h, 0)
        out = sp.simplify(out)
    return {"result": expr_result(out), "confidence": None, "meta": {"method": method}}


register(ToolSpec(
    name="sympy.diff", domain=Domain.MATH, kind="solver", description="derivative of expr w.r.t. var",
    schema=schema({"expr": EXPR, "var": VAR, "order": {"type": "integer", "minimum": 1, "maximum": 10},
                   "method": {"enum": ["default", "first_principles", "expand_first"]},
                   "assumptions": ASSUMPTIONS_SCHEMA}, ["expr", "var"]),
    fn=diff_fn, expr_args=("expr",), methods=("default", "first_principles", "expand_first"),
    script=lambda a: _script("diff", f"x = Symbol({a['var']!r})\n"
                             f"print(diff(sympify({a['expr']!r}), x, {int(a.get('order', 1))}))"),
))


# ------------------------------------------------------------------ integrate
def integrate_fn(args: dict[str, Any]) -> dict[str, Any]:
    e = parse(args["expr"], _a(args))
    x = var(args["var"], _a(args))
    method = args.get("method", "default")
    flags = {"default": {}, "risch": {"risch": True}, "meijerg": {"meijerg": True},
             "manual": {"manual": True}, "heurisch": {"heurisch": True}}[method]
    lower, upper = args.get("lower"), args.get("upper")
    if (lower is None) != (upper is None):
        raise ValueError("give both lower and upper, or neither")
    if lower is None:
        out = sp.integrate(e, x, **flags)
    else:
        out = sp.integrate(e, (x, parse(lower, _a(args)), parse(upper, _a(args))), **flags)
    if out.has(sp.Integral):
        raise ValueError(f"method {method!r} could not evaluate the integral")
    out = sp.simplify(out)
    return {"result": expr_result(out), "confidence": None, "meta": {"method": method,
                                                                      "definite": lower is not None}}


register(ToolSpec(
    name="sympy.integrate", domain=Domain.MATH, kind="solver",
    description="integral of expr d(var); give lower and upper for a definite integral",
    schema=schema({"expr": EXPR, "var": VAR, "lower": BOUND, "upper": BOUND,
                   "method": {"enum": ["default", "risch", "meijerg", "manual", "heurisch"]},
                   "assumptions": ASSUMPTIONS_SCHEMA}, ["expr", "var"]),
    fn=integrate_fn, expr_args=("expr", "lower", "upper"),
    methods=("default", "manual", "meijerg", "risch", "heurisch"), cost=3,
    script=lambda a: _script("integrate", f"x = Symbol({a['var']!r})\n" + (
        f"print(integrate(sympify({a['expr']!r}), (x, sympify({a['lower']!r}), sympify({a['upper']!r}))))"
        if a.get("lower") is not None else f"print(integrate(sympify({a['expr']!r}), x))")),
))


# ---------------------------------------------------------------------- limit
def limit_fn(args: dict[str, Any]) -> dict[str, Any]:
    e = parse(args["expr"], _a(args))
    x = var(args["var"], _a(args))
    point = parse(args["point"], _a(args))
    direction = args.get("dir", "+-")
    method = args.get("method", "default")
    if method == "default":
        out = sp.limit(e, x, point, dir=direction)
    else:  # series-based: leading term of the expansion about the point
        if point in (sp.oo, -sp.oo):
            t = sp.Dummy("t", positive=True)
            out = sp.limit(e.subs(x, (1 if point == sp.oo else -1) / t), t, 0, dir="+")
        else:
            out = sp.series(e, x, point, 3).removeO().subs(x, point)
            out = sp.simplify(out)
    return {"result": expr_result(out), "confidence": None, "meta": {"method": method}}


register(ToolSpec(
    name="sympy.limit", domain=Domain.MATH, kind="solver", description="limit of expr as var -> point",
    schema=schema({"expr": EXPR, "var": VAR, "point": BOUND, "dir": {"enum": ["+", "-", "+-"]},
                   "method": {"enum": ["default", "series"]}, "assumptions": ASSUMPTIONS_SCHEMA},
                  ["expr", "var", "point"]),
    fn=limit_fn, expr_args=("expr", "point"), methods=("default", "series"),
    script=lambda a: _script("limit", f"x = Symbol({a['var']!r})\n"
                             f"print(limit(sympify({a['expr']!r}), x, sympify({a['point']!r})))"),
))


# --------------------------------------------------------------------- series
def series_fn(args: dict[str, Any]) -> dict[str, Any]:
    e = parse(args["expr"], _a(args))
    x = var(args["var"], _a(args))
    point = parse(args.get("point", "0"), _a(args))
    order = int(args.get("order", 6))
    out = sp.series(e, x, point, order).removeO()
    return {"result": expr_result(out), "confidence": None, "meta": {"order": order}}


register(ToolSpec(
    name="sympy.series", domain=Domain.MATH, kind="solver",
    description="Taylor series of expr about point, terms below (var-point)**order",
    schema=schema({"expr": EXPR, "var": VAR, "point": BOUND, "order": {"type": "integer", "minimum": 1,
                                                                        "maximum": 20},
                   "assumptions": ASSUMPTIONS_SCHEMA}, ["expr", "var"]),
    fn=series_fn, expr_args=("expr", "point"),
    script=lambda a: _script("series", f"x = Symbol({a['var']!r})\n"
                             f"print(series(sympify({a['expr']!r}), x, sympify({a.get('point', '0')!r}), "
                             f"{int(a.get('order', 6))}))"),
))


# ----------------------------------------------------------------------- subs
def subs_fn(args: dict[str, Any]) -> dict[str, Any]:
    e = parse(args["expr"], _a(args))
    values = {var(k, _a(args)): parse(v, _a(args)) for k, v in args["values"].items()}
    out = sp.simplify(e.subs(values))
    return {"result": expr_result(out), "confidence": None, "meta": {}}


register(ToolSpec(
    name="sympy.subs", domain=Domain.MATH, kind="solver",
    description="substitute values into expr exactly, e.g. values={'x': 'pi'}",
    schema=schema({"expr": EXPR, "values": {"type": "object", "additionalProperties": BOUND},
                   "assumptions": ASSUMPTIONS_SCHEMA}, ["expr", "values"]),
    fn=subs_fn, expr_args=("expr", "values"),
    script=lambda a: _script("subs", "values = {Symbol(k): sympify(v) for k, v in "
                             f"{dict(a['values'])!r}.items()}}\nprint(simplify(sympify({a['expr']!r}).subs(values)))"),
))


# ------------------------------------------------------------ symbolic checker
def equals_fn(args: dict[str, Any]) -> dict[str, Any]:
    a = parse(args["a"], _a(args))
    b = parse(args["b"], _a(args))
    diff = sp.simplify(a - b)
    if diff == 0:
        outcome = "pass"
    else:
        eq = a.equals(b)
        outcome = "pass" if eq is True else "fail" if eq is False else "inconclusive"
    return {"result": {"kind": "check", "outcome": outcome, "difference": str(diff)},
            "confidence": None, "meta": {}}


register(ToolSpec(
    name="sympy.equals", domain=Domain.MATH, kind="checker",
    description="symbolic check that a - b simplifies to zero",
    schema=schema({"a": EXPR, "b": EXPR, "assumptions": ASSUMPTIONS_SCHEMA}, ["a", "b"]),
    fn=equals_fn, expr_args=("a", "b"),
))

__all__ = ["ParseError"]
