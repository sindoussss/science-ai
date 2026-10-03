"""Numeric solvers (SciPy/NumPy) and numeric checkers (mpmath).

Checkers are deliberately built on a *different* method from the solver they
check: symbolic results are checked numerically, SciPy results are checked with
mpmath's tanh-sinh quadrature or high-precision residuals.
"""
from __future__ import annotations

import math
import random
from typing import Any, Callable

import mpmath
import numpy as np
import sympy as sp
from scipy import integrate as sci_integrate
from scipy import optimize

from sciai.graph.model import Domain
from sciai.tools.parsing import expr_result, number_result, parse, parse_relation, var
from sciai.tools.registry import ASSUMPTIONS_SCHEMA, ToolSpec, register, schema
from sciai.tools.sympy_tools import BOUND, EXPR, VAR

DPS = 30
REL_TOL = 1e-8


def _a(args: dict[str, Any]) -> dict[str, str]:
    return args.get("assumptions") or {}


def _check(outcome: str, **detail: Any) -> dict[str, Any]:
    return {"result": {"kind": "check", "outcome": outcome, **detail}, "confidence": None, "meta": {}}


def _mp_func(expr: sp.Expr, symbols: list[sp.Symbol]) -> Callable[..., Any]:
    return sp.lambdify(symbols, expr, modules="mpmath")


def _close(a: Any, b: Any, tol: float = REL_TOL) -> bool:
    a, b = complex(a), complex(b)
    if not (math.isfinite(a.real) and math.isfinite(b.real)):
        return False
    return abs(a - b) <= tol * max(1.0, abs(a), abs(b))


def _sample_point(symbol: sp.Symbol, rng: random.Random) -> float:
    if symbol.is_positive:
        return rng.uniform(0.2, 3.0)
    if symbol.is_negative:
        return -rng.uniform(0.2, 3.0)
    if symbol.is_nonnegative:
        return rng.uniform(0.0, 3.0)
    if symbol.is_integer:
        return float(rng.randint(-5, 5))
    return rng.uniform(-3.0, 3.0)


def _finite_real(v: Any) -> bool:
    try:
        c = complex(v)
    except (TypeError, ValueError):
        return False
    return math.isfinite(c.real) and math.isfinite(c.imag)


# ---------------------------------------------------------------- solvers
def evaluate_fn(args: dict[str, Any]) -> dict[str, Any]:
    e = parse(args["expr"], _a(args))
    values = {var(k, _a(args)): parse(v, _a(args)) for k, v in (args.get("values") or {}).items()}
    out = e.subs(values)
    if out.free_symbols:
        raise ValueError(f"unbound symbols: {sorted(map(str, out.free_symbols))}")
    f = sp.lambdify([], out, modules="numpy")
    with np.errstate(all="ignore"):
        val = complex(f())
    if not _finite_real(val):
        raise ValueError("result is not finite")
    if abs(val.imag) > 1e-12 * max(1.0, abs(val.real)):
        return {"result": {"kind": "number", "value": val.real, "imag": val.imag}, "confidence": None, "meta": {}}
    return {"result": number_result(val.real), "confidence": None, "meta": {"precision": "float64"}}


register(ToolSpec(
    name="numeric.evaluate", domain=Domain.MATH, kind="solver",
    description="evaluate expr numerically (float64), values={'x': '2.5'}",
    schema=schema({"expr": EXPR, "values": {"type": "object", "additionalProperties": BOUND},
                   "assumptions": ASSUMPTIONS_SCHEMA}, ["expr"]),
    fn=evaluate_fn, expr_args=("expr", "values"),
    script=lambda a: "import sympy as sp\nfrom sympy import *\n"
                     f"e = sympify({a['expr']!r}).subs({{Symbol(k): sympify(v) for k, v in {dict(a.get('values') or {})!r}.items()}})\n"
                     "print(float(e))\n",
))


def quad_fn(args: dict[str, Any]) -> dict[str, Any]:
    e = parse(args["expr"], _a(args))
    x = var(args["var"], _a(args))
    lo = float(sp.N(parse(args["lower"], _a(args))))
    hi = float(sp.N(parse(args["upper"], _a(args))))
    f = sp.lambdify([x], e, modules="numpy")
    with np.errstate(all="ignore"):
        val, abserr = sci_integrate.quad(lambda t: float(np.real(f(t))), lo, hi, limit=200)
    rel = abserr / max(1.0, abs(val))
    confidence = 1.0 if rel < 1e-10 else 0.95 if rel < 1e-7 else 0.5
    return {"result": number_result(val, abserr=abserr), "confidence": confidence,
            "meta": {"abserr": abserr, "algorithm": "QUADPACK"}}


register(ToolSpec(
    name="numeric.quad", domain=Domain.MATH, kind="solver",
    description="numerical definite integral (SciPy quad)",
    schema=schema({"expr": EXPR, "var": VAR, "lower": BOUND, "upper": BOUND,
                   "assumptions": ASSUMPTIONS_SCHEMA}, ["expr", "var", "lower", "upper"]),
    fn=quad_fn, expr_args=("expr", "lower", "upper"), always_check=True,
    script=lambda a: "import numpy as np, sympy as sp\nfrom scipy.integrate import quad\n"
                     f"x = sp.Symbol({a['var']!r})\nf = sp.lambdify([x], sp.sympify({a['expr']!r}), 'numpy')\n"
                     f"print(quad(f, float(sp.sympify({a['lower']!r})), float(sp.sympify({a['upper']!r}))))\n",
))


def root_fn(args: dict[str, Any]) -> dict[str, Any]:
    rel = parse_relation(args["equation"], _a(args))
    x = var(args["var"], _a(args))
    g = sp.lambdify([x], rel.lhs - rel.rhs, modules="numpy")
    if "bracket" in args:
        a, b = (float(sp.N(parse(s, _a(args)))) for s in args["bracket"])
        sol = optimize.brentq(g, a, b, xtol=1e-14, maxiter=500)
        method = "brentq"
    else:
        x0 = float(sp.N(parse(args.get("x0", "1"), _a(args))))
        res = optimize.root_scalar(g, x0=x0, x1=x0 + 0.1, method="secant", xtol=1e-14, maxiter=500)
        if not res.converged:
            raise ValueError("root finder did not converge")
        sol, method = res.root, "secant"
    resid = abs(g(sol))
    return {"result": number_result(sol, residual=float(resid)),
            "confidence": 1.0 if resid < 1e-9 else 0.5, "meta": {"method": method}}


register(ToolSpec(
    name="numeric.root", domain=Domain.MATH, kind="solver",
    description="numeric root of equation in var; give bracket=[a,b] or x0",
    schema=schema({"equation": EXPR, "var": VAR, "x0": BOUND,
                   "bracket": {"type": "array", "items": BOUND, "minItems": 2, "maxItems": 2},
                   "assumptions": ASSUMPTIONS_SCHEMA}, ["equation", "var"]),
    fn=root_fn, expr_args=("equation", "x0", "bracket"),
))


# --------------------------------------------------------------- checkers
def check_identity_fn(args: dict[str, Any]) -> dict[str, Any]:
    """a == b at random sample points (high-precision mpmath evaluation)."""
    a = parse(args["a"], _a(args))
    b = parse(args["b"], _a(args))
    syms = sorted(a.free_symbols | b.free_symbols, key=str)
    rng = random.Random(args.get("seed", 1234))
    mpmath.mp.dps = DPS
    fa, fb = _mp_func(a, syms), _mp_func(b, syms)
    good, bad = 0, []
    for _ in range(int(args.get("samples", 12)) * 3):
        if good >= int(args.get("samples", 12)):
            break
        pt = [_sample_point(s, rng) for s in syms]
        try:
            va, vb = fa(*pt), fb(*pt)
        except (ZeroDivisionError, ValueError, TypeError, OverflowError):
            continue
        if not (_finite_real(va) and _finite_real(vb)):
            continue
        good += 1
        if not _close(va, vb):
            bad.append({"point": pt, "a": str(va), "b": str(vb)})
    if bad:
        return _check("fail", mismatches=bad[:3], samples=good)
    if good < 3:
        return _check("inconclusive", reason="too few valid sample points", samples=good)
    return _check("pass", samples=good)


register(ToolSpec(
    name="numeric.check_identity", domain=Domain.MATH, kind="checker",
    description="numerically compare two expressions at random points",
    schema=schema({"a": EXPR, "b": EXPR, "samples": {"type": "integer", "minimum": 3, "maximum": 50},
                   "seed": {"type": "integer"}, "assumptions": ASSUMPTIONS_SCHEMA}, ["a", "b"]),
    fn=check_identity_fn, expr_args=("a", "b"),
))


def check_derivative_fn(args: dict[str, Any]) -> dict[str, Any]:
    """df vs numerical differentiation of f (mpmath.diff), at random points."""
    f = parse(args["f"], _a(args))
    df = parse(args["df"], _a(args))
    x = var(args["var"], _a(args))
    order = int(args.get("order", 1))
    others = sorted((f.free_symbols | df.free_symbols) - {x}, key=str)
    rng = random.Random(args.get("seed", 99))
    mpmath.mp.dps = DPS
    good, bad = 0, []
    for _ in range(36):
        if good >= 10:
            break
        fixed = {s: _sample_point(s, rng) for s in others}
        x0 = _sample_point(x, rng)
        fx = sp.lambdify([x], f.subs(fixed), modules="mpmath")
        dfx = sp.lambdify([x], df.subs(fixed), modules="mpmath")
        try:
            num = mpmath.diff(fx, x0, order)
            sym = dfx(x0)
        except (ZeroDivisionError, ValueError, TypeError, OverflowError):
            continue
        if not (_finite_real(num) and _finite_real(sym)):
            continue
        good += 1
        if not _close(num, sym, 1e-7):
            bad.append({"x": x0, "numeric": str(num), "claimed": str(sym)})
    if bad:
        return _check("fail", mismatches=bad[:3], samples=good)
    if good < 3:
        return _check("inconclusive", reason="too few valid points", samples=good)
    return _check("pass", samples=good)


register(ToolSpec(
    name="numeric.check_derivative", domain=Domain.MATH, kind="checker",
    description="compare a claimed derivative with numerical differentiation",
    schema=schema({"f": EXPR, "df": EXPR, "var": VAR, "order": {"type": "integer", "minimum": 1, "maximum": 4},
                   "seed": {"type": "integer"}, "assumptions": ASSUMPTIONS_SCHEMA}, ["f", "df", "var"]),
    fn=check_derivative_fn, expr_args=("f", "df"),
))


def check_antiderivative_fn(args: dict[str, Any]) -> dict[str, Any]:
    """Indefinite: F(b)-F(a) vs mpmath.quad(f, a, b) on random intervals.
    Definite: claimed value vs mpmath.quad over the given bounds."""
    f = parse(args["f"], _a(args))
    x = var(args["var"], _a(args))
    mpmath.mp.dps = DPS
    others = sorted(f.free_symbols - {x}, key=str)
    rng = random.Random(args.get("seed", 7))
    if args.get("lower") is not None:
        if others:
            return _check("inconclusive", reason="free parameters in a definite integral")
        claimed = parse(args["value"], _a(args))
        lo, hi = parse(args["lower"], _a(args)), parse(args["upper"], _a(args))
        fx = sp.lambdify([x], f, modules="mpmath")
        try:
            num = mpmath.quad(fx, [mpmath.mpmathify(sp.N(lo, DPS)) if lo.is_finite else
                                   (mpmath.inf if lo == sp.oo else -mpmath.inf),
                                   mpmath.mpmathify(sp.N(hi, DPS)) if hi.is_finite else
                                   (mpmath.inf if hi == sp.oo else -mpmath.inf)])
            val = complex(sp.N(claimed, DPS))
        except Exception as exc:  # noqa: BLE001 - any numeric failure is inconclusive
            return _check("inconclusive", reason=str(exc))
        ok = _close(num, val, 1e-7)
        return _check("pass" if ok else "fail", numeric=str(num), claimed=str(val))
    F = parse(args["F"], _a(args))
    good, bad = 0, []
    for _ in range(30):
        if good >= 6:
            break
        fixed = {s: _sample_point(s, rng) for s in others}
        a = _sample_point(x, rng)
        b = a + rng.uniform(0.3, 1.5)
        fx = sp.lambdify([x], f.subs(fixed), modules="mpmath")
        Fx = sp.lambdify([x], F.subs(fixed), modules="mpmath")
        try:
            num = mpmath.quad(fx, [a, b])
            claimed = Fx(b) - Fx(a)
        except (ZeroDivisionError, ValueError, TypeError, OverflowError):
            continue
        if not (_finite_real(num) and _finite_real(claimed)):
            continue
        good += 1
        if not _close(num, claimed, 1e-7):
            bad.append({"interval": [a, b], "numeric": str(num), "claimed": str(claimed)})
    if bad:
        return _check("fail", mismatches=bad[:3], samples=good)
    if good < 3:
        return _check("inconclusive", reason="too few valid intervals", samples=good)
    return _check("pass", samples=good)


register(ToolSpec(
    name="numeric.check_antiderivative", domain=Domain.MATH, kind="checker",
    description="check an integral result with tanh-sinh quadrature",
    schema=schema({"f": EXPR, "var": VAR, "F": EXPR, "lower": BOUND, "upper": BOUND, "value": EXPR,
                   "seed": {"type": "integer"}, "assumptions": ASSUMPTIONS_SCHEMA}, ["f", "var"]),
    fn=check_antiderivative_fn, expr_args=("f", "F", "lower", "upper", "value"),
))


def check_solution_fn(args: dict[str, Any]) -> dict[str, Any]:
    """Residual of each claimed solution at high precision; for polynomials also
    compare the count of distinct roots with numpy.roots."""
    rel = parse_relation(args["equation"], _a(args))
    x = var(args["var"], _a(args))
    expr = rel.lhs - rel.rhs
    mpmath.mp.dps = DPS
    bad = []
    sols = [parse(s, _a(args)) for s in args["solutions"]]
    for s in sols:
        try:
            r = complex(sp.N(expr.subs(x, s), DPS))
        except (TypeError, ValueError):
            return _check("inconclusive", reason=f"cannot evaluate residual at {s}")
        scale = max(1.0, abs(complex(sp.N(s, DPS))))
        if abs(r) > 1e-12 * scale:
            bad.append({"solution": str(s), "residual": abs(r)})
    detail: dict[str, Any] = {}
    others = expr.free_symbols - {x}
    if not bad and not others:
        try:
            num, _den = sp.fraction(sp.together(expr))
            poly = sp.Poly(sp.expand(num), x)
        except sp.PolynomialError:
            poly = None
        if poly is not None and poly.degree() > 0:
            coeffs = [complex(c) for c in poly.all_coeffs()]
            roots = np.roots(coeffs)
            distinct: list[complex] = []
            for r in roots:
                if all(abs(r - d) > 1e-6 * max(1, abs(r)) for d in distinct):
                    distinct.append(complex(r))
            if x.is_real:
                distinct = [d for d in distinct if abs(d.imag) < 1e-9]
            detail["numeric_root_count"] = len(distinct)
            if len(distinct) != len(sols):
                bad.append({"reason": f"expected {len(distinct)} distinct roots, got {len(sols)}"})
    if bad:
        return _check("fail", problems=bad, **detail)
    return _check("pass", **detail)


register(ToolSpec(
    name="numeric.check_solution", domain=Domain.MATH, kind="checker",
    description="check that claimed solutions satisfy the equation (and are complete for polynomials)",
    schema=schema({"equation": EXPR, "var": VAR, "solutions": {"type": "array", "items": EXPR},
                   "assumptions": ASSUMPTIONS_SCHEMA}, ["equation", "var", "solutions"]),
    fn=check_solution_fn, expr_args=("equation", "solutions"),
))


def check_value_fn(args: dict[str, Any]) -> dict[str, Any]:
    """Evaluate expr (with values) at 30 digits with mpmath and compare to a claimed value."""
    e = parse(args["expr"], _a(args))
    values = {var(k, _a(args)): parse(v, _a(args)) for k, v in (args.get("values") or {}).items()}
    try:
        ref = complex(sp.N(e.subs(values), DPS))
        claimed = complex(sp.N(parse(args["value"], _a(args)), DPS))
    except (TypeError, ValueError) as exc:
        return _check("inconclusive", reason=str(exc))
    tol = float(args.get("tol", 1e-9))
    return _check("pass" if _close(ref, claimed, tol) else "fail", reference=str(ref), claimed=str(claimed))


register(ToolSpec(
    name="numeric.check_value", domain=Domain.MATH, kind="checker",
    description="compare a claimed value with a 30-digit evaluation",
    schema=schema({"expr": EXPR, "values": {"type": "object", "additionalProperties": BOUND}, "value": EXPR,
                   "tol": {"type": "number"}, "assumptions": ASSUMPTIONS_SCHEMA}, ["expr", "value"]),
    fn=check_value_fn, expr_args=("expr", "values", "value"),
))


def check_limit_fn(args: dict[str, Any]) -> dict[str, Any]:
    """Approach the point numerically (mpmath, 50 digits) and watch convergence."""
    e = parse(args["expr"], _a(args))
    x = var(args["var"], _a(args))
    point = parse(args["point"], _a(args))
    claimed = parse(args["value"], _a(args))
    if e.free_symbols - {x}:
        return _check("inconclusive", reason="free parameters")
    mpmath.mp.dps = 50
    f = sp.lambdify([x], e, modules="mpmath")
    direction = args.get("dir", "+-")
    sides = {"+": [1], "-": [-1], "+-": [1, -1]}[direction]
    vals = []
    try:
        for k in (6, 9, 12):
            for s in sides:
                if point == sp.oo:
                    t = mpmath.mpf(10) ** k
                elif point == -sp.oo:
                    t = -mpmath.mpf(10) ** k
                else:
                    t = mpmath.mpmathify(sp.N(point, 50)) + s * mpmath.mpf(10) ** (-k)
                vals.append(f(t))
    except (ZeroDivisionError, ValueError, TypeError, OverflowError) as exc:
        return _check("inconclusive", reason=str(exc))
    if claimed in (sp.oo, -sp.oo):
        grows = all(abs(complex(v)) > 1e5 for v in vals[-len(sides):])
        return _check("pass" if grows else "fail", approach=[str(v) for v in vals])
    target = complex(sp.N(claimed, 30))
    last = vals[-len(sides):]
    ok = all(_close(v, target, 1e-5) for v in last)
    return _check("pass" if ok else "fail", approach=[mpmath.nstr(v, 12) for v in vals], claimed=str(target))


register(ToolSpec(
    name="numeric.check_limit", domain=Domain.MATH, kind="checker",
    description="check a limit by numerical approach",
    schema=schema({"expr": EXPR, "var": VAR, "point": BOUND, "value": EXPR, "dir": {"enum": ["+", "-", "+-"]},
                   "assumptions": ASSUMPTIONS_SCHEMA}, ["expr", "var", "point", "value"]),
    fn=check_limit_fn, expr_args=("expr", "point", "value"),
))


def check_series_fn(args: dict[str, Any]) -> dict[str, Any]:
    """Truncation error must shrink like h**order near the expansion point."""
    e = parse(args["expr"], _a(args))
    s = parse(args["series"], _a(args))
    x = var(args["var"], _a(args))
    point = parse(args.get("point", "0"), _a(args))
    order = int(args.get("order", 6))
    if (e.free_symbols | s.free_symbols) - {x}:
        return _check("inconclusive", reason="free parameters")
    mpmath.mp.dps = 60
    fe = sp.lambdify([x], e, modules="mpmath")
    fs = sp.lambdify([x], s, modules="mpmath")
    p = mpmath.mpmathify(sp.N(point, 60))
    errs = []
    try:
        for h in (mpmath.mpf("1e-2"), mpmath.mpf("1e-3")):
            errs.append(abs(fe(p + h) - fs(p + h)))
    except (ZeroDivisionError, ValueError, TypeError) as exc:
        return _check("inconclusive", reason=str(exc))
    if errs[1] < mpmath.mpf(10) ** (-40):
        return _check("pass", errors=[mpmath.nstr(e_, 5) for e_ in errs])
    ratio = errs[0] / errs[1] if errs[1] else mpmath.inf
    ok = ratio >= mpmath.mpf(10) ** (order - 0.5)
    return _check("pass" if ok else "fail", errors=[mpmath.nstr(e_, 5) for e_ in errs],
                  ratio=mpmath.nstr(ratio, 5), expected_min=10 ** (order - 0.5))


register(ToolSpec(
    name="numeric.check_series", domain=Domain.MATH, kind="checker",
    description="check a truncated series by its error scaling",
    schema=schema({"expr": EXPR, "series": EXPR, "var": VAR, "point": BOUND,
                   "order": {"type": "integer", "minimum": 1, "maximum": 20},
                   "assumptions": ASSUMPTIONS_SCHEMA}, ["expr", "series", "var"]),
    fn=check_series_fn, expr_args=("expr", "series", "point"),
))


def check_quad_fn(args: dict[str, Any]) -> dict[str, Any]:
    """A different quadrature algorithm (mpmath tanh-sinh) than SciPy's QUADPACK."""
    return check_antiderivative_fn({"f": args["expr"], "var": args["var"], "lower": args["lower"],
                                    "upper": args["upper"], "value": args["value"],
                                    "assumptions": args.get("assumptions")})


register(ToolSpec(
    name="numeric.check_quad", domain=Domain.MATH, kind="checker",
    description="check a numeric integral with a different quadrature algorithm",
    schema=schema({"expr": EXPR, "var": VAR, "lower": BOUND, "upper": BOUND, "value": EXPR,
                   "assumptions": ASSUMPTIONS_SCHEMA}, ["expr", "var", "lower", "upper", "value"]),
    fn=check_quad_fn, expr_args=("expr", "lower", "upper", "value"),
))


def check_root_fn(args: dict[str, Any]) -> dict[str, Any]:
    """Residual of a numeric root at 30 digits plus a sign change around it."""
    rel = parse_relation(args["equation"], _a(args))
    x = var(args["var"], _a(args))
    g = rel.lhs - rel.rhs
    r = float(args["root"])
    mpmath.mp.dps = DPS
    f = sp.lambdify([x], g, modules="mpmath")
    res = abs(f(mpmath.mpf(r)))
    h = 1e-7 * max(1.0, abs(r))
    sign_change = (f(mpmath.mpf(r - h)) * f(mpmath.mpf(r + h))) <= 0
    ok = res < 1e-8 or sign_change
    return _check("pass" if ok else "fail", residual=mpmath.nstr(res, 5), sign_change=bool(sign_change))


register(ToolSpec(
    name="numeric.check_root", domain=Domain.MATH, kind="checker",
    description="check a numeric root by residual and sign change",
    schema=schema({"equation": EXPR, "var": VAR, "root": {"type": "number"},
                   "assumptions": ASSUMPTIONS_SCHEMA}, ["equation", "var", "root"]),
    fn=check_root_fn, expr_args=("equation",),
))

_ = expr_result  # re-exported helpers used by other tool modules
