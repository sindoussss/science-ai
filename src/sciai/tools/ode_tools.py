"""ODE solvers: closed form (SymPy dsolve) and numeric (SciPy solve_ivp), each checking the other.

Every equation and right-hand side goes through the restricted parser
(``tools/parsing.py``); ``solve_ivp`` gets a function compiled from the parsed SymPy
expression, never from the model's text. Values are plain numbers in SI units.
"""
from __future__ import annotations

import random
from typing import Any

import numpy as np
import sympy as sp
from scipy.integrate import solve_ivp

from sciai.graph.model import Domain
from sciai.tools.parsing import canonical, expr_result, parse, parse_relation, var
from sciai.tools.registry import ASSUMPTIONS_SCHEMA, ToolSpec, register, schema
from sciai.tools.sympy_tools import BOUND, EXPR, VAR

DPS = 30
MAX_ORDER = 4
DEFAULT_SPAN = 1.0
CHECK_POINTS = 20
DSOLVE_HINTS = ("default", "separable", "1st_linear", "1st_exact", "Bernoulli",
                "nth_linear_constant_coeff_homogeneous", "nth_linear_constant_coeff_undetermined_coefficients",
                "nth_linear_constant_coeff_variation_of_parameters")
IVP_METHODS = ("RK45", "DOP853", "Radau", "BDF", "LSODA")

IC = {"type": "object", "properties": {"at": BOUND, "value": BOUND,
                                       "order": {"type": "integer", "minimum": 0, "maximum": MAX_ORDER - 1}},
      "required": ["at", "value"], "additionalProperties": False}
ICS = {"type": "array", "items": IC, "maxItems": MAX_ORDER}
SPAN = {"type": "array", "items": BOUND, "minItems": 2, "maxItems": 2}


def _a(args: dict[str, Any]) -> dict[str, str]:
    return args.get("assumptions") or {}


def _check(outcome: str, **detail: Any) -> dict[str, Any]:
    return {"result": {"kind": "check", "outcome": outcome, **detail}, "confidence": None, "meta": {}}


def _num(text: str, a: dict[str, str] | None = None) -> float:
    v = parse(text, a)
    if v.free_symbols:
        raise ValueError(f"{text!r} must be a number")
    return float(sp.N(v))


# ------------------------------------------------------------------ dsolve
def _ode(args: dict[str, Any]) -> tuple[sp.Eq, sp.FunctionClass, sp.Symbol, dict[sp.Basic, sp.Expr], int]:
    a = _a(args)
    name = args["func"]
    t = var(args["var"], a)
    eq = parse_relation(args["equation"], a, functions=(name,))
    if not isinstance(eq, sp.Eq):
        raise ValueError("the equation simplifies to a constant; it is not an ODE")
    F = sp.Function(name)
    order = 0
    for d in eq.atoms(sp.Derivative):
        if d.expr != F(t) or any(v != t for v, _ in d.variable_count):
            raise ValueError(f"only derivatives of {name}({t}) with respect to {t} are allowed, got {d}")
        order = max(order, d.derivative_count)
    if order == 0:
        raise ValueError(f"the equation has no derivative of {name}({t})")
    for applied in eq.atoms(sp.core.function.AppliedUndef):
        if applied != F(t):
            raise ValueError(f"{applied} is not {name}({t})")
    ics: dict[sp.Basic, sp.Expr] = {}
    for ic in args.get("ics") or []:
        point, value, k = parse(ic["at"], a), parse(ic["value"], a), int(ic.get("order", 0))
        if k >= order:
            raise ValueError(f"an initial condition of order {k} is too high for an order-{order} ODE")
        key = F(point) if k == 0 else F(t).diff(t, k).subs(t, point)
        ics[key] = value
    return eq, F, t, ics, order


def dsolve_fn(args: dict[str, Any]) -> dict[str, Any]:
    eq, F, t, ics, _ = _ode(args)
    hint = args.get("method", "default")
    sol = sp.dsolve(eq, F(t), ics=ics or None, hint=hint)
    if isinstance(sol, list):
        items = [expr_result(s.rhs) for s in sol]
        return {"result": {"kind": "list", "value": items}, "confidence": None, "meta": {"hint": hint}}
    return {"result": expr_result(sol.rhs), "confidence": None, "meta": {"hint": hint}}


def _canon_dsolve(args: dict[str, Any]) -> dict[str, Any]:
    eq, F, t, ics, _ = _ode(args)
    return {"equation": canonical(eq), "func": args["func"], "var": args["var"],
            "ics": sorted((canonical(k), canonical(v)) for k, v in ics.items()),
            "method": args.get("method", "default")}


register(ToolSpec(
    name="ode.dsolve", domain=Domain.PHYSICS, kind="solver",
    description='closed-form ODE solution, e.g. equation="Derivative(y(t), t) = -y(t)/tau", func="y", '
                'var="t", ics=[{"at": "0", "value": "5"}] (order 1 = first derivative); numbers in SI',
    schema=schema({"equation": EXPR, "func": VAR, "var": VAR, "ics": ICS, "t_span": SPAN,
                   "method": {"enum": list(DSOLVE_HINTS)}, "assumptions": ASSUMPTIONS_SCHEMA},
                  ["equation", "func", "var"]),
    fn=dsolve_fn, always_check=True, methods=DSOLVE_HINTS, canonical=_canon_dsolve, cost=3,
    script=lambda a: ("import sympy as sp\nfrom sympy import *\n"
                      f"t = Symbol({a['var']!r}); {a['func']} = Function({a['func']!r})\n"
                      f"lhs, rhs = {a['equation']!r}.split('=')\n"
                      "print(dsolve(Eq(sympify(lhs, locals()), sympify(rhs, locals()))))\n"),
))


def _sub_solution(expr: sp.Basic, F: sp.FunctionClass, t: sp.Symbol, y: sp.Expr) -> sp.Expr:
    return expr.subs(F(t), y).doit()


def check_residual_fn(args: dict[str, Any]) -> dict[str, Any]:
    """Residual: substitute the solution back into the ODE and its initial conditions."""
    eq, F, t, ics, _ = _ode(args)
    y = parse(args["solution"], _a(args))
    residual = _sub_solution(eq.lhs - eq.rhs, F, t, y)
    problems = []
    if sp.simplify(residual) != 0:
        # Numeric fallback, relative to the size of the equation's own terms (not to 1: a decaying
        # solution makes every term tiny, and an absolute floor would pass anything).
        rng = random.Random(7)
        syms = sorted(residual.free_symbols, key=str)
        span = args.get("t_span")
        lo, hi = (_num(span[0]), _num(span[1])) if span else (0.1, 2.0)
        lhs = _sub_solution(eq.lhs, F, t, y)
        rhs = _sub_solution(eq.rhs, F, t, y)
        for _ in range(8):
            point = {s: sp.Float(rng.uniform(lo, hi) if s == t else rng.uniform(0.1, 2.0), DPS) for s in syms}
            try:
                r = abs(complex(sp.N(residual.subs(point), DPS)))
                scale = max(abs(complex(sp.N(lhs.subs(point), DPS))), abs(complex(sp.N(rhs.subs(point), DPS))))
            except (TypeError, ValueError):
                return _check("inconclusive", reason="the residual could not be evaluated")
            if r > 1e-9 * scale + 1e-300:
                problems.append(f"ODE residual {r:.3g} (terms of size {scale:.3g}) at "
                                f"{ {str(k): float(v) for k, v in point.items()} }")
                break
    for key, value in ics.items():
        if isinstance(key, sp.Derivative) or isinstance(key, sp.Subs):
            got = key.subs(F(t), y).doit() if not isinstance(key, sp.Subs) else \
                key.expr.subs(F(t), y).doit().subs(key.variables[0], key.point[0])
        else:
            got = y.subs(t, key.args[0])
        diff = sp.simplify(got - value)
        if diff != 0:
            try:
                bad = abs(complex(sp.N(diff, DPS))) > 1e-9 * max(1.0, abs(complex(sp.N(value, DPS))))
            except (TypeError, ValueError):
                bad = True
            if bad:
                problems.append(f"initial condition {key} = {value} gives {sp.simplify(got)}")
    return _check("fail" if problems else "pass", reason="; ".join(problems) or "residual is zero")


register(ToolSpec(
    name="ode.check_residual", domain=Domain.PHYSICS, kind="checker",
    description="substitute an ODE solution back into the equation and initial conditions",
    schema=schema({"equation": EXPR, "func": VAR, "var": VAR, "ics": ICS, "solution": EXPR, "t_span": SPAN,
                   "assumptions": ASSUMPTIONS_SCHEMA}, ["equation", "func", "var", "solution"]),
    fn=check_residual_fn,
))


def _first_order_system(eq: sp.Eq, F: sp.FunctionClass, t: sp.Symbol, order: int):
    """y^(n) = g(t, y, ..., y^(n-1)) as a numpy function of (t, [y, y', ...])."""
    top = F(t).diff(t, order)
    sols = sp.solve(eq, top)
    if len(sols) != 1:
        raise ValueError("the ODE can't be written explicitly for its highest derivative")
    ys = sp.symbols(f"_y0:{order}")
    g = sols[0]
    for k in range(order - 1, 0, -1):
        g = g.subs(F(t).diff(t, k), ys[k])
    g = g.subs(F(t), ys[0])
    if g.free_symbols - {t, *ys}:
        raise ValueError(f"unbound parameters {sorted(map(str, g.free_symbols - {t, *ys}))}")
    fg = sp.lambdify([t, *ys], g, modules="numpy")

    def rhs(tt: float, yy: np.ndarray) -> list[float]:
        return [*yy[1:], float(np.real(fg(tt, *yy)))]
    return rhs


def check_numeric_fn(args: dict[str, Any]) -> dict[str, Any]:
    """Symbolic vs numeric: integrate the ODE numerically from its initial conditions (LSODA)
    and compare with the closed form at 20 points."""
    eq, F, t, ics, order = _ode(args)
    y = parse(args["solution"], _a(args))
    if y.free_symbols - {t}:
        return _check("inconclusive", reason="not applicable: the solution has free constants or parameters")
    points = {}
    for key, value in ics.items():
        if isinstance(key, sp.Subs):
            k, point = key.expr.derivative_count, key.point[0]
        elif isinstance(key, sp.Derivative):
            return _check("inconclusive", reason="not applicable: unsupported initial condition form")
        else:
            k, point = 0, key.args[0]
        points[k] = (float(sp.N(point)), float(sp.N(value)))
    if sorted(points) != list(range(order)) or len({p for p, _ in points.values()}) != 1:
        return _check("inconclusive", reason="not applicable: needs every initial condition at one point")
    t0 = points[0][0]
    span = args.get("t_span")
    t1 = _num(span[1]) if span else t0 + DEFAULT_SPAN
    rhs = _first_order_system(eq, F, t, order)
    y0 = [points[k][1] for k in range(order)]
    t_eval = np.linspace(t0, t1, CHECK_POINTS)
    sol = solve_ivp(rhs, (t0, t1), y0, method="LSODA", t_eval=t_eval, rtol=1e-10, atol=1e-12 * max(1.0, *map(abs, y0)))
    if not sol.success:
        return _check("inconclusive", reason=f"numeric integration failed: {sol.message}")
    f = sp.lambdify([t], y, modules="numpy")
    with np.errstate(all="ignore"):
        exact = np.real(np.asarray(f(t_eval), dtype=complex)) * np.ones_like(t_eval)
    scale = max(1e-300, float(np.max(np.abs(exact))), max(map(abs, y0)))
    err = float(np.max(np.abs(sol.y[0] - exact)))
    ok = err <= 1e-6 * scale
    return _check("pass" if ok else "fail", max_error=err, scale=scale, points=CHECK_POINTS, t_span=[t0, t1])


register(ToolSpec(
    name="ode.check_numeric", domain=Domain.PHYSICS, kind="checker",
    description="compare a closed-form ODE solution with a numeric integration",
    schema=schema({"equation": EXPR, "func": VAR, "var": VAR, "ics": ICS, "solution": EXPR, "t_span": SPAN,
                   "assumptions": ASSUMPTIONS_SCHEMA}, ["equation", "func", "var", "solution"]),
    fn=check_numeric_fn,
))


# --------------------------------------------------------------- solve_ivp
def _ivp(args: dict[str, Any]):
    names = args["funcs"]
    if len(names) != len(args["rhs"]) or len(names) != len(args["y0"]) or len(set(names)) != len(names):
        raise ValueError("funcs, rhs and y0 must have the same length and distinct names")
    t = var(args["var"])
    ys = [var(n) for n in names]
    exprs = [parse(r) for r in args["rhs"]]
    unbound = set().union(*(e.free_symbols for e in exprs)) - {t, *ys}
    if unbound:
        raise ValueError(f"unbound symbols in rhs: {sorted(map(str, unbound))}; substitute their values")
    y0 = [_num(v) for v in args["y0"]]
    t0, t1 = (_num(v) for v in args["t_span"])
    if not t1 > t0:
        raise ValueError("t_span must increase")
    return t, ys, exprs, y0, (t0, t1)


def _integrate(args: dict[str, Any], method: str, rtol: float):
    t, ys, exprs, y0, span = _ivp(args)
    f = sp.lambdify([t, *ys], exprs, modules="numpy")

    def rhs(tt: float, yy: np.ndarray) -> np.ndarray:
        return np.array([float(np.real(v)) for v in f(tt, *yy)], dtype=float)

    t_eval = np.linspace(span[0], span[1], int(args.get("points", 50)))
    atol = rtol * 1e-3 * max(1.0, *map(abs, y0))
    return solve_ivp(rhs, span, y0, method=method, t_eval=t_eval, rtol=rtol, atol=atol)


def solve_ivp_fn(args: dict[str, Any]) -> dict[str, Any]:
    method = args.get("method", "RK45")
    rtol = float(args.get("rtol", 1e-8))
    sol = _integrate(args, method, rtol)
    if not sol.success:
        raise ValueError(f"solve_ivp failed: {sol.message}")
    if not np.all(np.isfinite(sol.y)):
        raise ValueError("the solution is not finite")
    # Confidence comes from the solver, never from the model: a very expensive run is
    # usually a stiff problem for an explicit method.
    stiff = method in ("RK45", "DOP853") and sol.nfev > 20000
    confidence = 0.6 if stiff else 1.0
    result = {"kind": "series", "var": args["var"], "funcs": list(args["funcs"]),
              "t": sol.t.tolist(), "y": sol.y.tolist(), "value": [float(row[-1]) for row in sol.y],
              "method": method, "rtol": rtol}
    return {"result": result, "confidence": confidence,
            "meta": {"nfev": int(sol.nfev), "stiff_suspected": stiff, "message": sol.message}}


def _canon_ivp(args: dict[str, Any]) -> dict[str, Any]:
    t, ys, exprs, y0, span = _ivp(args)
    return {"rhs": [canonical(e) for e in exprs], "funcs": list(args["funcs"]), "var": args["var"],
            "y0": [repr(v) for v in y0], "t_span": [repr(v) for v in span], "points": int(args.get("points", 50)),
            "method": args.get("method", "RK45"), "rtol": float(args.get("rtol", 1e-8))}


register(ToolSpec(
    name="ode.solve_ivp", domain=Domain.PHYSICS, kind="solver",
    description='numeric ODE system y\' = rhs, e.g. funcs=["x", "v"], rhs=["v", "-9.81"], var="t", '
                'y0=["0", "20"], t_span=["0", "2"]; value = y at the end of t_span',
    schema=schema({"rhs": {"type": "array", "items": EXPR, "minItems": 1, "maxItems": 6},
                   "funcs": {"type": "array", "items": VAR, "minItems": 1, "maxItems": 6},
                   "var": VAR, "y0": {"type": "array", "items": BOUND, "minItems": 1, "maxItems": 6},
                   "t_span": SPAN, "points": {"type": "integer", "minimum": 2, "maximum": 2000},
                   "method": {"enum": list(IVP_METHODS)}, "rtol": {"type": "number", "minimum": 1e-12,
                                                                   "maximum": 1e-3}},
                  ["rhs", "funcs", "var", "y0", "t_span"]),
    fn=solve_ivp_fn, always_check=True, methods=IVP_METHODS, canonical=_canon_ivp,
    script=lambda a: ("import numpy as np, sympy as sp\nfrom scipy.integrate import solve_ivp\n"
                      f"t = sp.Symbol({a['var']!r}); ys = sp.symbols({' '.join(a['funcs'])!r}, seq=True)\n"
                      f"f = sp.lambdify([t, *ys], [sp.sympify(r) for r in {list(a['rhs'])!r}], 'numpy')\n"
                      f"y0 = [float(sp.sympify(v)) for v in {list(a['y0'])!r}]\n"
                      f"span = [float(sp.sympify(v)) for v in {list(a['t_span'])!r}]\n"
                      "print(solve_ivp(lambda tt, yy: f(tt, *yy), span, y0).y[:, -1])\n"),
))


def _series_close(a: np.ndarray, b: np.ndarray, rtol: float) -> tuple[bool, float]:
    scale = max(1e-300, float(np.max(np.abs(b))))
    err = float(np.max(np.abs(a - b)))
    return err <= max(10 * rtol, 1e-9) * scale, err


def check_ivp_fn(args: dict[str, Any]) -> dict[str, Any]:
    """Alternative algorithm: rerun with the implicit Radau method at 100x tighter tolerance."""
    run_args = {k: v for k, v in args.items() if k not in ("y", "rtol_used")}
    rtol = float(args.get("rtol_used", 1e-8))
    sol = _integrate(run_args, "Radau", max(rtol / 100, 1e-13))
    if not sol.success:
        return _check("inconclusive", reason=f"Radau failed: {sol.message}")
    claimed = np.asarray(args["y"], dtype=float)
    if claimed.shape != sol.y.shape:
        return _check("inconclusive", reason="the claimed series has a different shape")
    ok, err = _series_close(claimed, sol.y, rtol)
    return _check("pass" if ok else "fail", max_error=err, method="Radau", rtol=max(rtol / 100, 1e-13))


_IVP_CHECK_PROPS = {"rhs": {"type": "array", "items": EXPR}, "funcs": {"type": "array", "items": VAR}, "var": VAR,
                    "y0": {"type": "array", "items": BOUND}, "t_span": SPAN, "points": {"type": "integer"},
                    "y": {"type": "array"}, "rtol_used": {"type": "number"}}

register(ToolSpec(
    name="ode.check_ivp", domain=Domain.PHYSICS, kind="checker",
    description="rerun a numeric ODE solution with Radau at a tighter tolerance",
    schema=schema(_IVP_CHECK_PROPS, ["rhs", "funcs", "var", "y0", "t_span", "y"]),
    fn=check_ivp_fn,
))


def check_ivp_symbolic_fn(args: dict[str, Any]) -> dict[str, Any]:
    """For one first-order equation: the closed form from dsolve against the numeric series."""
    if len(args["funcs"]) != 1:
        return _check("inconclusive", reason="not applicable: closed form only checked for one equation")
    t, ys, exprs, y0, span = _ivp(args)
    Y = sp.Function("Y")
    try:
        sol = sp.dsolve(sp.Eq(Y(t).diff(t), exprs[0].subs(ys[0], Y(t))), Y(t), ics={Y(span[0]): y0[0]})
    except (NotImplementedError, ValueError) as exc:
        return _check("inconclusive", reason=f"not applicable: no closed form ({exc})")
    if isinstance(sol, list):
        return _check("inconclusive", reason="not applicable: several closed-form branches")
    f = sp.lambdify([t], sol.rhs, modules="numpy")
    ts = np.linspace(span[0], span[1], int(args.get("points", 50)))
    with np.errstate(all="ignore"):
        exact = np.real(np.asarray(f(ts), dtype=complex)) * np.ones_like(ts)
    if not np.all(np.isfinite(exact)):
        return _check("inconclusive", reason="the closed form is not finite on t_span")
    ok, err = _series_close(np.asarray(args["y"][0], dtype=float), exact, float(args.get("rtol_used", 1e-8)))
    return _check("pass" if ok else "fail", max_error=err, closed_form=str(sol.rhs))


register(ToolSpec(
    name="ode.check_ivp_symbolic", domain=Domain.PHYSICS, kind="checker",
    description="compare a numeric ODE solution with its closed form",
    schema=schema(_IVP_CHECK_PROPS, ["rhs", "funcs", "var", "y0", "t_span", "y"]),
    fn=check_ivp_symbolic_fn,
))

