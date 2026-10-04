"""Plot tools return data (a PlotSpec), never pixels; the UI renders it."""
from __future__ import annotations

from typing import Any

import numpy as np
import sympy as sp

from sciai.graph.model import Domain
from sciai.tools.parsing import parse, var
from sciai.tools.registry import ASSUMPTIONS_SCHEMA, ToolSpec, register, schema
from sciai.tools.sympy_tools import BOUND, EXPR, VAR


def function_fn(args: dict[str, Any]) -> dict[str, Any]:
    a = args.get("assumptions") or {}
    e = parse(args["expr"], a)
    x = var(args["var"], a)
    lo = float(sp.N(parse(args["lower"], a)))
    hi = float(sp.N(parse(args["upper"], a)))
    n = int(args.get("points", 200))
    xs = np.linspace(lo, hi, n)
    f = sp.lambdify([x], e, modules="numpy")
    with np.errstate(all="ignore"):
        ys = np.asarray(f(xs), dtype=complex) * np.ones_like(xs)
    ys = np.where(np.abs(ys.imag) < 1e-12, ys.real, np.nan)
    ys = [None if not np.isfinite(v) else float(v) for v in ys]
    return {"result": {"kind": "plotspec", "value": {"x": xs.tolist(), "y": ys, "label": str(e),
                                                     "xlabel": str(x)}},
            "confidence": None, "meta": {}}


register(ToolSpec(
    name="plot.function", domain=Domain.MATH, kind="solver",
    description="sample expr over [lower, upper] for plotting",
    schema=schema({"expr": EXPR, "var": VAR, "lower": BOUND, "upper": BOUND,
                   "points": {"type": "integer", "minimum": 10, "maximum": 2000},
                   "assumptions": ASSUMPTIONS_SCHEMA}, ["expr", "var", "lower", "upper"]),
    fn=function_fn, expr_args=("expr", "lower", "upper"),
))


def series_fn(args: dict[str, Any]) -> dict[str, Any]:
    data = args.get("data")
    if not isinstance(data, dict) or data.get("kind") != "series":
        raise ValueError(f"{args['node']} is not a numeric ODE solution (a series)")
    funcs = list(data.get("funcs") or [])
    comp = args.get("component") or (funcs[0] if funcs else None)
    if comp not in funcs:
        raise ValueError(f"component must be one of {funcs}")
    ys = [None if v is None or not np.isfinite(v) else float(v) for v in data["y"][funcs.index(comp)]]
    return {"result": {"kind": "plotspec", "value": {"x": [float(t) for t in data["t"]], "y": ys, "label": comp,
                                                     "xlabel": str(data.get("var", "t"))}},
            "confidence": None, "meta": {}}


def _canon_series(args: dict[str, Any]) -> dict[str, Any]:
    """The plotted node's id fixes its data (a new version is a new id)."""
    return {"node": args["node"], "component": args.get("component")}


register(ToolSpec(
    name="plot.series", domain=Domain.PHYSICS, kind="solver",
    description='plot a numeric ODE solution: node="n5" (an ode.solve_ivp result), component="x"',
    schema=schema({"node": {"type": "string", "pattern": "^(n[0-9]+|[0-9a-f]{32})$"}, "component": VAR,
                   "data": {"type": "object"}}, ["node"]),
    fn=series_fn, node_arg="node", canonical=_canon_series,
))
