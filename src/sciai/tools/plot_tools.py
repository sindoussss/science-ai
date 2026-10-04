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
