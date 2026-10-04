"""Linear systems: exact (SymPy) for rational input, NumPy otherwise; checked by residual."""
from __future__ import annotations

from typing import Any

import numpy as np
import sympy as sp

from sciai.graph.model import Domain
from sciai.tools.parsing import canonical, expr_result, number_result, parse
from sciai.tools.registry import ToolSpec, register, schema
from sciai.tools.sympy_tools import BOUND

MAX_N = 12
DPS = 30
ROW = {"type": "array", "items": BOUND, "minItems": 1, "maxItems": MAX_N}
MATRIX = {"type": "array", "items": ROW, "minItems": 1, "maxItems": MAX_N}


def _check(outcome: str, **detail: Any) -> dict[str, Any]:
    return {"result": {"kind": "check", "outcome": outcome, **detail}, "confidence": None, "meta": {}}


def _system(args: dict[str, Any]) -> tuple[sp.Matrix, sp.Matrix]:
    A = sp.Matrix([[parse(x) for x in row] for row in args["A"]])
    b = sp.Matrix([parse(x) for x in args["b"]])
    n = A.rows
    if A.cols != n or any(len(row) != n for row in args["A"]):
        raise ValueError("A must be square")
    if b.rows != n:
        raise ValueError(f"b must have {n} entries")
    if (A.free_symbols | b.free_symbols) and n > 4:
        raise ValueError("symbolic systems are limited to 4x4")
    return A, b


def solve_fn(args: dict[str, Any]) -> dict[str, Any]:
    A, b = _system(args)
    labels = [f"x{i + 1}" for i in range(A.rows)]
    exact = all(e.is_Rational for e in [*A, *b]) or bool(A.free_symbols | b.free_symbols)
    if exact:
        if A.det() == 0:
            raise ValueError("A is singular: the system has no unique solution")
        x = A.LUsolve(b)
        items = [expr_result(sp.simplify(v)) for v in x]
        return {"result": {"kind": "list", "value": items, "labels": labels}, "confidence": None,
                "meta": {"algorithm": "exact LU (SymPy)"}}
    An = np.array(A.evalf(), dtype=float)
    bn = np.array(b.evalf(), dtype=float).ravel()
    cond = float(np.linalg.cond(An))
    if not np.isfinite(cond) or cond > 1e14:
        raise ValueError("A is singular or nearly singular")
    x = np.linalg.solve(An, bn)
    confidence = 1.0 if cond < 1e8 else 0.5  # ill-conditioned: the solver's own warning sign
    return {"result": {"kind": "list", "value": [number_result(v) for v in x], "labels": labels},
            "confidence": confidence, "meta": {"algorithm": "LAPACK gesv (NumPy)", "condition_number": cond}}


register(ToolSpec(
    name="linalg.solve", domain=Domain.PHYSICS, kind="solver",
    description='solve A x = b, e.g. A=[["2", "1"], ["1", "3"]], b=["3", "5"]',
    schema=schema({"A": MATRIX, "b": ROW}, ["A", "b"]),
    fn=solve_fn, canonical=lambda a: {"A": [[canonical(parse(x)) for x in row] for row in a["A"]],
                                      "b": [canonical(parse(x)) for x in a["b"]]},
    script=lambda a: f"import sympy as sp\nA = sp.Matrix({a['A']!r}).applyfunc(sp.sympify)\n"
                     f"b = sp.Matrix({a['b']!r}).applyfunc(sp.sympify)\nprint(A.LUsolve(b))\n",
))


def check_residual_fn(args: dict[str, Any]) -> dict[str, Any]:
    """A x - b at 30 digits (mpmath), relative to the size of A x and b."""
    A, b = _system(args)
    x = sp.Matrix([parse(v) for v in args["solution"]])
    if x.rows != A.rows:
        return _check("fail", reason=f"{x.rows} values for {A.rows} unknowns")
    if A.free_symbols | b.free_symbols | x.free_symbols:
        r = (A * x - b).applyfunc(sp.simplify)
        return _check("pass" if all(v == 0 for v in r) else "fail", residual=[str(v) for v in r])
    r = [abs(complex(sp.N(v, DPS))) for v in (A * x - b)]
    scale = max(max(abs(complex(sp.N(v, DPS))) for v in A * x), max(abs(complex(sp.N(v, DPS))) for v in b), 1e-300)
    worst = max(r)
    return _check("pass" if worst <= 1e-9 * scale else "fail", max_residual=worst, scale=scale)


register(ToolSpec(
    name="linalg.check_residual", domain=Domain.PHYSICS, kind="checker",
    description="residual of a linear-system solution at 30 digits",
    schema=schema({"A": MATRIX, "b": ROW, "solution": ROW}, ["A", "b", "solution"]),
    fn=check_residual_fn,
))
