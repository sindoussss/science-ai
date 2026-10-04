"""Which independent checks exist for which solver.

Every plan uses a *different method* from the solver it checks (symbolic vs
numeric, a different algorithm, or a known-value comparison). A plan's builder
returns ``None`` when the result can't be fed to the checker (e.g. a Piecewise
the parser doesn't accept); that plan is then not offered.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

Builder = Callable[[dict[str, Any], dict[str, Any]], "dict[str, Any] | None"]


@dataclass(frozen=True)
class CheckPlan:
    method: str        # symbolic_vs_numeric | alt_algorithm | known_value | units
    checker: str       # checker tool name
    build: Builder
    cost: int = 1


def _val(result: dict[str, Any]) -> str | None:
    kind = result.get("kind")
    if kind == "expr":
        return result["value"]
    if kind == "number":
        return repr(float(result["value"]))
    return None


def _with_assumptions(args: dict[str, Any], out: dict[str, Any]) -> dict[str, Any]:
    if args.get("assumptions"):
        out["assumptions"] = args["assumptions"]
    return out


def _diff(a: dict, r: dict) -> dict | None:
    v = _val(r)
    return v and _with_assumptions(a, {"f": a["expr"], "df": v, "var": a["var"], "order": int(a.get("order", 1))})


def _integrate(a: dict, r: dict) -> dict | None:
    v = _val(r)
    if v is None:
        return None
    if a.get("lower") is not None:
        return _with_assumptions(a, {"f": a["expr"], "var": a["var"], "lower": a["lower"],
                                     "upper": a["upper"], "value": v})
    return _with_assumptions(a, {"f": a["expr"], "var": a["var"], "F": v})


def _solve(a: dict, r: dict) -> dict | None:
    if r.get("kind") != "list":
        return None
    sols = [_val(s) for s in r["value"]]
    if any(s is None for s in sols):
        return None
    return _with_assumptions(a, {"equation": a["equation"], "var": a["var"], "solutions": sols})


def _identity(a: dict, r: dict) -> dict | None:
    v = _val(r)
    return v and _with_assumptions(a, {"a": a["expr"], "b": v})


def _limit(a: dict, r: dict) -> dict | None:
    v = _val(r)
    return v and _with_assumptions(a, {"expr": a["expr"], "var": a["var"], "point": a["point"], "value": v,
                                       "dir": a.get("dir", "+-")})


def _series(a: dict, r: dict) -> dict | None:
    v = _val(r)
    return v and _with_assumptions(a, {"expr": a["expr"], "series": v, "var": a["var"],
                                       "point": a.get("point", "0"), "order": int(a.get("order", 6))})


def _value(a: dict, r: dict) -> dict | None:
    v = _val(r)
    out = {"expr": a["expr"], "value": v}
    if a.get("values"):
        out["values"] = a["values"]
    return v and _with_assumptions(a, out)


def _quad(a: dict, r: dict) -> dict | None:
    v = _val(r)
    return v and _with_assumptions(a, {"expr": a["expr"], "var": a["var"], "lower": a["lower"],
                                       "upper": a["upper"], "value": v})


def _root(a: dict, r: dict) -> dict | None:
    if r.get("kind") != "number":
        return None
    return _with_assumptions(a, {"equation": a["equation"], "var": a["var"], "root": float(r["value"])})


def _convert(a: dict, r: dict) -> dict | None:
    if r.get("kind") != "number":
        return None
    return {"value": a["value"], "from_unit": a["from_unit"], "to_unit": a["to_unit"],
            "converted": float(r["value"])}


PLANS: dict[str, list[CheckPlan]] = {
    "sympy.diff": [CheckPlan("symbolic_vs_numeric", "numeric.check_derivative", _diff)],
    "sympy.integrate": [CheckPlan("symbolic_vs_numeric", "numeric.check_antiderivative", _integrate, 2)],
    "sympy.solve": [CheckPlan("symbolic_vs_numeric", "numeric.check_solution", _solve)],
    "sympy.simplify": [CheckPlan("symbolic_vs_numeric", "numeric.check_identity", _identity)],
    "sympy.expand": [CheckPlan("symbolic_vs_numeric", "numeric.check_identity", _identity),
                     CheckPlan("alt_algorithm", "sympy.equals", lambda a, r: _val(r) and
                               _with_assumptions(a, {"a": a["expr"], "b": _val(r)}), 2)],
    "sympy.factor": [CheckPlan("symbolic_vs_numeric", "numeric.check_identity", _identity),
                     CheckPlan("alt_algorithm", "sympy.equals", lambda a, r: _val(r) and
                               _with_assumptions(a, {"a": a["expr"], "b": _val(r)}), 2)],
    "sympy.limit": [CheckPlan("symbolic_vs_numeric", "numeric.check_limit", _limit)],
    "sympy.series": [CheckPlan("symbolic_vs_numeric", "numeric.check_series", _series)],
    "sympy.subs": [CheckPlan("symbolic_vs_numeric", "numeric.check_value", _value)],
    "numeric.evaluate": [CheckPlan("alt_algorithm", "numeric.check_value", _value)],
    "numeric.quad": [CheckPlan("alt_algorithm", "numeric.check_quad", _quad)],
    "numeric.root": [CheckPlan("alt_algorithm", "numeric.check_root", _root)],
    "units.convert": [CheckPlan("units", "units.check_convert", _convert)],
}


def plans_for(tool_name: str | None, args: dict[str, Any] | None,
              result: dict[str, Any] | None) -> list[tuple[CheckPlan, dict[str, Any]]]:
    if not tool_name or args is None or result is None:
        return []
    out = []
    for plan in sorted(PLANS.get(tool_name, []), key=lambda p: p.cost):
        try:
            built = plan.build(args, result)
        except (KeyError, TypeError, ValueError):
            built = None
        if built:
            out.append((plan, built))
    return out
