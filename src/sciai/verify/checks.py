"""Which independent checks exist for which solver.

Every plan uses a *different method* from the solver it checks (symbolic vs
numeric, a different algorithm, or a known-value comparison). A plan's builder
returns ``None`` when the result can't be fed to the checker (e.g. a Piecewise
the parser doesn't accept); that plan is then not offered.

``required`` plans always run (a quantity's dimensional and plausibility checks,
every circuit check); ``must_pass`` ones also block verification when they come
back inconclusive. Optional plans run cheapest-first until one passes, as in Phase 1.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

Builder = Callable[[dict[str, Any], dict[str, Any]], "dict[str, Any] | None"]


@dataclass(frozen=True)
class CheckPlan:
    method: str        # symbolic_vs_numeric | alt_algorithm | known_value | units | dimensional | ...
    checker: str       # checker tool name
    build: Builder
    cost: int = 1
    required: bool = False   # always runs, whatever else passes
    must_pass: bool = False  # inconclusive blocks verification too


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


# ------------------------------------------------------------------ physics
def _plausibility(a: dict, r: dict) -> dict | None:
    return {"result": r} if r.get("kind") == "quantity" and r.get("quantity_kind") else None


def _phys_dims(a: dict, r: dict) -> dict | None:
    if r.get("kind") != "quantity":
        return None
    out = {"expr": a["expr"], "values": a.get("values") or {}, "dims": r["dims"]}
    for k in ("to_unit", "kind"):
        if a.get(k):
            out[k] = a[k]
    return out


def _phys_value(a: dict, r: dict) -> dict | None:
    if r.get("kind") != "quantity":
        return None
    return {"expr": a["expr"], "values": a.get("values") or {}, "si_value": float(r["si_value"])}


def _constant(a: dict, r: dict) -> dict | None:
    if r.get("kind") != "quantity":
        return None
    return {"name": a["name"], "si_value": float(r["si_value"]), "dims": r["dims"]}


def _ode_solution(a: dict, r: dict) -> dict | None:
    v = _val(r)
    if v is None:
        return None
    out = {"equation": a["equation"], "func": a["func"], "var": a["var"], "solution": v}
    for k in ("ics", "t_span", "assumptions"):
        if a.get(k):
            out[k] = a[k]
    return out


def _ode_residual(a: dict, r: dict) -> dict | None:
    return _ode_solution(a, r)


def _ode_numeric(a: dict, r: dict) -> dict | None:
    out = _ode_solution(a, r)
    return out if out and a.get("ics") else None


def _ivp(a: dict, r: dict) -> dict | None:
    if r.get("kind") != "series":
        return None
    out = {k: a[k] for k in ("rhs", "funcs", "var", "y0", "t_span") if k in a}
    if a.get("points"):
        out["points"] = a["points"]
    return {**out, "y": r["y"], "rtol_used": float(r.get("rtol", 1e-8))}


def _linalg(a: dict, r: dict) -> dict | None:
    if r.get("kind") != "list":
        return None
    sol = [_val(item) for item in r["value"]]
    if any(v is None for v in sol):
        return None
    return {"A": a["A"], "b": a["b"], "solution": sol}


def _circuit(a: dict, r: dict) -> dict | None:
    if r.get("kind") != "list" or "solution" not in r:
        return None
    return {"netlist": a["netlist"], "solution": r["solution"]}


# ------------------------------------------------------------------ data (Phase 3)
# ``a["dataset"]`` is the dataset descriptor here: the verifier swaps the stored node id for
# that node's result before building plans (Verifier.offered).
def _loaded(a: dict, r: dict) -> dict | None:
    return {"loaded": r} if r.get("kind") == "dataset" else None


def _table(a: dict, r: dict) -> dict | None:
    return {"dataset": a["dataset"], "table": r} if r.get("kind") == "table" and isinstance(a.get("dataset"), dict) \
        else None


def _frame(a: dict, r: dict) -> dict | None:
    return {"result": r} if r.get("kind") == "dataset" and r.get("recipe") else None


def _stats(a: dict, r: dict) -> dict | None:
    if r.get("kind") != "stats" or not isinstance(a.get("dataset"), dict):
        return None
    return {"dataset": a["dataset"], "test_args": {k: v for k, v in a.items() if k != "dataset"}, "result": r}


def _stats_for(*tests: str) -> Builder:
    def build(a: dict, r: dict) -> dict | None:
        return _stats(a, r) if r.get("test") in tests else None
    return build


def _adjusted(a: dict, r: dict) -> dict | None:
    return {"result": r} if r.get("kind") == "adjusted" else None


_FORMULA = CheckPlan("formula", "stats.check_formula", _stats, 1, required=True, must_pass=True)
_PERMUTATION = CheckPlan("permutation", "stats.check_permutation",
                         _stats_for("welch_t", "student_t", "one_sample_t", "paired_t", "mann_whitney", "pearson",
                                    "spearman", "anova", "chi2"), 2)
_STATSMODELS = CheckPlan("statsmodels", "stats.check_statsmodels",
                         _stats_for("welch_t", "student_t", "one_sample_t", "paired_t", "anova", "ols"), 3)

PLANS: dict[str, list[CheckPlan]] = {
    "data.load": [CheckPlan("reread", "data.check_load", _loaded, 1, required=True, must_pass=True)],
    "data.describe": [CheckPlan("alt_algorithm", "data.check_describe", _table, 1)],
    "data.group": [CheckPlan("alt_algorithm", "data.check_group", _table, 1)],
    "data.filter": [CheckPlan("row_by_row", "data.check_frame", _frame, 1)],
    "data.derive": [CheckPlan("row_by_row", "data.check_frame", _frame, 1)],
    "stats.ttest": [_FORMULA, _PERMUTATION, _STATSMODELS],
    "stats.mannwhitney": [_FORMULA, _PERMUTATION],
    "stats.correlation": [_FORMULA, _PERMUTATION],
    "stats.chi2": [_FORMULA, _PERMUTATION],
    "stats.anova": [_FORMULA, _PERMUTATION, _STATSMODELS],
    "stats.regression": [_FORMULA, _STATSMODELS],
    "stats.adjust": [CheckPlan("alt_algorithm", "stats.check_adjust", _adjusted, 1, required=True, must_pass=True)],
    "phys.evaluate": [
        CheckPlan("plausibility", "phys.check_plausibility", _plausibility, 0, required=True, must_pass=True),
        CheckPlan("dimensional", "phys.check_dimensions", _phys_dims, 1, required=True, must_pass=True),
        CheckPlan("alt_algorithm", "phys.check_value", _phys_value, 2),
    ],
    "phys.constant": [
        CheckPlan("plausibility", "phys.check_plausibility", _plausibility, 0, required=True, must_pass=True),
        CheckPlan("known_value", "phys.check_constant", _constant, 1, required=True, must_pass=True),
    ],
    "ode.dsolve": [
        CheckPlan("residual", "ode.check_residual", _ode_residual, 2, required=True),
        CheckPlan("symbolic_vs_numeric", "ode.check_numeric", _ode_numeric, 3, required=True),
    ],
    "ode.solve_ivp": [
        CheckPlan("alt_algorithm", "ode.check_ivp", _ivp, 2, required=True),
        CheckPlan("symbolic_vs_numeric", "ode.check_ivp_symbolic", _ivp, 3, required=True),
    ],
    "linalg.solve": [CheckPlan("residual", "linalg.check_residual", _linalg)],
    "circuit.dc": [
        CheckPlan("kirchhoff", "circuit.check_kirchhoff", _circuit, 1, required=True, must_pass=True),
        CheckPlan("power_balance", "circuit.check_power", _circuit, 1, required=True, must_pass=True),
        CheckPlan("series_parallel", "circuit.check_reduction", _circuit, 2, required=True),
    ],
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
