"""Physics solvers and checkers: constants, formula evaluation with units, and the
dimensional, value, known-value and plausibility checks.

``phys.evaluate`` works in SI throughout. Every input quantity is converted to SI
base units before it is substituted (so an angle in degrees arrives in radians),
and the formula is evaluated either with pint doing the unit arithmetic
(``default``) or with plain SI numbers plus a separate dimension pass (``si_first``),
so a retry can take a genuinely different path.
"""
from __future__ import annotations

import json
import math
from typing import Any

import numpy as np
import pint
import sympy as sp

from sciai.domains.physics import quantities as Q
from sciai.domains.physics.constants import CONSTANTS, codata_release, scipy_version
from sciai.graph.model import Domain
from sciai.tools.parsing import canonical, parse, var
from sciai.tools.registry import ToolSpec, register, schema
from sciai.tools.sympy_tools import EXPR

DPS = 30
REL_TOL = 1e-9
EXACT_REL_TOL = 1e-12   # exact SI constants: scipy and pint agree to float precision
CONSTANT_REL_TOL = 1e-6  # measured constants: allows a pint on an older CODATA release

QUANTITY = {
    "type": "object",
    "properties": {"value": {"type": "number"}, "unit": {"type": "string", "maxLength": 100},
                   "kind": {"enum": sorted(Q.KINDS)}},
    "required": ["value", "unit"],
    "additionalProperties": False,
}
VALUE = {"anyOf": [QUANTITY, {"type": "number"}]}
VALUES = {"type": "object", "additionalProperties": VALUE, "maxProperties": 30}
UNIT = {"type": "string", "maxLength": 100, "pattern": Q.UNIT_RE.pattern}
KIND = {"enum": sorted(Q.KINDS)}
TRIG = (sp.sin, sp.cos, sp.tan, sp.cot, sp.sec, sp.csc)
BASE_UNITS = {"[length]": "meter", "[mass]": "kilogram", "[time]": "second", "[current]": "ampere",
              "[temperature]": "kelvin", "[substance]": "mole", "[luminosity]": "candela"}


class DimensionError(ValueError):
    pass


def _check(outcome: str, **detail: Any) -> dict[str, Any]:
    return {"result": {"kind": "check", "outcome": outcome, **detail}, "confidence": None, "meta": {}}


# ----------------------------------------------------------------- inputs
def _inputs(args: dict[str, Any]) -> tuple[sp.Expr, dict[sp.Symbol, pint.Quantity], set[sp.Symbol]]:
    """(expression, SI quantity per symbol, symbols given as plain numbers)."""
    u = Q.ureg()
    quantities: dict[sp.Symbol, pint.Quantity] = {}
    plain: set[sp.Symbol] = set()
    for name, value in (args.get("values") or {}).items():
        s = var(name)
        if isinstance(value, dict):
            q, _, _ = Q.from_object(value)
        elif isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
            q = u.Quantity(float(value), u.dimensionless)
            plain.add(s)
        else:
            raise Q.QuantityError(f"value of {name} must be a number or a quantity object")
        quantities[s] = Q.to_si(q)
    expr = parse(args["expr"])
    missing = sorted(str(s) for s in expr.free_symbols if s not in quantities)
    if missing:
        raise ValueError(f"no value given for {', '.join(missing)}")
    _check_trig(expr, plain)
    return expr, quantities, plain


def _check_trig(expr: sp.Expr, plain: set[sp.Symbol]) -> None:
    """SymPy reads sin(30) as 30 radians. Angles must arrive as quantities (converted to rad)."""
    for node in sp.preorder_traversal(expr):
        if not isinstance(node, TRIG):
            continue
        arg = node.args[0]
        name = type(node).__name__
        if not arg.free_symbols:
            ratio = sp.simplify(arg / sp.pi)
            if not (ratio.is_Rational or ratio.is_Float):
                raise ValueError(f"{name}({arg}) would read {arg} as radians; give the angle as a quantity, "
                                 f'e.g. {{"value": 30, "unit": "deg"}}, or write a multiple of pi')
        bare = sorted(str(s) for s in arg.free_symbols & plain)
        if bare:
            raise ValueError(f"{', '.join(bare)} inside {name}() is a plain number; give it as a quantity with an "
                             f'angle unit, e.g. {{"value": 30, "unit": "deg"}}')


def _target(args: dict[str, Any]) -> pint.Unit | None:
    if not args.get("to_unit"):
        return None
    unit = Q._unit(args["to_unit"])
    if Q.is_offset(unit):
        kind = args.get("kind")
        if kind not in ("absolute_temperature", "temperature_difference"):
            raise Q.QuantityError(f"to_unit {args['to_unit']} needs kind absolute_temperature or "
                                  f"temperature_difference")
        if kind == "temperature_difference":
            unit = Q.ureg().parse_units(f"delta_{unit}")
    return unit


# -------------------------------------------------------------- dimensions
def _norm_dims(d: dict[str, float]) -> dict[str, float]:
    out = {}
    for k, v in sorted(d.items()):
        if abs(v) > 1e-12:
            out[k] = int(round(v)) if abs(v - round(v)) < 1e-12 else float(v)
    return out


def dimension_of(e: sp.Basic, dims: dict[sp.Symbol, dict[str, float]]) -> dict[str, float]:
    """Dimensions of an expression by propagation over the SymPy tree (no pint arithmetic)."""
    if e.is_Symbol:
        return dict(dims[e])
    if e.is_number:
        return {}
    if isinstance(e, sp.Add):
        terms = [_norm_dims(dimension_of(a, dims)) for a in e.args]
        for t in terms[1:]:
            if t != terms[0]:
                raise DimensionError(f"adds {terms[0] or 'dimensionless'} to {t or 'dimensionless'}")
        return terms[0]
    if isinstance(e, sp.Mul):
        out: dict[str, float] = {}
        for a in e.args:
            for k, v in dimension_of(a, dims).items():
                out[k] = out.get(k, 0) + v
        return _norm_dims(out)
    if isinstance(e, sp.Pow):
        base = _norm_dims(dimension_of(e.base, dims))
        if _norm_dims(dimension_of(e.exp, dims)):
            raise DimensionError("an exponent must be dimensionless")
        if not base:
            return {}
        if e.exp.free_symbols or not e.exp.is_number:
            raise DimensionError("a quantity with units can only be raised to a fixed number")
        p = float(e.exp)
        return _norm_dims({k: v * p for k, v in base.items()})
    if isinstance(e, sp.Abs):
        return dimension_of(e.args[0], dims)
    if isinstance(e, (sp.Min, sp.Max)):
        terms = [_norm_dims(dimension_of(a, dims)) for a in e.args]
        if any(t != terms[0] for t in terms):
            raise DimensionError(f"{type(e).__name__} of quantities with different dimensions")
        return terms[0]
    if isinstance(e, sp.Function):
        for a in e.args:
            d = _norm_dims(dimension_of(a, dims))
            if d:
                raise DimensionError(f"{type(e).__name__}() needs a dimensionless argument, got {d}")
        return {}
    raise DimensionError(f"cannot work out the dimensions of {type(e).__name__}")


def _units_from_dims(d: dict[str, float]) -> pint.Unit:
    u = Q.ureg()
    unit = u.dimensionless
    for k, v in d.items():
        unit = unit * u.parse_units(BASE_UNITS[k]) ** v
    return unit


# ---------------------------------------------------------------- evaluate
def _evaluate_pint(expr: sp.Expr, qs: dict[sp.Symbol, pint.Quantity]) -> pint.Quantity:
    syms = list(qs)
    f = sp.lambdify(syms, expr, modules="numpy")
    try:
        with np.errstate(all="ignore"):
            out = f(*[qs[s] for s in syms])
    except pint.errors.DimensionalityError as exc:
        raise DimensionError(f"formula is not dimensionally consistent: {exc}") from None
    if not isinstance(out, pint.Quantity):
        out = Q.ureg().Quantity(out, Q.ureg().dimensionless)
    mag = complex(np.asarray(out.magnitude).item())
    if abs(mag.imag) > 1e-12 * max(1.0, abs(mag.real)):
        raise ValueError("the result is complex")
    return Q.ureg().Quantity(mag.real, out.units)


def _evaluate_si_first(expr: sp.Expr, qs: dict[sp.Symbol, pint.Quantity]) -> pint.Quantity:
    subs = {s: sp.Float(repr(float(q.magnitude)), DPS) for s, q in qs.items()}
    val = complex(sp.N(expr.subs(subs), DPS))
    if abs(val.imag) > 1e-12 * max(1.0, abs(val.real)):
        raise ValueError("the result is complex")
    dims = dimension_of(expr, {s: Q.dims_dict(q) for s, q in qs.items()})
    return Q.ureg().Quantity(val.real, _units_from_dims(dims))


def evaluate_fn(args: dict[str, Any]) -> dict[str, Any]:
    expr, qs, _ = _inputs(args)
    method = args.get("method", "default")
    out = _evaluate_si_first(expr, qs) if method == "si_first" else _evaluate_pint(expr, qs)
    if not math.isfinite(float(out.magnitude)):
        raise ValueError("the result is not finite")
    target = _target(args)
    if target is not None:
        try:
            out = out.to(target)
        except pint.errors.DimensionalityError:
            raise DimensionError(f"the result has dimension {out.dimensionality} but to_unit {args['to_unit']} "
                                 f"has {Q.ureg().Quantity(1, target).dimensionality}") from None
        shown = args["to_unit"]
    else:
        out = out.to_base_units()
        shown = f"{out.units:~}" if out.units != Q.ureg().dimensionless else ""
    kind = args.get("kind")
    res = Q.result(out, shown, kind)
    issues = Q.kind_issues(res)
    if issues:
        raise Q.QuantityError(issues[0])
    return {"result": res, "confidence": None,
            "meta": {"method": method, "values_si": {str(s): f"{q:~}" for s, q in qs.items()}}}


def _canon_values(values: dict[str, Any]) -> dict[str, Any]:
    out = {}
    for name, v in sorted((values or {}).items()):
        out[name] = Q.canonical_object(v) if isinstance(v, dict) else {"number": Q.si_key(float(v))}
    return out


def _canon_evaluate(args: dict[str, Any]) -> dict[str, Any]:
    target = _target(args)
    return {"expr": canonical(parse(args["expr"])), "values": _canon_values(args.get("values") or {}),
            "to_unit": None if target is None else str(target), "kind": args.get("kind"),
            "method": args.get("method", "default")}


def _evaluate_script(a: dict[str, Any]) -> str:
    vals = json.dumps(a.get("values") or {})
    last = f"print(r.to({a['to_unit']!r}))\n" if a.get("to_unit") else "print(r)\n"
    return ("import json, pint, sympy as sp\nu = pint.UnitRegistry()\n"
            f"values = json.loads({vals!r})\n"
            "q = {k: (u.Quantity(v['value'], v['unit']) if isinstance(v, dict) else v) for k, v in values.items()}\n"
            "q = {k: (v.to_base_units() if hasattr(v, 'to_base_units') else v) for k, v in q.items()}\n"
            f"f = sp.lambdify([sp.Symbol(k) for k in q], sp.sympify({a['expr']!r}), 'numpy')\n"
            "r = f(*q.values())\n" + last)


register(ToolSpec(
    name="phys.evaluate", domain=Domain.PHYSICS, kind="solver",
    description='evaluate a formula with units; values={"v": {"value": 20, "unit": "m/s"}, '
                '"theta": {"value": 30, "unit": "deg"}}; optional to_unit and kind',
    schema=schema({"expr": EXPR, "values": VALUES, "to_unit": UNIT, "kind": KIND,
                   "method": {"enum": ["default", "si_first"]}}, ["expr", "values"]),
    fn=evaluate_fn, expr_args=("expr",), methods=("default", "si_first"),
    canonical=_canon_evaluate, script=_evaluate_script,
))


# ---------------------------------------------------------------- constants
def _constant_quantity(name: str) -> tuple[pint.Quantity, str, float]:
    from scipy import constants

    key = CONSTANTS[name][0]
    value, unit, uncertainty = constants.physical_constants[key]
    return Q.ureg().Quantity(value, Q.ureg().parse_units(unit.replace("^", "**"))), unit, uncertainty


def constant_fn(args: dict[str, Any]) -> dict[str, Any]:
    name = args["name"]
    q, unit, uncertainty = _constant_quantity(name)
    res = Q.result(q, unit, CONSTANTS[name][2])
    res["source"] = f"CODATA {codata_release()} via SciPy {scipy_version()}"
    return {"result": res, "confidence": None,
            "meta": {"codata": codata_release(), "scipy": scipy_version(), "key": CONSTANTS[name][0],
                     "uncertainty": uncertainty, "exact": uncertainty == 0}}


register(ToolSpec(
    name="phys.constant", domain=Domain.PHYSICS, kind="solver",
    description=f"a physical constant (CODATA, offline): name in {', '.join(CONSTANTS)}",
    schema=schema({"name": {"enum": list(CONSTANTS)}}, ["name"]),
    fn=constant_fn, canonical=lambda a: {"name": a["name"]},
    script=lambda a: f"from scipy import constants\nprint(constants.physical_constants[{CONSTANTS[a['name']][0]!r}])\n",
))


def check_constant_fn(args: dict[str, Any]) -> dict[str, Any]:
    """Known value: the same constant from pint's registry, an independent table."""
    from scipy import constants

    name = args["name"]
    key, pint_name, _ = CONSTANTS[name]
    ref = Q.ureg().Quantity(1.0, pint_name).to_base_units()
    exact = constants.physical_constants[key][2] == 0
    tol = EXACT_REL_TOL if exact else CONSTANT_REL_TOL
    claimed = float(args["si_value"])
    dims_ok = Q.dims_dict(ref) == args["dims"]
    rel = abs(claimed - ref.magnitude) / max(abs(ref.magnitude), 1e-300)
    ok = dims_ok and rel <= tol
    return _check("pass" if ok else "fail", pint_value=f"{ref:~}", claimed_si=claimed, rel_diff=rel, tolerance=tol,
                  exact=exact, dims_match=dims_ok, codata=codata_release(), scipy=scipy_version(),
                  pint=pint.__version__)


register(ToolSpec(
    name="phys.check_constant", domain=Domain.PHYSICS, kind="checker",
    description="compare a constant with pint's registry value",
    schema=schema({"name": {"enum": list(CONSTANTS)}, "si_value": {"type": "number"},
                   "dims": {"type": "object"}}, ["name", "si_value", "dims"]),
    fn=check_constant_fn,
))


# ----------------------------------------------------------------- checkers
def check_dimensions_fn(args: dict[str, Any]) -> dict[str, Any]:
    """Dimensional analysis by propagation over the formula, compared with the stored result,
    the requested unit and the kind."""
    expr, qs, _ = _inputs(args)
    try:
        derived = _norm_dims(dimension_of(expr, {s: Q.dims_dict(q) for s, q in qs.items()}))
    except DimensionError as exc:
        return _check("fail", reason=f"formula is not dimensionally consistent: {exc}")
    claimed = _norm_dims(args["dims"])
    problems = []
    if derived != claimed:
        problems.append(f"formula gives {derived or 'dimensionless'}, result says {claimed or 'dimensionless'}")
    target = _target(args)
    if target is not None:
        want = _norm_dims(Q.dims_dict(Q.ureg().Quantity(1.0, target)))
        if want != derived:
            problems.append(f"to_unit {args['to_unit']} is {want or 'dimensionless'}")
    if args.get("kind"):
        want = _norm_dims({k: float(v) for k, v in dict(Q.KINDS[args["kind"]].dimensionality()).items()})
        if want != derived:
            problems.append(f"kind {args['kind']} needs {want or 'dimensionless'}")
    if problems:
        return _check("fail", derived=derived, reason="; ".join(problems))
    return _check("pass", derived=derived)


register(ToolSpec(
    name="phys.check_dimensions", domain=Domain.PHYSICS, kind="checker",
    description="dimensional analysis of a formula against the result's dimensions",
    schema=schema({"expr": EXPR, "values": VALUES, "dims": {"type": "object"}, "to_unit": UNIT, "kind": KIND},
                  ["expr", "values", "dims"]),
    fn=check_dimensions_fn,
))


def check_value_fn(args: dict[str, Any]) -> dict[str, Any]:
    """Re-evaluation in SI with mpmath at 30 digits; pint only converts the inputs to SI."""
    expr, qs, _ = _inputs(args)
    subs = {s: sp.Float(repr(float(q.magnitude)), DPS) for s, q in qs.items()}
    try:
        ref = complex(sp.N(expr.subs(subs), DPS))
    except (TypeError, ValueError) as exc:
        return _check("inconclusive", reason=str(exc))
    claimed = float(args["si_value"])
    scale = max(abs(ref), abs(claimed))
    ok = abs(ref - claimed) <= REL_TOL * scale + 1e-300
    return _check("pass" if ok else "fail", reference_si=str(ref.real if not ref.imag else ref), claimed_si=claimed)


register(ToolSpec(
    name="phys.check_value", domain=Domain.PHYSICS, kind="checker",
    description="re-evaluate a formula in SI at 30 digits and compare",
    schema=schema({"expr": EXPR, "values": VALUES, "si_value": {"type": "number"}},
                  ["expr", "values", "si_value"]),
    fn=check_value_fn,
))


def check_plausibility_fn(args: dict[str, Any]) -> dict[str, Any]:
    issues = Q.kind_issues(args["result"]) + Q.range_issues(args["result"])
    return _check("fail" if issues else "pass", reason="; ".join(issues) if issues else "within range",
                  quantity_kind=args["result"].get("quantity_kind"))


register(ToolSpec(
    name="phys.check_plausibility", domain=Domain.PHYSICS, kind="checker",
    description="a quantity against its kind's physical range",
    schema=schema({"result": {"type": "object"}}, ["result"]),
    fn=check_plausibility_fn,
))
