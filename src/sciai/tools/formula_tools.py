"""Closed-form solvers for the recipe library, each checked by a genuinely different method.

Every tool here computes one named physical or geometric quantity from named inputs, so a
recipe is a single tool call rather than a formula the model has to assemble. That is the point:
the formula lives in code, where it can be tested, instead of in a prompt.

Each one's checker reaches the same number another way, never by re-running the formula:

- ``phys.projectile_range``: the range from the closed form; the checker integrates the motion
  numerically and reads off where the trajectory crosses the ground.
- ``phys.photon_energy``: h*c/lambda from CODATA via SciPy; the checker recomputes it from
  pint's own constant table, which is a different source.
- ``phys.rc_discharge``: V0*exp(-t/RC); the checker solves dV/dt = -V/(RC) numerically.
- ``calc.area_between``: the area between two curves by symbolic integration of |f-g| between
  consecutive crossings; the checker recomputes it by numeric quadrature on a fine grid and
  also refuses an area that is negative or zero.

Every result is a quantity (or a number for the dimensionless area), so the dimensional,
plausibility and target-unit checks apply to these exactly as they do to ``phys.evaluate``.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import sympy as sp

from sciai.domains.physics import quantities as Q
from sciai.graph.model import Domain
from sciai.tools.parsing import parse, var
from sciai.tools.phys_tools import KIND, QUANTITY, UNIT
from sciai.tools.registry import ToolSpec, register, schema
from sciai.tools.sympy_tools import BOUND, EXPR, VAR

REL_TOL = 1e-9
SIM_REL_TOL = 1e-4        # a numeric trajectory or quadrature agrees to about this
MIN_AREA = 1e-12          # an enclosed region has a positive area; this is the floor
GRID = 4001               # quadrature points for the area check
MAX_CROSSINGS = 20

G0 = 9.80665              # standard gravity, the default when a question gives no g


def _q(obj: dict[str, Any], kind: str) -> Any:
    """One quantity argument as a pint quantity, with the recipe's kind applied."""
    return Q.make(float(obj["value"]), obj["unit"], obj.get("kind") or kind)


def _result(q: Any, to_unit: str | None, kind: str) -> dict[str, Any]:
    shown = q.to(to_unit) if to_unit else q.to_base_units()
    return Q.result(shown, to_unit if to_unit else None, kind)


def _canon(*quantities: str, expr: tuple[str, ...] = ()) -> Any:
    """Canonical args for a formula tool: every quantity reduced to SI.

    Without this, 500 nm and 0.5 um are different tool calls and the second one recomputes a
    result the store already holds, verified. Reuse is supposed to be about the value, not
    about how the question happened to spell the unit.
    """
    def canonical(args: dict[str, Any]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for name in quantities:
            obj = args.get(name)
            out[name] = Q.canonical_object(obj) if obj is not None else None
        for name in expr:
            out[name] = str(args.get(name) or "")
        target = args.get("to_unit")
        out["to_unit"] = str(Q._unit(target)) if target else None
        return out
    return canonical


def _check(outcome: str, **detail: Any) -> dict[str, Any]:
    return {"result": {"kind": "check", "outcome": outcome, **detail}, "confidence": None,
            "meta": {}}


def _close(a: float, b: float, rel_tol: float) -> bool:
    return abs(a - b) <= rel_tol * max(abs(a), abs(b), 1e-300)


# ============================================================ phys.projectile_range
def projectile_range_fn(args: dict[str, Any]) -> dict[str, Any]:
    v0 = _q(args["v0"], "speed").to("m/s")
    theta = _q(args["angle"], "angle").to("radian")
    g = _q(args["g"], "acceleration").to("m/s**2") if args.get("g") else \
        Q.ureg().Quantity(G0, "m/s**2")
    if float(g.magnitude) <= 0:
        raise ValueError("gravity must be positive; a projectile needs something to fall towards")
    rng = v0 ** 2 * math.sin(2.0 * float(theta.magnitude)) / g
    return {"result": _result(rng, args.get("to_unit"), "distance"), "confidence": None,
            "meta": {"g": f"{g:~}", "formula": "v0**2*sin(2*theta)/g",
                     "default_g": not args.get("g")}}


def check_projectile_range_fn(args: dict[str, Any]) -> dict[str, Any]:
    """Simulate the motion: step the trajectory and find where it returns to the ground.

    This shares no code with the closed form. It integrates x'' = 0, y'' = -g with the fourth
    order Runge-Kutta the solver never touches, then solves for the crossing by bisection on
    the interpolated height, so a sign error or a degrees/radians slip shows up as a gross
    disagreement rather than a rounding one.
    """
    v0 = float(_q(args["v0"], "speed").to("m/s").magnitude)
    theta = float(_q(args["angle"], "angle").to("radian").magnitude)
    g = float(_q(args["g"], "acceleration").to("m/s**2").magnitude) if args.get("g") else G0
    claimed_si = float(args["si_value"])
    if not math.isfinite(claimed_si):
        return _check("fail", reason="the reported range is not a finite number")
    if math.sin(2.0 * theta) <= 0:
        # launched into the ground or straight up: the closed form is 0 or negative, and a
        # simulation has no crossing to find
        return _check("pass" if claimed_si <= 0 else "fail",
                      reason="the launch angle gives no forward range")

    vx, vy = v0 * math.cos(theta), v0 * math.sin(theta)
    flight = 2.0 * vy / g
    steps = 20_000
    h = flight * 1.5 / steps

    def deriv(state: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
        _, _, ux, uy = state
        return (ux, uy, 0.0, -g)

    def step(state: tuple[float, float, float, float], dt: float) -> tuple[float, ...]:
        k1 = deriv(state)
        s2 = tuple(s + 0.5 * dt * k for s, k in zip(state, k1))
        k2 = deriv(s2)  # type: ignore[arg-type]
        s3 = tuple(s + 0.5 * dt * k for s, k in zip(state, k2))
        k3 = deriv(s3)  # type: ignore[arg-type]
        s4 = tuple(s + dt * k for s, k in zip(state, k3))
        k4 = deriv(s4)  # type: ignore[arg-type]
        return tuple(s + dt * (a + 2 * b + 2 * c + d) / 6.0
                     for s, a, b, c, d in zip(state, k1, k2, k3, k4))

    state: tuple[float, ...] = (0.0, 0.0, vx, vy)
    simulated = None
    for _ in range(steps):
        nxt = step(state, h)  # type: ignore[arg-type]
        if nxt[1] <= 0.0 < state[1] or (state[1] == 0.0 and nxt[1] < 0.0):
            lo, hi = 0.0, h
            base = state
            for _ in range(80):  # bisect on the sub-step that crosses the ground
                mid = 0.5 * (lo + hi)
                probe = step(base, mid)  # type: ignore[arg-type]
                if probe[1] > 0.0:
                    lo = mid
                else:
                    hi = mid
            simulated = step(base, 0.5 * (lo + hi))[0]  # type: ignore[arg-type]
            break
        state = nxt
    if simulated is None:
        return _check("inconclusive", reason="the simulated trajectory never returned to the ground")
    ok = _close(simulated, claimed_si, SIM_REL_TOL)
    return _check("pass" if ok else "fail", method="numeric trajectory", simulated=simulated,
                  claimed_si=claimed_si, tolerance=SIM_REL_TOL,
                  reason="" if ok else (f"simulating the motion gives {simulated:.6g} m, the "
                                        f"result says {claimed_si:.6g} m"))


register(ToolSpec(
    name="phys.projectile_range", domain=Domain.PHYSICS, kind="solver",
    description='horizontal range of a projectile on level ground: v0={"value": 20, "unit": '
                '"m/s"}, angle={"value": 30, "unit": "deg"}, optional g (default standard '
                'gravity), optional to_unit',
    schema=schema({"v0": QUANTITY, "angle": QUANTITY, "g": QUANTITY, "to_unit": UNIT},
                  ["v0", "angle"]),
    canonical=_canon("v0", "angle", "g"),
    fn=projectile_range_fn, always_check=True,
))

register(ToolSpec(
    name="phys.check_projectile_range", domain=Domain.PHYSICS, kind="checker",
    description="integrate the projectile's motion numerically and compare the range",
    schema=schema({"v0": QUANTITY, "angle": QUANTITY, "g": QUANTITY,
                   "si_value": {"type": "number"}}, ["v0", "angle", "si_value"]),
    fn=check_projectile_range_fn,
))


# ================================================================ phys.photon_energy
def _scipy_hc() -> tuple[float, float]:
    from scipy import constants

    return float(constants.physical_constants["Planck constant"][0]), float(constants.c)


def photon_energy_fn(args: dict[str, Any]) -> dict[str, Any]:
    lam = _q(args["wavelength"], "length").to("m")
    if float(lam.magnitude) <= 0:
        raise ValueError("a wavelength must be positive")
    h, c = _scipy_hc()
    energy = Q.ureg().Quantity(h * c / float(lam.magnitude), "J")
    return {"result": _result(energy, args.get("to_unit"), "energy"), "confidence": None,
            "meta": {"formula": "h*c/wavelength", "h": h, "c": c}}


def check_photon_energy_fn(args: dict[str, Any]) -> dict[str, Any]:
    """Known value: the same energy from pint's own constant table, a different source."""
    u = Q.ureg()
    lam = _q(args["wavelength"], "length").to("m")
    h = u.Quantity(1.0, "planck_constant").to_base_units()
    c = u.Quantity(1.0, "speed_of_light").to_base_units()
    reference = (h * c / lam).to("J")
    claimed_si = float(args["si_value"])
    ref_si = float(reference.magnitude)
    ok = _close(ref_si, claimed_si, 1e-9)
    return _check("pass" if ok else "fail", method="pint constant table",
                  pint_value=f"{reference:~}", claimed_si=claimed_si,
                  reason="" if ok else (f"pint's constants give {ref_si:.6g} J, the result says "
                                        f"{claimed_si:.6g} J"))


register(ToolSpec(
    name="phys.photon_energy", domain=Domain.PHYSICS, kind="solver",
    description='photon energy from its wavelength: wavelength={"value": 500, "unit": "nm"}, '
                'optional to_unit (J, eV, ...)',
    schema=schema({"wavelength": QUANTITY, "to_unit": UNIT}, ["wavelength"]),
    canonical=_canon("wavelength"),
    fn=photon_energy_fn, always_check=True,
))

register(ToolSpec(
    name="phys.check_photon_energy", domain=Domain.PHYSICS, kind="checker",
    description="recompute a photon energy from pint's constant table",
    schema=schema({"wavelength": QUANTITY, "si_value": {"type": "number"}},
                  ["wavelength", "si_value"]),
    fn=check_photon_energy_fn,
))


# ================================================================== phys.rc_discharge
def rc_discharge_fn(args: dict[str, Any]) -> dict[str, Any]:
    v0 = _q(args["v0"], "voltage").to("V")
    r = _q(args["resistance"], "resistance").to("ohm")
    c = _q(args["capacitance"], "capacitance").to("F")
    t = _q(args["t"], "time").to("s")
    tau = float(r.magnitude) * float(c.magnitude)
    if tau <= 0:
        raise ValueError("the time constant R*C must be positive")
    value = Q.ureg().Quantity(float(v0.magnitude) * math.exp(-float(t.magnitude) / tau), "V")
    return {"result": _result(value, args.get("to_unit"), "voltage"), "confidence": None,
            "meta": {"formula": "V0*exp(-t/(R*C))", "tau_s": tau}}


def check_rc_discharge_fn(args: dict[str, Any]) -> dict[str, Any]:
    """Solve the circuit's own differential equation numerically instead of using its solution."""
    from scipy.integrate import solve_ivp

    v0 = float(_q(args["v0"], "voltage").to("V").magnitude)
    r = float(_q(args["resistance"], "resistance").to("ohm").magnitude)
    c = float(_q(args["capacitance"], "capacitance").to("F").magnitude)
    t = float(_q(args["t"], "time").to("s").magnitude)
    claimed_si = float(args["si_value"])
    tau = r * c
    if tau <= 0:
        return _check("fail", reason="the time constant R*C is not positive")
    if t == 0.0:
        ok = _close(v0, claimed_si, REL_TOL)
        return _check("pass" if ok else "fail", method="initial value", simulated=v0,
                      claimed_si=claimed_si)
    sol = solve_ivp(lambda _t, y: [-y[0] / tau], (0.0, t), [v0], method="Radau",
                    rtol=1e-11, atol=1e-14, dense_output=True)
    if not sol.success:
        return _check("inconclusive", reason=f"the numeric solve failed: {sol.message}")
    simulated = float(sol.y[0][-1])
    ok = _close(simulated, claimed_si, 1e-7)
    return _check("pass" if ok else "fail", method="numeric ODE solve", simulated=simulated,
                  claimed_si=claimed_si, tau_s=tau,
                  reason="" if ok else (f"integrating dV/dt = -V/RC gives {simulated:.6g} V, the "
                                        f"result says {claimed_si:.6g} V"))


register(ToolSpec(
    name="phys.rc_discharge", domain=Domain.PHYSICS, kind="solver",
    description='voltage on a discharging RC circuit at time t: v0, resistance, capacitance and '
                't as quantities, optional to_unit',
    schema=schema({"v0": QUANTITY, "resistance": QUANTITY, "capacitance": QUANTITY,
                   "t": QUANTITY, "to_unit": UNIT},
                  ["v0", "resistance", "capacitance", "t"]),
    canonical=_canon("v0", "resistance", "capacitance", "t"),
    fn=rc_discharge_fn, always_check=True,
))

register(ToolSpec(
    name="phys.check_rc_discharge", domain=Domain.PHYSICS, kind="checker",
    description="solve the RC circuit's differential equation numerically and compare",
    schema=schema({"v0": QUANTITY, "resistance": QUANTITY, "capacitance": QUANTITY,
                   "t": QUANTITY, "si_value": {"type": "number"}},
                  ["v0", "resistance", "capacitance", "t", "si_value"]),
    fn=check_rc_discharge_fn,
))


# =================================================================== calc.area_between
def _curves(args: dict[str, Any]) -> tuple[sp.Expr, sp.Expr, sp.Symbol]:
    a = args.get("assumptions") or {}
    x = var(args["var"], a)
    return parse(args["curve1"], a), parse(args["curve2"], a), x


def _crossings(f: sp.Expr, g: sp.Expr, x: sp.Symbol,
               lower: sp.Expr | None, upper: sp.Expr | None) -> list[float]:
    """The real x where the curves meet, inside the bounds when bounds are given."""
    if lower is not None and upper is not None:
        return [float(sp.N(lower)), float(sp.N(upper))]
    roots = sp.solve(sp.Eq(f, g), x)
    out = []
    for root in roots:
        try:
            value = complex(sp.N(root, 30))
        except (TypeError, ValueError):
            continue
        if abs(value.imag) <= 1e-12 * max(1.0, abs(value.real)):
            out.append(value.real)
    out = sorted(set(round(v, 12) for v in out))
    if len(out) < 2:
        raise ValueError("the curves do not enclose a region: they meet at fewer than two points")
    if len(out) > MAX_CROSSINGS:
        raise ValueError(f"the curves meet at more than {MAX_CROSSINGS} points")
    return out


def area_between_fn(args: dict[str, Any]) -> dict[str, Any]:
    f, g, x = _curves(args)
    a = args.get("assumptions") or {}
    lower = parse(args["lower"], a) if args.get("lower") else None
    upper = parse(args["upper"], a) if args.get("upper") else None
    points = _crossings(f, g, x, lower, upper)
    diff = f - g
    total = 0.0
    pieces = []
    for lo, hi in zip(points, points[1:]):
        piece = sp.integrate(diff, (x, sp.Float(lo, 30), sp.Float(hi, 30)))
        value = abs(float(sp.N(piece, 30)))
        pieces.append({"from": lo, "to": hi, "area": value})
        total += value
    if not math.isfinite(total):
        raise ValueError("the area is not finite")
    if total <= MIN_AREA:
        # an enclosed region has a positive area; zero means the pieces cancelled, which is
        # what happens when the absolute value is left out
        raise ValueError(f"the area between the curves came out as {total:.3g}, which is not a "
                         "positive area")
    return {"result": {"kind": "number", "value": total, "crossings": points, "pieces": pieces},
            "confidence": None,
            "meta": {"intervals": len(pieces), "integrand": "Abs(curve1 - curve2)"}}


def check_area_between_fn(args: dict[str, Any]) -> dict[str, Any]:
    """Numeric quadrature of |f-g| on a fine grid, plus the sign and nonzero guards."""
    f, g, x = _curves(args)
    claimed = float(args["area"])
    points = list(args["crossings"])
    if claimed < 0:
        return _check("fail", reason=f"an area cannot be negative, got {claimed:.6g}")
    if claimed <= MIN_AREA:
        return _check("fail", reason=f"an enclosed region has a positive area, got {claimed:.6g}")
    if len(points) < 2:
        return _check("fail", reason="fewer than two crossings were recorded")
    try:
        diff = sp.lambdify(x, sp.Abs(f - g), modules="numpy")
    except (TypeError, ValueError) as exc:
        return _check("inconclusive", reason=f"the curves could not be evaluated ({exc})")
    total = 0.0
    for lo, hi in zip(points, points[1:]):
        grid = np.linspace(float(lo), float(hi), GRID)
        try:
            values = np.abs(np.real(np.asarray(diff(grid), dtype=complex)))
        except (TypeError, ValueError, FloatingPointError) as exc:
            return _check("inconclusive", reason=f"the curves could not be evaluated ({exc})")
        if not np.all(np.isfinite(values)):
            return _check("inconclusive", reason="the curves are not finite on the interval")
        total += float(np.trapezoid(values, grid))
    ok = _close(total, claimed, SIM_REL_TOL)
    return _check("pass" if ok else "fail", method="numeric quadrature of |curve1 - curve2|",
                  quadrature=total, claimed=claimed, tolerance=SIM_REL_TOL, points=GRID,
                  reason="" if ok else (f"quadrature of |f-g| gives {total:.6g}, the result says "
                                        f"{claimed:.6g}"))


register(ToolSpec(
    name="calc.area_between", domain=Domain.MATH, kind="solver",
    description='area enclosed between two curves: curve1="x**3 - 3*x", curve2="x", var="x"; '
                'without lower and upper the curves\' own crossings are the bounds',
    schema=schema({"curve1": EXPR, "curve2": EXPR, "var": VAR, "lower": BOUND, "upper": BOUND},
                  ["curve1", "curve2", "var"]),
    fn=area_between_fn, always_check=True, expr_args=("curve1", "curve2"),
))

register(ToolSpec(
    name="calc.check_area_between", domain=Domain.MATH, kind="checker",
    description="recompute the area by numeric quadrature of |curve1 - curve2| and refuse a "
                "negative or zero area",
    schema=schema({"curve1": EXPR, "curve2": EXPR, "var": VAR, "area": {"type": "number"},
                   "crossings": {"type": "array", "items": {"type": "number"}, "minItems": 2,
                                 "maxItems": MAX_CROSSINGS}},
                  ["curve1", "curve2", "var", "area", "crossings"]),
    fn=check_area_between_fn,
))


__all__ = ["G0", "MIN_AREA", "SIM_REL_TOL", "area_between_fn", "photon_energy_fn",
           "projectile_range_fn", "rc_discharge_fn"]
