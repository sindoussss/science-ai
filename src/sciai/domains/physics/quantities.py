"""Physical quantities: pint <-> the stored ``quantity`` result, SI form, and quantity kinds.

A stored quantity keeps the value and unit as shown plus its SI value, SI unit and
dimensions, so checks and reuse compare in SI. Its optional ``quantity_kind`` (the
result's own ``kind`` key already says "quantity") comes from the fixed list in
``KINDS``: each kind has a required dimension and, for some, a physical range.

Temperatures are stored in kelvin. The kind decides the conversion of an offset
unit: an absolute 10 degC is 283.15 K, a 10 degC difference is 10 K. An offset unit
with no kind is refused, because guessing is exactly the classic offset bug.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import pint

SIG_DIGITS = 9  # SI values are rounded to this many significant digits for fingerprints
UNIT_RE = re.compile(r"^[A-Za-z0-9_ */^().\-]{1,100}$")
SPEED_OF_LIGHT = 299_792_458.0  # m/s, exact by definition of the metre


@lru_cache(maxsize=1)
def ureg() -> pint.UnitRegistry:
    return pint.UnitRegistry()


class QuantityError(ValueError):
    """A quantity that can't be accepted; the message is written for the model's retry."""


@dataclass(frozen=True)
class Kind:
    dims: str                              # pint dimensionality expression, "" for dimensionless
    low: float | None = None               # physical range in SI units, inclusive
    high: float | None = None
    magnitude: bool = False                # the range applies to the absolute value

    def dimensionality(self) -> Any:
        u = ureg()
        return u.dimensionless.dimensionality if not self.dims else u.get_dimensionality(self.dims)


KINDS: dict[str, Kind] = {
    "length": Kind("[length]"),
    "distance": Kind("[length]", low=0.0),
    "area": Kind("[length]**2", low=0.0),
    "volume": Kind("[length]**3", low=0.0),
    "mass": Kind("[mass]", low=0.0),
    "time": Kind("[time]"),
    "speed": Kind("[length]/[time]", low=0.0, high=SPEED_OF_LIGHT),
    "velocity": Kind("[length]/[time]", low=-SPEED_OF_LIGHT, high=SPEED_OF_LIGHT),
    "acceleration": Kind("[length]/[time]**2"),
    "force": Kind("[mass]*[length]/[time]**2"),
    "momentum": Kind("[mass]*[length]/[time]"),
    "energy": Kind("[mass]*[length]**2/[time]**2"),
    "power": Kind("[mass]*[length]**2/[time]**3"),
    "pressure": Kind("[mass]/[length]/[time]**2"),
    "density": Kind("[mass]/[length]**3", low=0.0),
    "frequency": Kind("1/[time]", low=0.0),
    "angle": Kind(""),
    "efficiency": Kind("", low=0.0, high=1.0),
    "ratio": Kind(""),
    "absolute_temperature": Kind("[temperature]", low=0.0),
    "temperature_difference": Kind("[temperature]"),
    "amount": Kind("[substance]", low=0.0),
    "charge": Kind("[current]*[time]"),
    "current": Kind("[current]"),
    "voltage": Kind("[mass]*[length]**2/[current]/[time]**3"),
    "resistance": Kind("[mass]*[length]**2/[current]**2/[time]**3", low=0.0),
    "capacitance": Kind("[current]**2*[time]**4/[mass]/[length]**2", low=0.0),
    "inductance": Kind("[mass]*[length]**2/[current]**2/[time]**2", low=0.0),
}


def _unit(text: str) -> pint.Unit:
    """Unit text -> pint unit. "" and "1" mean dimensionless (an efficiency, a ratio)."""
    if isinstance(text, str) and text.strip() in ("", "1"):
        return ureg().dimensionless
    if not isinstance(text, str) or not UNIT_RE.match(text):
        raise QuantityError(f"unit {text!r} is not allowed; use plain unit names like m/s, kg, degC")
    try:
        return ureg().parse_units(text.replace("^", "**"))
    except (pint.errors.UndefinedUnitError, pint.errors.DefinitionSyntaxError, AttributeError,
            TypeError, ValueError, AssertionError) as exc:
        raise QuantityError(f"unknown unit {text!r} ({exc})") from None


def is_offset(unit: pint.Unit) -> bool:
    """degC and degF have an offset; multiplying them is ambiguous (pint refuses it)."""
    q = ureg().Quantity(1.0, unit)
    try:
        q * q
    except pint.errors.OffsetUnitCalculusError:
        return True
    return False


def is_angle_unit(unit: pint.Unit) -> bool:
    """A named angle unit (rad, deg, ...), as opposed to a bare dimensionless number."""
    return str(unit) in _ANGLE_UNITS


_ANGLE_UNITS = {"radian", "degree", "arcminute", "arcsecond", "turn", "gradian", "milliradian",
                "microradian"}


def dims_dict(q: pint.Quantity) -> dict[str, float]:
    out = {}
    for k, v in sorted(dict(q.dimensionality).items()):
        out[k] = int(v) if float(v).is_integer() else float(v)
    return out


def make(value: float, unit: str, kind: str | None = None) -> pint.Quantity:
    """A pint quantity from a value, unit text and optional kind, with the kind rules applied."""
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise QuantityError(f"quantity value must be a finite number, got {value!r}")
    if kind is not None and kind not in KINDS:
        raise QuantityError(f"unknown kind {kind!r}; use one of: {', '.join(sorted(KINDS))}")
    unit_obj = _unit(unit)
    u = ureg()
    if is_offset(unit_obj):
        if kind not in ("absolute_temperature", "temperature_difference"):
            raise QuantityError(f"a temperature in {unit} needs kind absolute_temperature or "
                                f"temperature_difference")
        if kind == "temperature_difference":
            unit_obj = u.parse_units(f"delta_{unit_obj}")
    q = u.Quantity(float(value), unit_obj)
    if kind is not None:
        want = KINDS[kind].dimensionality()
        if q.dimensionality != want:
            raise QuantityError(f"{value} {unit} has dimension {q.dimensionality} but kind {kind} needs {want}")
    return q


def from_object(obj: Any) -> tuple[pint.Quantity, str, str | None]:
    """Parse ``{"value": .., "unit": .., "kind"?: ..}`` -> (quantity, unit text, kind)."""
    if not isinstance(obj, dict) or "value" not in obj or "unit" not in obj:
        raise QuantityError(f'a quantity is {{"value": number, "unit": "m/s", "kind": optional}}, got {obj!r}')
    extra = set(obj) - {"value", "unit", "kind"}
    if extra:
        raise QuantityError(f"unexpected quantity field(s): {', '.join(sorted(extra))}")
    return make(obj["value"], obj["unit"], obj.get("kind")), obj["unit"], obj.get("kind")


def to_si(q: pint.Quantity) -> pint.Quantity:
    return q.to_base_units()


def si_key(si_value: float) -> str:
    """The SI value rounded to SIG_DIGITS significant digits, as a canonical string.

    60 mph -> 26.8224 m/s and 96.56064 km/h -> 26.822400000000002 m/s both give 2.68224000e+01."""
    if not math.isfinite(si_value):
        raise QuantityError("quantity is not finite")
    key = f"{si_value:.{SIG_DIGITS - 1}e}"
    return key[1:] if key.startswith("-") and float(key) == 0 else key


def is_quantity_object(obj: Any) -> bool:
    return isinstance(obj, dict) and "value" in obj and "unit" in obj and set(obj) <= {"value", "unit", "kind"}


def result(q: pint.Quantity, unit_text: str | None = None, kind: str | None = None) -> dict[str, Any]:
    """The stored ``quantity`` result for ``q``, shown in ``unit_text`` (or q's own unit)."""
    si = to_si(q)
    si_value = float(si.magnitude)
    if not math.isfinite(si_value):
        raise QuantityError("quantity is not finite")
    out: dict[str, Any] = {
        "kind": "quantity",
        "value": float(q.magnitude),
        "unit": unit_text if unit_text is not None else f"{q.units:~}",
        "si_value": si_value,
        "si_unit": str(si.units),
        "dims": dims_dict(si),
    }
    if kind is not None:
        out["quantity_kind"] = kind
    return out


def fingerprint_payload(si_value: float, dims: dict[str, float], kind: str | None) -> dict[str, Any]:
    return {"si": si_key(si_value), "dims": sorted(dims.items()), "kind": kind}


def canonical_object(obj: Any) -> dict[str, Any]:
    """Canonical form of a quantity argument, for tool fingerprints."""
    q, _, kind = from_object(obj)
    si = to_si(q)
    return {"quantity": fingerprint_payload(float(si.magnitude), dims_dict(si), kind)}


def range_issues(res: dict[str, Any]) -> list[str]:
    """Plausibility: the value against its kind's physical range (SI)."""
    kind = res.get("quantity_kind")
    if res.get("kind") != "quantity" or kind not in KINDS:
        return []
    rule = KINDS[kind]
    v = float(res["si_value"])
    x = abs(v) if rule.magnitude else v
    shown = f"{res['value']:.6g} {res['unit']}"
    if rule.low is not None and x < rule.low - 1e-12 * max(1.0, abs(rule.low)):
        return [f"{kind} {shown} is below its physical minimum ({rule.low:g} {res['si_unit']})"]
    if rule.high is not None and x > rule.high + 1e-12 * max(1.0, abs(rule.high)):
        return [f"{kind} {shown} is above its physical maximum ({rule.high:g} {res['si_unit']})"]
    return []


def kind_issues(res: dict[str, Any]) -> list[str]:
    """The stored dimensions against the kind's required dimension."""
    kind = res.get("quantity_kind")
    if res.get("kind") != "quantity" or kind is None:
        return []
    if kind not in KINDS:
        return [f"unknown kind {kind!r}"]
    want = {k: (int(v) if float(v).is_integer() else float(v))
            for k, v in sorted(dict(KINDS[kind].dimensionality()).items())}
    return [] if want == res.get("dims") else [f"dimensions {res.get('dims')} don't match kind {kind} ({want})"]
