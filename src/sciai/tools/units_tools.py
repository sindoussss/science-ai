"""Unit conversion and dimension checks (pint; its parser does not use eval)."""
from __future__ import annotations

from functools import lru_cache
from typing import Any

import pint

from sciai.graph.model import Domain
from sciai.tools.parsing import number_result
from sciai.tools.registry import ToolSpec, register, schema

UNIT = {"type": "string", "maxLength": 100, "pattern": r"^[A-Za-z0-9_ */^().\-]+$"}


@lru_cache(maxsize=1)
def ureg() -> pint.UnitRegistry:
    return pint.UnitRegistry()


def convert_fn(args: dict[str, Any]) -> dict[str, Any]:
    q = ureg().Quantity(float(args["value"]), args["from_unit"])
    out = q.to(args["to_unit"])
    # The unit shown is the one the question asked for. pint's own spelling ("meter / second")
    # is kept in meta: it is right but it is not what was asked, and an answer that reads
    # "26.82 meter / second" next to a question about m/s invites the unit being written twice.
    return {"result": number_result(out.magnitude, units=str(args["to_unit"])), "confidence": None,
            "meta": {"from": str(q), "pint_unit": str(out.units)}}


register(ToolSpec(
    name="units.convert", domain=Domain.MATH, kind="solver",
    description="convert value from_unit -> to_unit",
    schema=schema({"value": {"type": "number"}, "from_unit": UNIT, "to_unit": UNIT},
                  ["value", "from_unit", "to_unit"]),
    fn=convert_fn, always_check=True,
    script=lambda a: f"import pint\nu = pint.UnitRegistry()\nprint(u.Quantity({a['value']!r}, {a['from_unit']!r}).to({a['to_unit']!r}))\n",
))


def check_convert_fn(args: dict[str, Any]) -> dict[str, Any]:
    """Independent path: both sides reduced to SI base units and compared."""
    u = ureg()
    a = u.Quantity(float(args["value"]), args["from_unit"]).to_base_units()
    b = u.Quantity(float(args["converted"]), args["to_unit"]).to_base_units()
    same_dim = a.dimensionality == b.dimensionality
    ok = same_dim and abs(a.magnitude - b.magnitude) <= 1e-9 * max(1.0, abs(a.magnitude))
    return {"result": {"kind": "check", "outcome": "pass" if ok else "fail",
                       "base_a": str(a), "base_b": str(b)}, "confidence": None, "meta": {}}


register(ToolSpec(
    name="units.check_convert", domain=Domain.MATH, kind="checker",
    description="check a unit conversion via SI base units",
    schema=schema({"value": {"type": "number"}, "from_unit": UNIT, "to_unit": UNIT,
                   "converted": {"type": "number"}}, ["value", "from_unit", "to_unit", "converted"]),
    fn=check_convert_fn,
))


def _units_equal(a: str, b: str) -> bool:
    """Same unit, not merely the same dimension. m and meter are the same; m/s and m/h are not.

    A temperature difference is stored in pint's delta form, and the quantity kind already
    distinguishes it from an absolute temperature, so the prefix is dropped before comparing.
    """
    u = ureg()
    try:
        pa = u.parse_units(str(a).replace("^", "**").replace("delta_", ""))
        pb = u.parse_units(str(b).replace("^", "**").replace("delta_", ""))
    except Exception:  # noqa: BLE001 - any unparsable unit is a failed check, not a crash
        return False
    return pa == pb


def check_target_unit_fn(args: dict[str, Any]) -> dict[str, Any]:
    """The answer is in the unit the question asked for, and in the right dimension.

    Two separate failures, and both have been seen from real models. A result whose dimension
    is wrong ("1 / m J" for an energy) is nonsense. A result whose dimension is right but whose
    unit is not the one asked for (96560.64 m/h when the question said m/s) is a correct number
    answering a different question, and it reads as the answer, so it fails the node too.
    """
    u = ureg()
    target, got = str(args["target_unit"]), str(args["unit"])
    try:
        want = u.parse_units(target.replace("^", "**"))
    except Exception as exc:  # noqa: BLE001
        return {"result": {"kind": "check", "outcome": "inconclusive",
                           "reason": f"the requested unit {target!r} could not be parsed ({exc})"},
                "confidence": None, "meta": {}}
    try:
        have = u.parse_units(got.replace("^", "**").replace("delta_", ""))
    except Exception as exc:  # noqa: BLE001
        return {"result": {"kind": "check", "outcome": "fail", "target_unit": target, "unit": got,
                           "reason": f"the result's unit {got!r} could not be parsed ({exc})"},
                "confidence": None, "meta": {}}
    want_dims = u.Quantity(1.0, want).dimensionality
    have_dims = u.Quantity(1.0, have).dimensionality
    if want_dims != have_dims:
        return {"result": {"kind": "check", "outcome": "fail", "target_unit": target, "unit": got,
                           "want_dims": str(want_dims), "got_dims": str(have_dims),
                           "reason": f"the question asked for {target} ({want_dims}) but the "
                                     f"result is in {got} ({have_dims})"},
                "confidence": None, "meta": {}}
    if not _units_equal(target, got):
        return {"result": {"kind": "check", "outcome": "fail", "target_unit": target, "unit": got,
                           "reason": f"the question asked for {target} and the result is in "
                                     f"{got}; the number answers a different question"},
                "confidence": None, "meta": {}}
    return {"result": {"kind": "check", "outcome": "pass", "target_unit": target, "unit": got,
                       "dims": str(want_dims)}, "confidence": None, "meta": {}}


register(ToolSpec(
    name="units.check_target", domain=Domain.MATH, kind="checker",
    description="check that a result is in the unit the question asked for, and in the right "
                "dimension",
    schema=schema({"target_unit": UNIT, "unit": {"type": "string", "maxLength": 100}},
                  ["target_unit", "unit"]),
    fn=check_target_unit_fn,
))


def check_dimensions_fn(args: dict[str, Any]) -> dict[str, Any]:
    u = ureg()
    q = u.parse_expression(args["quantity"])
    expected = u.parse_expression(args["expected_unit"])
    ok = getattr(q, "dimensionality", None) == getattr(expected, "dimensionality", None)
    return {"result": {"kind": "check", "outcome": "pass" if ok else "fail",
                       "got": str(getattr(q, "dimensionality", "dimensionless")),
                       "expected": str(getattr(expected, "dimensionality", "dimensionless"))},
            "confidence": None, "meta": {}}


register(ToolSpec(
    name="units.check_dimensions", domain=Domain.MATH, kind="checker",
    description="check that a quantity expression has the expected dimension",
    schema=schema({"quantity": {"type": "string", "maxLength": 300}, "expected_unit": UNIT},
                  ["quantity", "expected_unit"]),
    fn=check_dimensions_fn,
))
