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
    return {"result": number_result(out.magnitude, units=str(out.units)), "confidence": None,
            "meta": {"from": str(q)}}


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
