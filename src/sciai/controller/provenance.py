"""Number provenance for tool arguments (approved rule 2).

Numbers in a tool call must come from the formalized problem (the root node)
or from an ancestor node, or be small structural integers (orders, exponents).
Anything else is not rejected: the node is flagged and the verifier must check it.
"""
from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation
from typing import Any, Iterable

_NUM = re.compile(r"(?<![A-Za-z_\d.])(\d+\.?\d*(?:[eE][+-]?\d+)?|\.\d+(?:[eE][+-]?\d+)?)")


def _norm(s: str) -> str | None:
    try:
        d = Decimal(s)
    except InvalidOperation:
        return None
    return format(d.normalize(), "f")


def numbers_in(value: Any) -> set[str]:
    out: set[str] = set()
    if isinstance(value, bool) or value is None:
        return out
    if isinstance(value, (int, float)):
        n = _norm(repr(value))
        return {n} if n else out
    if isinstance(value, str):
        for m in _NUM.findall(value):
            n = _norm(m)
            if n:
                out.add(n)
        return out
    if isinstance(value, dict):
        for v in value.values():
            out |= numbers_in(v)
        return out
    if isinstance(value, (list, tuple)):
        for v in value:
            out |= numbers_in(v)
    return out


def _is_structural(n: str, limit: int) -> bool:
    try:
        d = Decimal(n)
    except InvalidOperation:
        return False
    return d == d.to_integral_value() and abs(d) <= limit


def unsourced(args: dict[str, Any], sources: Iterable[Any], structural_max: int) -> list[str]:
    allowed: set[str] = set()
    for s in sources:
        allowed |= numbers_in(s)
    skip = {"method", "order", "points", "samples", "seed", "dir", "assumptions"}
    used = numbers_in({k: v for k, v in args.items() if k not in skip})
    return sorted(n for n in used if n not in allowed and not _is_structural(n, structural_max))
