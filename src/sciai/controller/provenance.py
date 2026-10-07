"""Number provenance for tool arguments (approved rule 2).

Numbers in a tool call must come from the formalized problem (the root node)
or from an ancestor node, or be small structural integers (orders, exponents).
Anything else is not rejected: the node is flagged and the verifier must check it.

A value with units ({"value": 20, "unit": "m/s"}) is sourced only by a given or an
earlier result with the same SI value and dimensions: 20 m/s does not source 20 km/h,
and its bare 20 is left out of the plain-number scan. A quantity's SI value (2 ms -> 0.002)
sources that plain number, since the ODE, linear-algebra and circuit tools take SI numbers.
"""
from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation
from typing import Any, Iterable

from sciai.domains.physics import quantities as Q

_NUM = re.compile(r"(?<![A-Za-z_\d.])(\d+\.?\d*(?:[eE][+-]?\d+)?|\.\d+(?:[eE][+-]?\d+)?)")


# No number a question or a result can hold is anywhere near this big or this small, while a
# hex id like "53827e2321134" reads as a Decimal with an exponent of two million: normalizing
# that raises Overflow, and writing it out plainly would be megabytes of digits. Caught here,
# because the scan runs over node ids and file names as well as over numbers.
MAX_EXPONENT = 1000


def _norm(s: str) -> str | None:
    try:
        d = Decimal(s)
    except InvalidOperation:
        return None
    if not d.is_finite() or (d != 0 and abs(d.adjusted()) > MAX_EXPONENT):
        return None
    try:
        return format(d.normalize(), "f")
    except (InvalidOperation, ArithmeticError, ValueError):
        return None


def numbers_in(value: Any, skip_quantities: bool = False) -> set[str]:
    out: set[str] = set()
    if skip_quantities and Q.is_quantity_object(value):
        return out
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
            out |= numbers_in(v, skip_quantities)
        return out
    if isinstance(value, (list, tuple)):
        for v in value:
            out |= numbers_in(v, skip_quantities)
    return out


# Numbers a question writes in words. "x squared" contains the 2 in "x**2" as surely as "x^2"
# does, and a slot filled from it is sourced: the test is the question's text, not its digits.
WORD_NUMBERS: dict[str, str] = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5", "six": "6",
    "seven": "7", "eight": "8", "nine": "9", "ten": "10", "eleven": "11", "twelve": "12",
    "thirteen": "13", "fourteen": "14", "fifteen": "15", "sixteen": "16", "seventeen": "17",
    "eighteen": "18", "nineteen": "19", "twenty": "20", "thirty": "30", "forty": "40",
    "fifty": "50", "sixty": "60", "seventy": "70", "eighty": "80", "ninety": "90",
    "hundred": "100", "thousand": "1000", "million": "1000000", "billion": "1000000000",
    "dozen": "12",
    # a power or a multiple is a number too, and this is how a question writes it
    "squared": "2", "square": "2", "cubed": "3", "cube": "3", "twice": "2", "double": "2",
    "doubled": "2", "half": "2", "halved": "2", "triple": "3", "tripled": "3",
    "quadruple": "4", "quarter": "4", "quartered": "4",
    "first": "1", "second": "2", "third": "3", "fourth": "4", "fifth": "5", "sixth": "6",
    "seventh": "7", "eighth": "8", "ninth": "9", "tenth": "10",
}
_WORD = re.compile(r"[A-Za-z]+")


def words_as_numbers(text: str) -> set[str]:
    """The numbers a text writes in words, normalized like ``numbers_in``'s."""
    out: set[str] = set()
    for word in _WORD.findall(text or ""):
        value = WORD_NUMBERS.get(word.lower())
        if value is not None:
            n = _norm(value)
            if n:
                out.add(n)
    return out


def numbers_asked(question: str) -> set[str]:
    """Every number the question contains, in digits or in words."""
    return numbers_in(question) | words_as_numbers(question)


def _quantity_key(value: Any) -> tuple[str, str] | None:
    """(rounded SI value, dimensions) of a quantity object or a stored quantity result."""
    try:
        if isinstance(value, dict) and value.get("kind") == "quantity" and "si_value" in value:
            return Q.si_key(float(value["si_value"])), repr(sorted(value.get("dims", {}).items()))
        if Q.is_quantity_object(value):
            q, _, _ = Q.from_object(value)
            si = Q.to_si(q)
            return Q.si_key(float(si.magnitude)), repr(sorted(Q.dims_dict(si).items()))
    except (Q.QuantityError, TypeError, ValueError):
        return None
    return None


def quantities_in(value: Any) -> list[tuple[Any, tuple[str, str] | None]]:
    """Every quantity object or quantity result inside ``value``, with its key."""
    key = _quantity_key(value)
    if key is not None or Q.is_quantity_object(value):
        return [(value, key)]
    out: list[tuple[Any, tuple[str, str] | None]] = []
    if isinstance(value, dict):
        for v in value.values():
            out += quantities_in(v)
    elif isinstance(value, (list, tuple)):
        for v in value:
            out += quantities_in(v)
    return out


def _is_structural(n: str, limit: int) -> bool:
    try:
        d = Decimal(n)
    except InvalidOperation:
        return False
    return d == d.to_integral_value() and abs(d) <= limit


def unsourced(args: dict[str, Any], sources: Iterable[Any], structural_max: int) -> list[str]:
    sources = list(sources)
    allowed: set[str] = set()
    known: set[tuple[str, str]] = set()
    for s in sources:
        allowed |= numbers_in(s)
        for _, key in quantities_in(s):
            if key is not None:
                known.add(key)
                allowed |= numbers_in(float(key[0]))  # its SI value, for tools that take plain SI numbers
    skip = {"method", "order", "points", "samples", "seed", "dir", "assumptions"}
    scanned = {k: v for k, v in args.items() if k not in skip}
    used = numbers_in(scanned, skip_quantities=True)
    missing = sorted(n for n in used if n not in allowed and not _is_structural(n, structural_max))
    for obj, key in quantities_in(scanned):
        if key is None or key not in known:
            shown = f"{obj.get('value')} {obj.get('unit')}".strip() if isinstance(obj, dict) else str(obj)
            if shown not in missing:
                missing.append(shown)
    return missing
