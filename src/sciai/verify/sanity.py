"""Cheap sanity checks on a tool result: magnitude, finiteness, domain, units, physical range."""
from __future__ import annotations

import math
from typing import Any

from sciai.domains.physics import quantities as Q

_BAD_TOKENS = ("nan", "zoo", "oo")
MAX_MAGNITUDE = 1e100


def _num_issues(value: Any) -> list[str]:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return []
    if not math.isfinite(v):
        return ["result is not finite"]
    if abs(v) > MAX_MAGNITUDE:
        return [f"magnitude {v:.3g} is implausibly large"]
    return []


def sanity_issues(result: dict[str, Any] | None, args: dict[str, Any] | None = None,
                  plausibility: bool = True) -> list[str]:
    if not result:
        return []
    issues: list[str] = []
    kind = result.get("kind")
    real_vars = {k for k, v in ((args or {}).get("assumptions") or {}).items()
                 if v in ("real", "positive", "negative", "nonnegative", "nonpositive", "integer")}
    if kind == "number":
        issues += _num_issues(result.get("value"))
        if result.get("imag") and real_vars:
            issues.append("complex value where the inputs are declared real")
    elif kind == "expr":
        text = str(result.get("value", ""))
        tokens = set(text.replace("(", " ").replace(")", " ").replace("*", " ").split())
        for bad in _BAD_TOKENS:
            if bad in tokens:
                issues.append(f"result contains {bad}")
        if "numeric" in result:
            num = result["numeric"]
            if isinstance(num, list) and real_vars:
                issues.append("complex value where the inputs are declared real")
            elif not isinstance(num, list):
                issues += _num_issues(num)
    elif kind == "quantity":
        issues += _num_issues(result.get("si_value"))
        issues += Q.kind_issues(result)
        if plausibility:
            issues += Q.range_issues(result)
    elif kind == "series":
        values = [v for row in result.get("y", []) for v in row]
        if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in values):
            issues.append("series is not finite")
        elif values:
            issues += _num_issues(max(values, key=abs))
    elif kind == "list":
        for item in result.get("value", []):
            issues += sanity_issues(item, args, plausibility)
    return issues
