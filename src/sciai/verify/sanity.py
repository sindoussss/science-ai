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
    elif kind == "stats":
        issues += _stats_issues(result)
    elif kind == "adjusted":
        if not all(isinstance(p, (int, float)) and 0 <= p <= 1 for p in result.get("p_adjusted", [])):
            issues.append("an adjusted p-value is outside [0, 1]")
    elif kind == "dataset":
        parent = result.get("parent_rows")
        if parent and result.get("rows", 0) < 0.5 * parent:
            issues.append(f"the filter keeps {result['rows']} of {parent} rows (under half)")
    elif kind == "table":
        for row in result.get("rows", []):
            if len(row) >= 3 and result.get("columns", [None, None, None])[1:3] == ["n", "missing"]:
                n, miss = row[1] or 0, row[2] or 0
                if n + miss and miss > 0.2 * (n + miss):
                    issues.append(f"column {row[0]} is {round(100 * miss / (n + miss))}% missing")
    elif kind == "list":
        for item in result.get("value", []):
            issues += sanity_issues(item, args, plausibility)
    return issues


def _stats_issues(r: dict[str, Any]) -> list[str]:
    issues = []
    p = r.get("p")
    if not (isinstance(p, (int, float)) and 0 <= p <= 1):
        issues.append("p-value is outside [0, 1]")
    stat = (r.get("statistic") or {}).get("value")
    if not (isinstance(stat, (int, float)) and math.isfinite(stat)):
        issues.append("test statistic is not finite")
    effect = r.get("effect") or {}
    if r.get("test") in ("pearson", "spearman") or "biserial" in str(effect.get("name", "")):
        if not -1 <= float(effect.get("value", 0)) <= 1:
            issues.append("correlation is outside [-1, 1]")
    for name in ("r2",):
        if name in r and not 0 <= float(r[name]) <= 1 + 1e-12:
            issues.append("R^2 is outside [0, 1]")
    ci = effect.get("ci")
    if ci and not (ci[0] <= effect.get("value", ci[0]) <= ci[1] or "bootstrap" in str(effect.get("ci_method"))):
        issues.append("the effect lies outside its own confidence interval")
    return issues
