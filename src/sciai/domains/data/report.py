"""Text for data results (datasets, tables, test results). Pure Python: the graph layer uses it.

A test result always reads as the statistic, the p-value with the alpha it was judged at,
and the effect size with its 95% confidence interval, so an answer that cites a test
carries all three (the Phase 3 reporting rule).
"""
from __future__ import annotations

import math
from typing import Any

MAX_TABLE_ROWS = 8


def num(x: Any, digits: int = 4) -> str:
    if x is None:
        return "n/a"
    if isinstance(x, bool):
        return str(x)
    if isinstance(x, int):
        return str(x)
    try:
        v = float(x)
    except (TypeError, ValueError):
        return str(x)
    if math.isnan(v):
        return "n/a"
    if math.isinf(v):
        return "inf" if v > 0 else "-inf"
    if v == 0:
        return "0"
    if abs(v) < 1e-4 or abs(v) >= 1e6:
        return f"{v:.{digits - 1}e}"
    if abs(v) >= 10 ** digits:  # 19990.3, not 1.999e+04
        return f"{v:.1f}".removesuffix(".0")
    return f"{v:.{digits}g}"


def p_text(p: Any) -> str:
    """Reads "p = 0.0123", or "p < 1e-16" below double precision's resolution."""
    try:
        v = float(p)
    except (TypeError, ValueError):
        return "p n/a"
    return "p < 1e-16" if v < 1e-16 else f"p = {num(v, 3)}"


def ci_text(ci: Any) -> str:
    if not ci or len(ci) != 2:
        return ""
    return f"95% CI [{num(ci[0])}, {num(ci[1])}]"


def dataset_text(r: dict[str, Any]) -> str:
    cols = r.get("columns") or []
    text = f"{r.get('name', 'dataset')}: {r.get('rows')} rows x {len(cols)} columns"
    steps = r.get("recipe") or []
    if steps:
        text += " (" + "; ".join(recipe_step_text(s) for s in steps) + ")"
    return text


def recipe_step_text(step: dict[str, Any]) -> str:
    if step.get("op") == "filter":
        conds = []
        for w in step.get("where", []):
            if w["op"] in ("is_missing", "not_missing"):
                conds.append(f"{w['column']} {w['op'].replace('_', ' ')}")
            else:
                conds.append(f"{w['column']} {w['op']} {w.get('value')}")
        return "rows where " + " and ".join(conds)
    if step.get("op") == "derive":
        return f"{step.get('name')} = {step.get('expr')}"
    return str(step.get("op"))


def table_text(r: dict[str, Any]) -> str:
    cols = r.get("columns") or []
    rows = r.get("rows") or []
    parts = []
    for row in rows[:MAX_TABLE_ROWS]:
        label, *rest = row
        parts.append(f"{label}: " + ", ".join(f"{c} {num(v)}" for c, v in zip(cols[1:], rest)))
    more = f"; +{len(rows) - MAX_TABLE_ROWS} more rows" if len(rows) > MAX_TABLE_ROWS else ""
    title = r.get("title")
    return (f"{title}: " if title else "") + "; ".join(parts) + more


def _df_text(df: Any) -> str:
    if df is None:
        return ""
    if isinstance(df, (list, tuple)):
        return "(" + ", ".join(num(d) for d in df) + ")"
    return f"({num(df)})"


def stats_text(r: dict[str, Any]) -> str:
    if r.get("test") == "ols":
        return regression_text(r)
    stat = r.get("statistic") or {}
    head = f"{r.get('label', r.get('test', 'test'))}: {stat.get('name', 'statistic')}{_df_text(r.get('df'))} = " \
           f"{num(stat.get('value'))}, {p_text(r.get('p'))}"
    if r.get("alternative") and r["alternative"] != "two-sided":
        head += f" ({r['alternative']})"
    head += f", alpha {num(r.get('alpha'))}"
    eff = r.get("effect") or {}
    parts = [head]
    if eff:
        ci = f", {ci_text(eff['ci'])}" if eff.get("ci") else f" ({eff['ci_method']})" if eff.get("ci_method") else ""
        parts.append(f"{eff.get('name')} = {num(eff.get('value'))}{ci}")
    for name, value in (r.get("extra") or {}).items():
        parts.append(f"{name} = {num(value)}")
    n = r.get("n")
    if isinstance(n, dict) and n:
        parts.append("n = " + ", ".join(f"{v} ({k})" for k, v in n.items()))
    elif n is not None:
        parts.append(f"n = {n}")
    return "; ".join(parts)


def regression_text(r: dict[str, Any]) -> str:
    coefs = r.get("coefficients") or []
    slopes = [c for c in coefs if c.get("name") != "(intercept)"]
    parts = [f"OLS {r.get('formula', '')} (n = {r.get('n')})".replace("  ", " ")]
    for c in slopes:
        parts.append(f"slope {c['name']} = {num(c.get('estimate'))}, {ci_text(c.get('ci'))}, {p_text(c.get('p'))}")
    icpt = next((c for c in coefs if c.get("name") == "(intercept)"), None)
    if icpt is not None:
        parts.append(f"intercept = {num(icpt.get('estimate'))}")
    parts.append(f"R^2 = {num(r.get('r2'))}, F{_df_text(r.get('df'))} = {num((r.get('statistic') or {}).get('value'))}, "
                 f"{p_text(r.get('p'))}, alpha {num(r.get('alpha'))}")
    return "; ".join(parts)


def adjusted_text(r: dict[str, Any]) -> str:
    items = ", ".join(f"{lbl} {num(p, 3)} to {num(q, 3)}"
                      for lbl, p, q in zip(r.get("labels", []), r.get("p", []), r.get("p_adjusted", [])))
    return f"{r.get('method', 'Holm').capitalize()}-adjusted p for {len(r.get('p', []))} tests on " \
           f"{r.get('family', 'the same data')}: {items}; alpha {num(r.get('alpha'))}"


def diagnostic_text(d: dict[str, Any]) -> str:
    bits = [d.get("label", d.get("assumption", ""))]
    if d.get("test"):
        stat = f"{d['test']} {p_text(d['p'])}" if d.get("p") is not None else d["test"]
        bits.append(stat)
    if d.get("note"):
        bits.append(d["note"])
    return ": ".join(b for b in bits[:1]) + (" (" + "; ".join(bits[1:]) + ")" if len(bits) > 1 else "")
