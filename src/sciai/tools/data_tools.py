"""Data tools: load a dataset, summarize it, filter it, derive columns, aggregate by group.

Every tool reads its dataset through ``frames.load`` from a descriptor the controller puts
in the ``dataset`` argument (the model names a dataset node or a file name; it never
passes data). Filters and derived columns return a new descriptor (a recipe on the same
file), never stored data. Each solver has a checker that recomputes its figures by a
different route: the csv module or openpyxl instead of pandas, plain-Python sums,
quantiles and row tests instead of pandas methods.
"""
from __future__ import annotations

import math
import statistics
from typing import Any

from sciai.domains.data import frames
from sciai.domains.data.frames import DataError
from sciai.graph.model import Domain
from sciai.tools.registry import ToolSpec, register, schema

DATASET = {"type": ["string", "object"], "description": "a dataset node handle or file name"}
COLUMN = {"type": "string", "minLength": 1, "maxLength": 120}
COLUMNS = {"type": "array", "items": COLUMN, "minItems": 1, "maxItems": 30}
QUANTILE_METHOD = "linear interpolation between order statistics (Hyndman-Fan type 7)"
AGGS = ("mean", "median", "sum", "count", "sd", "min", "max")
METHODS = ("pandas", "numpy")  # how a summary is computed; a retry after a failed check switches
REL_TOL = 1e-9


def _ok(result: dict[str, Any], **meta: Any) -> dict[str, Any]:
    return {"result": result, "confidence": None, "meta": meta}


def _check(outcome: str, **detail: Any) -> dict[str, Any]:
    return {"result": {"kind": "check", "outcome": outcome, **detail}, "confidence": None, "meta": {}}


def _close(a: float, b: float, tol: float = REL_TOL) -> bool:
    if math.isnan(a) and math.isnan(b):
        return True
    return abs(a - b) <= tol * max(1.0, abs(a), abs(b))


def canon_dataset(args: dict[str, Any], columns: tuple[str, ...] = (), column_lists: tuple[str, ...] = (),
                  drop: tuple[str, ...] = ()) -> dict[str, Any]:
    """Fingerprint form: the dataset by its key (file hash, read options, recipe), column names
    resolved to the real ones, so "Score" and "score" are the same call."""
    desc = frames.require_descriptor(args.get("dataset"))
    out = {k: v for k, v in sorted(args.items()) if k != "dataset" and k not in drop}
    out["dataset"] = desc["key"]
    for k in columns:
        if out.get(k) is not None:
            out[k] = frames.resolve_column(desc, out[k])
    for k in column_lists:
        if out.get(k) is not None:
            out[k] = [frames.resolve_column(desc, c) for c in out[k]]
    return out


# ================================================================== data.load
def load_fn(args: dict[str, Any]) -> dict[str, Any]:
    path, fmt = args["path"], args["format"]
    sha = frames.file_sha256(path)
    if sha != args["sha256"]:
        raise DataError(f"the stored file changed since import (SHA-256 {sha[:12]} != {args['sha256'][:12]})")
    options = args.get("options") or frames.sniff_options(path, fmt)
    df = frames.read_source(path, fmt, options, int(args.get("max_rows", 5_000_000)))
    source = {"sha256": sha, "format": fmt, "options": options, "path": path}
    return _ok(frames.make_descriptor(args["name"], source, [], df, with_checks=True))


def _canon_load(args: dict[str, Any]) -> dict[str, Any]:
    return {"sha256": args["sha256"], "format": args["format"], "options": args.get("options"), "name": args["name"]}


register(ToolSpec(
    name="data.load", domain=Domain.DATA, kind="solver",
    description="read an imported data file (run by the app, not the model)",
    schema=schema({"path": {"type": "string"}, "format": {"enum": ["csv", "tsv", "xlsx"]},
                   "sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
                   "name": {"type": "string", "maxLength": 200}, "options": {"type": "object"},
                   "max_rows": {"type": "integer", "minimum": 1}}, ["path", "format", "sha256", "name"]),
    fn=load_fn, always_check=True, canonical=_canon_load,
))


def check_load_fn(args: dict[str, Any]) -> dict[str, Any]:
    desc = args["loaded"]
    src = desc["source"]
    reread = (frames.reread_xlsx if src["format"] == "xlsx" else frames.reread_csv)(src["path"], src["options"])
    problems = frames.compare_reread(desc, reread)
    method = "openpyxl" if src["format"] == "xlsx" else "Python csv module"
    if problems:
        return _check("fail", method=method, problems=problems[:10])
    return _check("pass", method=method, rows=reread["rows"], columns=len(reread["columns"]),
                  numeric_sums_checked=sorted(reread["sums"]))


register(ToolSpec(
    name="data.check_load", domain=Domain.DATA, kind="checker",
    description="re-read the file without pandas: same rows, columns, missing counts and numeric sums",
    schema=schema({"loaded": {"type": "object"}}, ["loaded"]), fn=check_load_fn,
))


# ================================================================== data.describe
def _numeric_columns(df: Any, desc: dict[str, Any], requested: list[str] | None) -> list[str]:
    if requested:
        cols = [frames.resolve_column(desc, c) for c in requested]
        for c in cols:
            frames.numeric(df, c)  # raises for a non-numeric column
        return cols
    cols = [c["name"] for c in desc["columns"] if c["type"] in ("integer", "number")]
    if not cols:
        raise DataError(f"{desc.get('name', 'the dataset')} has no numeric columns; use data.group to count")
    return cols


def _summary_row(values: Any, method: str) -> list[float | None]:
    """mean, sd, min, q1, median, q3, max of non-missing values: pandas methods, or NumPy
    functions on the raw array (the method a retry switches to)."""
    import numpy as np

    n = len(values)
    if method == "numpy":
        a = values.to_numpy(dtype=float)
        q = np.quantile(a, [0.25, 0.5, 0.75], method="linear")
        return [float(np.mean(a)), float(np.std(a, ddof=1)) if n > 1 else None, float(a.min()),
                float(q[0]), float(q[1]), float(q[2]), float(a.max())]
    q = values.quantile([0.25, 0.5, 0.75])
    return [float(values.mean()), float(values.std(ddof=1)) if n > 1 else None, float(values.min()),
            float(q.loc[0.25]), float(q.loc[0.5]), float(q.loc[0.75]), float(values.max())]


def describe_fn(args: dict[str, Any]) -> dict[str, Any]:
    desc = frames.require_descriptor(args["dataset"])
    df = frames.load(desc)
    rows = []
    for col in _numeric_columns(df, desc, args.get("columns")):
        s = frames.numeric(df, col)
        v = s.dropna()
        n = int(v.size)
        if n == 0:
            rows.append([col, 0, int(s.isna().sum())] + [None] * 7)
            continue
        rows.append([col, n, int(s.isna().sum()), *_summary_row(v, args.get("method", "pandas"))])
    return _ok({"kind": "table", "title": f"summary of {desc.get('name', 'dataset')}",
                "columns": ["column", "n", "missing", "mean", "sd", "min", "q1", "median", "q3", "max"],
                "rows": rows, "note": f"sd uses n - 1; quartiles by {QUANTILE_METHOD}",
                "dataset": {"name": desc.get("name"), "key": desc["key"]}})


register(ToolSpec(
    name="data.describe", domain=Domain.DATA, kind="solver",
    description="n, missing, mean, sd, min, quartiles and max of numeric columns",
    schema=schema({"dataset": DATASET, "columns": COLUMNS, "method": {"enum": list(METHODS)}}, ["dataset"]),
    fn=describe_fn, canonical=lambda a: {**canon_dataset(a, column_lists=("columns",)),
                                         "method": a.get("method", "pandas")},
    dataset_arg="dataset", methods=METHODS,
))


def _type7(sorted_vals: list[float], p: float) -> float:
    h = (len(sorted_vals) - 1) * p
    lo = math.floor(h)
    hi = min(lo + 1, len(sorted_vals) - 1)
    return sorted_vals[lo] + (h - lo) * (sorted_vals[hi] - sorted_vals[lo])


def _plain_stats(values: list[float]) -> list[float | None]:
    vals = sorted(values)
    n = len(vals)
    mean = math.fsum(vals) / n
    sd = math.sqrt(math.fsum((x - mean) ** 2 for x in vals) / (n - 1)) if n > 1 else None
    return [mean, sd, vals[0], _type7(vals, 0.25), _type7(vals, 0.5), _type7(vals, 0.75), vals[-1]]


def check_describe_fn(args: dict[str, Any]) -> dict[str, Any]:
    desc = frames.require_descriptor(args["dataset"])
    df = frames.load(desc)
    table = args["table"]
    problems = []
    for row in table["rows"]:
        col = row[0]
        raw = [None if frames.is_missing(x) else float(x)
               for x in df[col].tolist()]
        vals = [x for x in raw if x is not None]
        if [len(vals), len(raw) - len(vals)] != list(row[1:3]):
            problems.append(f"{col}: n/missing {row[1:3]} vs {[len(vals), len(raw) - len(vals)]}")
            continue
        if not vals:
            continue
        for name, mine, theirs in zip(table["columns"][3:], _plain_stats(vals), row[3:]):
            if (mine is None) != (theirs is None) or (mine is not None and not _close(mine, float(theirs), 1e-9)):
                problems.append(f"{col} {name}: {theirs} vs {mine}")
    if problems:
        return _check("fail", method="fsum and sorted-list quantiles", problems=problems[:10])
    return _check("pass", method="fsum and sorted-list quantiles", columns=[r[0] for r in table["rows"]])


register(ToolSpec(
    name="data.check_describe", domain=Domain.DATA, kind="checker",
    description="recompute the summary with math.fsum, a two-pass variance and type-7 quantiles",
    schema=schema({"dataset": {"type": "object"}, "table": {"type": "object"}}, ["dataset", "table"]),
    fn=check_describe_fn,
))


# ================================================================== data.filter / data.derive
WHERE = {"type": "array", "minItems": 1, "maxItems": 10, "items": {
    "type": "object", "additionalProperties": False, "required": ["column", "op"],
    "properties": {"column": COLUMN, "op": {"enum": list(frames.FILTER_OPS)},
                   "value": {"type": ["number", "string", "boolean", "array"]}}}}


def filter_fn(args: dict[str, Any]) -> dict[str, Any]:
    desc = frames.require_descriptor(args["dataset"])
    where = [{**w, "column": frames.resolve_column(desc, w["column"])} for w in args["where"]]
    op = {"op": "filter", "where": where}
    df = frames.apply_recipe(frames.load(desc), [op])
    if len(df) == 0:
        raise DataError("the filter keeps no rows")
    out = frames.derived(desc, op, df)
    out["checks"] = frames.checks_of(df)
    out["parent_rows"] = desc["rows"]
    return _ok(out)


register(ToolSpec(
    name="data.filter", domain=Domain.DATA, kind="solver",
    description='keep rows where every condition holds: where=[{"column": "group", "op": "==", "value": "A"}]',
    schema=schema({"dataset": DATASET, "where": WHERE}, ["dataset", "where"]),
    fn=filter_fn, canonical=lambda a: {**canon_dataset(a), "where": [
        {**w, "column": frames.resolve_column(a["dataset"], w["column"])} for w in a["where"]]},
    dataset_arg="dataset",
))


def derive_fn(args: dict[str, Any]) -> dict[str, Any]:
    desc = frames.require_descriptor(args["dataset"])
    variables = {k: frames.resolve_column(desc, c) for k, c in args["vars"].items()}
    name = args["name"].strip()
    op = {"op": "derive", "name": name, "expr": args["expr"], "vars": variables}
    df = frames.apply_recipe(frames.load(desc), [op])
    out = frames.derived(desc, op, df)
    out["checks"] = frames.checks_of(df)
    out["parent_rows"] = desc["rows"]
    return _ok(out)


def _canon_derive(args: dict[str, Any]) -> dict[str, Any]:
    from sciai.tools.parsing import canonical, parse

    out = canon_dataset(args)
    out["vars"] = {k: frames.resolve_column(args["dataset"], c) for k, c in sorted(args["vars"].items())}
    out["expr"] = canonical(parse(args["expr"]))
    return out


register(ToolSpec(
    name="data.derive", domain=Domain.DATA, kind="solver",
    description='add a computed column: name="bmi", expr="m/h**2", vars={"m": "mass [kg]", "h": "height [m]"}',
    schema=schema({"dataset": DATASET, "name": {"type": "string", "minLength": 1, "maxLength": 60},
                   "expr": {"type": "string", "maxLength": 500},
                   "vars": {"type": "object", "minProperties": 1, "maxProperties": 10,
                            "additionalProperties": COLUMN}}, ["dataset", "name", "expr", "vars"]),
    fn=derive_fn, canonical=_canon_derive, dataset_arg="dataset", expr_args=("expr",),
))


def _row_passes(row: dict[str, Any], cond: dict[str, Any]) -> bool:
    """One filter condition on one row, in plain Python (the solver uses pandas masks)."""
    v = row[cond["column"]]
    missing = frames.is_missing(v)
    op, target = cond["op"], cond.get("value")
    if op == "is_missing":
        return missing
    if op == "not_missing":
        return not missing
    if missing:
        return False
    is_num = isinstance(v, (int, float)) and not isinstance(v, bool)
    if op in ("in", "not_in"):
        pool = {float(t) for t in target} if is_num else {str(t) for t in target}
        hit = (float(v) if is_num else str(v)) in pool
        return hit if op == "in" else not hit
    if is_num:
        a, b = float(v), float(target)
    else:
        a, b = str(v), str(target)
    return {"==": a == b, "!=": a != b, "<": a < b, "<=": a <= b, ">": a > b, ">=": a >= b}[op]


def _scalar_derive(expr: str, variables: dict[str, str], rows: list[dict[str, Any]]) -> list[float]:
    import sympy as sp

    from sciai.tools.parsing import parse, var

    e = parse(expr)
    f = sp.lambdify([var(k) for k in variables], e, modules="math")
    out = []
    for row in rows:
        vals = [row[c] for c in variables.values()]
        if any(frames.is_missing(v) for v in vals):
            out.append(math.nan)
            continue
        try:
            x = complex(f(*[float(v) for v in vals]))
            out.append(x.real if abs(x.imag) < 1e-12 and math.isfinite(x.real) else math.nan)
        except (ValueError, ZeroDivisionError, OverflowError):
            out.append(math.nan)
    return out


SAMPLE_ROWS = 2000


def check_frame_fn(args: dict[str, Any]) -> dict[str, Any]:
    """Rebuild the last recipe step row by row from the parent frame and compare."""
    result = frames.require_descriptor(args["result"])
    *parent_recipe, op = result["recipe"]
    parent = {**result, "recipe": parent_recipe}
    rows = frames.load(parent).to_dict("records")
    if op["op"] == "filter":
        kept = [r for r in rows if all(_row_passes(r, c) for c in op["where"])]
        problems = []
        if len(kept) != result["rows"]:
            problems.append(f"{result['rows']} rows kept vs {len(kept)} by a row-by-row test")
        for col, figures in (result.get("checks") or {}).items():
            if "sum" in figures:
                vals = [float(r[col]) for r in kept if not frames.is_missing(r[col])]
                if len(vals) != figures["count"] or not _close(math.fsum(vals), figures["sum"]):
                    problems.append(f"{col}: sum {figures['sum']} over {figures['count']} vs "
                                    f"{math.fsum(vals)} over {len(vals)}")
        method = "row-by-row filter in plain Python"
    else:
        frame = frames.load(result)
        step = max(1, len(rows) // SAMPLE_ROWS)
        idx = list(range(0, len(rows), step))
        mine = _scalar_derive(op["expr"], op["vars"], [rows[i] for i in idx])
        theirs = frame[op["name"]].tolist()
        problems = [f"row {i}: {theirs[i]} vs {m}" for i, m in zip(idx, mine) if not _close(m, float(theirs[i]), 1e-9)]
        method = f"scalar math evaluation on {len(idx)} rows"
    if problems:
        return _check("fail", method=method, problems=problems[:10])
    return _check("pass", method=method)


register(ToolSpec(
    name="data.check_frame", domain=Domain.DATA, kind="checker",
    description="recompute a filter or derived column row by row in plain Python",
    schema=schema({"result": {"type": "object"}}, ["result"]), fn=check_frame_fn,
))


# ================================================================== data.group
def _agg(values: list[float], agg: str) -> float | None:
    if agg == "count":
        return float(len(values))
    if not values:
        return None
    if agg == "mean":
        return math.fsum(values) / len(values)
    if agg == "sum":
        return math.fsum(values)
    if agg == "median":
        return float(statistics.median(values))
    if agg == "sd":
        return statistics.stdev(values) if len(values) > 1 else None
    return float(min(values) if agg == "min" else max(values))


def group_fn(args: dict[str, Any]) -> dict[str, Any]:
    desc = frames.require_descriptor(args["dataset"])
    df = frames.load(desc)
    by = frames.resolve_column(desc, args["by"])
    col = frames.resolve_column(desc, args["column"])
    agg = args.get("agg", "mean")
    values = df[col] if agg == "count" else frames.numeric(df, col)
    keys = df[by].astype(str).where(df[by].notna(), None)
    rows = []
    for key in sorted(k for k in keys.dropna().unique()):
        part = values[keys == key].dropna()
        n = int(part.size)
        if agg == "count":
            stat: float | None = float(n)
        elif n == 0 or (agg == "sd" and n < 2):
            stat = None
        elif args.get("method") == "numpy":
            import numpy as np

            a = part.to_numpy(dtype=float)
            stat = float({"mean": np.mean, "median": np.median, "sum": np.sum, "min": np.min, "max": np.max,
                          "sd": lambda x: np.std(x, ddof=1)}[agg](a))
        else:
            stat = float({"mean": part.mean, "median": part.median, "sum": part.sum, "min": part.min,
                          "max": part.max}.get(agg, lambda: part.std(ddof=1))())
        rows.append([key, stat, n])
    if not rows:
        raise DataError(f"column {by!r} has no values to group by")
    return _ok({"kind": "table", "title": f"{agg} of {col} by {by}", "columns": [by, f"{agg}({col})", "n"],
                "rows": rows, "dataset": {"name": desc.get("name"), "key": desc["key"]},
                "group": {"by": by, "column": col, "agg": agg}})


register(ToolSpec(
    name="data.group", domain=Domain.DATA, kind="solver",
    description="aggregate a column per group: by, column, agg=mean|median|sum|count|sd|min|max",
    schema=schema({"dataset": DATASET, "by": COLUMN, "column": COLUMN, "agg": {"enum": list(AGGS)},
                   "method": {"enum": list(METHODS)}}, ["dataset", "by", "column"]),
    fn=group_fn, canonical=lambda a: {**canon_dataset(a, columns=("by", "column")), "agg": a.get("agg", "mean"),
                                      "method": a.get("method", "pandas")},
    dataset_arg="dataset", methods=METHODS,
))


def check_group_fn(args: dict[str, Any]) -> dict[str, Any]:
    desc = frames.require_descriptor(args["dataset"])
    table = args["table"]
    g = table["group"]
    buckets: dict[str, list[float]] = {}
    for r in frames.load(desc).to_dict("records"):
        key, v = r[g["by"]], r[g["column"]]
        if frames.is_missing(key):
            continue
        bucket = buckets.setdefault(str(key), [])
        if frames.is_missing(v):
            continue
        bucket.append(v if g["agg"] == "count" else float(v))
    problems = []
    if sorted(buckets) != [row[0] for row in table["rows"]]:
        problems.append(f"groups {[row[0] for row in table['rows']]} vs {sorted(buckets)}")
    for key, stat, n in table["rows"]:
        vals = buckets.get(key, [])
        mine = _agg([float(x) for x in vals] if g["agg"] != "count" else vals, g["agg"])
        if n != len(vals) or (mine is None) != (stat is None) or (mine is not None and not _close(mine, stat)):
            problems.append(f"{key}: {stat} (n={n}) vs {mine} (n={len(vals)})")
    if problems:
        return _check("fail", method="plain-Python grouping", problems=problems[:10])
    return _check("pass", method="plain-Python grouping", groups=len(table["rows"]))


register(ToolSpec(
    name="data.check_group", domain=Domain.DATA, kind="checker",
    description="recompute each group's aggregate with plain Python",
    schema=schema({"dataset": {"type": "object"}, "table": {"type": "object"}}, ["dataset", "table"]),
    fn=check_group_fn,
))
