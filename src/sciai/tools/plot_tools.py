"""Plot tools return data (a PlotSpec), never pixels; the UI renders it."""
from __future__ import annotations

from typing import Any

import numpy as np
import sympy as sp

from sciai.graph.model import Domain
from sciai.tools.parsing import parse, var
from sciai.tools.registry import ASSUMPTIONS_SCHEMA, ToolSpec, register, schema
from sciai.tools.sympy_tools import BOUND, EXPR, VAR


def function_fn(args: dict[str, Any]) -> dict[str, Any]:
    a = args.get("assumptions") or {}
    e = parse(args["expr"], a)
    x = var(args["var"], a)
    lo = float(sp.N(parse(args["lower"], a)))
    hi = float(sp.N(parse(args["upper"], a)))
    n = int(args.get("points", 200))
    xs = np.linspace(lo, hi, n)
    f = sp.lambdify([x], e, modules="numpy")
    with np.errstate(all="ignore"):
        ys = np.asarray(f(xs), dtype=complex) * np.ones_like(xs)
    ys = np.where(np.abs(ys.imag) < 1e-12, ys.real, np.nan)
    ys = [None if not np.isfinite(v) else float(v) for v in ys]
    return {"result": {"kind": "plotspec", "value": {"x": xs.tolist(), "y": ys, "label": str(e),
                                                     "xlabel": str(x)}},
            "confidence": None, "meta": {}}


register(ToolSpec(
    name="plot.function", domain=Domain.MATH, kind="solver",
    description="sample expr over [lower, upper] for plotting",
    schema=schema({"expr": EXPR, "var": VAR, "lower": BOUND, "upper": BOUND,
                   "points": {"type": "integer", "minimum": 10, "maximum": 2000},
                   "assumptions": ASSUMPTIONS_SCHEMA}, ["expr", "var", "lower", "upper"]),
    fn=function_fn, expr_args=("expr", "lower", "upper"),
))


def series_fn(args: dict[str, Any]) -> dict[str, Any]:
    data = args.get("data")
    if not isinstance(data, dict) or data.get("kind") != "series":
        raise ValueError(f"{args['node']} is not a numeric ODE solution (a series)")
    funcs = list(data.get("funcs") or [])
    comp = args.get("component") or (funcs[0] if funcs else None)
    if comp not in funcs:
        raise ValueError(f"component must be one of {funcs}")
    ys = [None if v is None or not np.isfinite(v) else float(v) for v in data["y"][funcs.index(comp)]]
    return {"result": {"kind": "plotspec", "value": {"x": [float(t) for t in data["t"]], "y": ys, "label": comp,
                                                     "xlabel": str(data.get("var", "t"))}},
            "confidence": None, "meta": {}}


def _canon_series(args: dict[str, Any]) -> dict[str, Any]:
    """The plotted node's id fixes its data (a new version is a new id)."""
    return {"node": args["node"], "component": args.get("component")}


register(ToolSpec(
    name="plot.series", domain=Domain.PHYSICS, kind="solver",
    description='plot a numeric ODE solution: node="n5" (an ode.solve_ivp result), component="x"',
    schema=schema({"node": {"type": "string", "pattern": "^(n[0-9]+|[0-9a-f]{32})$"}, "component": VAR,
                   "data": {"type": "object"}}, ["node"]),
    fn=series_fn, node_arg="node", canonical=_canon_series,
))


# ================================================================== data plots (PlotSpec v2, Phase 3)
# Each takes a dataset (the controller swaps the name or handle for its descriptor), so the plot
# node depends on the dataset node: invalidating the data invalidates the plot.
DATASET = {"type": ["string", "object"], "description": "a dataset node handle or file name"}
COLUMN = {"type": "string", "minLength": 1, "maxLength": 120}


def _frame_xy(desc: dict[str, Any], x: str, y: str | None, by: str | None) -> tuple[Any, str, str | None, str | None]:
    from sciai.domains.data import frames

    df = frames.load(desc)
    xc = frames.resolve_column(desc, x)
    yc = frames.resolve_column(desc, y) if y else None
    bc = frames.resolve_column(desc, by) if by else None
    frames.numeric(df, xc)
    if yc:
        frames.numeric(df, yc)
    return df, xc, yc, bc


def _groups(df: Any, by: str | None) -> list[tuple[str, Any]]:
    if by is None:
        return [("", df)]
    keys = df[by].astype(str).where(df[by].notna(), None)
    levels = sorted(k for k in keys.dropna().unique())
    if len(levels) > 12:
        from sciai.domains.data.frames import DataError

        raise DataError(f"column {by!r} has {len(levels)} levels; a plot shows at most 12 groups")
    return [(k, df[keys == k]) for k in levels]


def _xy_series(kind: str, args: dict[str, Any]) -> dict[str, Any]:
    from sciai.domains.data import frames
    from sciai.graph import plotspec

    desc = frames.require_descriptor(args["dataset"])
    df, xc, yc, bc = _frame_xy(desc, args["x"], args["y"], args.get("by"))
    series, notes = [], []
    for label, part in _groups(df, bc):
        part = part[[xc, yc]].dropna()
        if kind == "line":
            part = part.sort_values(xc, kind="mergesort")
        idx = plotspec.thin(len(part))
        if len(idx) < len(part):
            notes.append(f"{label or 'data'}: {len(idx)} of {len(part)} points shown")
        xs = part[xc].to_numpy(dtype=float)[idx]
        ys = part[yc].to_numpy(dtype=float)[idx]
        series.append({"type": kind, "label": label, "x": xs.tolist(), "y": ys.tolist()})
    if not any(s["x"] for s in series):
        raise frames.DataError(f"no rows have both {xc} and {yc}")
    title = f"{yc} against {xc}" + (f" by {bc}" if bc else "")
    return {"result": plotspec.make(series, title=title, xlabel=xc, ylabel=yc, xscale=args.get("xscale", "linear"),
                                    yscale=args.get("yscale", "linear"), notes=notes),
            "confidence": None, "meta": {}}


def _canon_plot(columns: tuple[str, ...]) -> Any:
    def canon(args: dict[str, Any]) -> dict[str, Any]:
        from sciai.tools.data_tools import canon_dataset

        return canon_dataset(args, columns=columns)
    return canon


SCALE = {"enum": ["linear", "log"]}
for _kind, _desc in (("scatter", "scatter plot of y against x, one marker shape per group of by"),
                     ("line", "line plot of y against x (sorted by x), one line style per group of by")):
    register(ToolSpec(
        name=f"plot.{_kind}", domain=Domain.DATA, kind="solver", description=_desc,
        schema=schema({"dataset": DATASET, "x": COLUMN, "y": COLUMN, "by": COLUMN, "xscale": SCALE, "yscale": SCALE},
                      ["dataset", "x", "y"]),
        fn=(lambda k: lambda args: _xy_series(k, args))(_kind), canonical=_canon_plot(("x", "y", "by")),
        dataset_arg="dataset",
    ))


def histogram_fn(args: dict[str, Any]) -> dict[str, Any]:
    from sciai.domains.data import frames
    from sciai.graph import plotspec

    desc = frames.require_descriptor(args["dataset"])
    df = frames.load(desc)
    col = frames.resolve_column(desc, args["column"])
    values = frames.numeric(df, col).dropna().to_numpy(dtype=float)
    if values.size == 0:
        raise frames.DataError(f"column {col!r} has no values")
    bins = args.get("bins")
    rule = f"{bins} equal-width bins" if bins else "Freedman-Diaconis rule"
    edges = np.histogram_bin_edges(values, bins=int(bins) if bins else "fd")
    if len(edges) > 201:  # a long tail can make the rule ask for thousands of bins
        edges = np.histogram_bin_edges(values, bins=200)
        rule += " (capped at 200 bins)"
    counts, edges = np.histogram(values, bins=edges)
    series = [{"type": "hist", "label": col, "edges": edges.tolist(), "counts": [int(c) for c in counts]}]
    return {"result": plotspec.make(series, title=f"distribution of {col}", xlabel=col, ylabel="count",
                                    notes=[f"bins: {rule}"]), "confidence": None, "meta": {}}


register(ToolSpec(
    name="plot.histogram", domain=Domain.DATA, kind="solver",
    description="histogram of a numeric column (bins: a count, or the Freedman-Diaconis rule when left out)",
    schema=schema({"dataset": DATASET, "column": COLUMN, "bins": {"type": "integer", "minimum": 2, "maximum": 200}},
                  ["dataset", "column"]),
    fn=histogram_fn, canonical=_canon_plot(("column",)), dataset_arg="dataset",
))


def _box(label: str, values: Any) -> dict[str, Any]:
    q1, med, q3 = (float(v) for v in np.quantile(values, [0.25, 0.5, 0.75]))
    iqr = q3 - q1
    inside = values[(values >= q1 - 1.5 * iqr) & (values <= q3 + 1.5 * iqr)]
    fliers = values[(values < q1 - 1.5 * iqr) | (values > q3 + 1.5 * iqr)]
    return {"label": label, "q1": q1, "med": med, "q3": q3, "whislo": float(inside.min()),
            "whishi": float(inside.max()), "fliers": [float(v) for v in fliers[:200]], "n": int(values.size)}


def box_fn(args: dict[str, Any]) -> dict[str, Any]:
    from sciai.domains.data import frames
    from sciai.graph import plotspec

    desc = frames.require_descriptor(args["dataset"])
    df = frames.load(desc)
    col = frames.resolve_column(desc, args["column"])
    by = frames.resolve_column(desc, args["by"]) if args.get("by") else None
    frames.numeric(df, col)
    boxes = []
    for label, part in _groups(df, by):
        vals = part[col].dropna().to_numpy(dtype=float)
        if vals.size:
            boxes.append(_box(label or col, vals))
    if not boxes:
        raise frames.DataError(f"column {col!r} has no values")
    return {"result": plotspec.make([{"type": "box", "label": col, "boxes": boxes}],
                                    title=f"{col}" + (f" by {by}" if by else ""), xlabel=by or "", ylabel=col,
                                    notes=["quartiles by type-7 interpolation; whiskers at 1.5 IQR"]),
            "confidence": None, "meta": {}}


register(ToolSpec(
    name="plot.box", domain=Domain.DATA, kind="solver",
    description="box plot of a numeric column, one box per group of by",
    schema=schema({"dataset": DATASET, "column": COLUMN, "by": COLUMN}, ["dataset", "column"]),
    fn=box_fn, canonical=_canon_plot(("column", "by")), dataset_arg="dataset",
))


def fit_fn(args: dict[str, Any]) -> dict[str, Any]:
    """The regression's data and its fitted line. The line comes from the regression node's own
    coefficients, so the plot shows exactly the fit that was checked."""
    from sciai.domains.data import frames
    from sciai.graph import plotspec

    fit = args.get("data") or {}
    if fit.get("kind") != "stats" or fit.get("test") != "ols":
        raise frames.DataError(f"{args['node']} is not a regression result (stats.regression)")
    desc = frames.require_descriptor(args["dataset"])
    if (fit.get("dataset") or {}).get("key") != desc["key"]:
        raise frames.DataError(f"{args['node']} was fit on another dataset ({(fit.get('dataset') or {}).get('name')})")
    xs_names = fit["spec"]["x"]
    if len(xs_names) != 1:
        raise frames.DataError("plot.fit draws a regression with one predictor; this one has "
                               f"{len(xs_names)}: plot the residuals or each predictor instead")
    xc, yc = xs_names[0], fit["spec"]["y"]
    df = frames.load(desc)[[xc, yc]].dropna()
    idx = plotspec.thin(len(df))
    x = df[xc].to_numpy(dtype=float)
    b0, b1 = (c["estimate"] for c in fit["coefficients"])
    grid = np.linspace(float(x.min()), float(x.max()), 100)
    series = [{"type": "scatter", "label": "data", "x": x[idx].tolist(),
               "y": df[yc].to_numpy(dtype=float)[idx].tolist()},
              {"type": "line", "label": "least-squares fit", "x": grid.tolist(), "y": (b0 + b1 * grid).tolist(),
               "dash": "-"}]
    notes = [f"{len(idx)} of {len(df)} points shown"] if len(idx) < len(df) else []
    return {"result": plotspec.make(series, title=f"{yc} against {xc} with the fitted line", xlabel=xc, ylabel=yc,
                                    notes=notes), "confidence": None, "meta": {}}


register(ToolSpec(
    name="plot.fit", domain=Domain.DATA, kind="solver",
    description='a regression\'s data with its fitted line: dataset, node="n7" (a stats.regression result)',
    schema=schema({"dataset": DATASET, "node": {"type": "string", "pattern": "^(n[0-9]+|[0-9a-f]{32})$"},
                   "data": {"type": "object"}}, ["dataset", "node"]),
    fn=fit_fn, canonical=lambda a: {**_canon_plot(())({"dataset": a["dataset"]}), "node": a["node"]},
    dataset_arg="dataset", node_arg="node",
))
