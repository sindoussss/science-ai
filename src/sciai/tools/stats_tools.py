"""Statistical tests on datasets, and the independent checks behind them.

Solvers use SciPy's test functions (and NumPy least squares for regression). Each result is
checked three ways, none of which calls the solver's function:

* ``stats.check_formula`` (required, must pass): recomputes the statistic, degrees of freedom,
  p-value, effect size and its confidence interval from the textbook formulas, with the data
  re-extracted row by row. SciPy is used only for distribution tails (t, F, chi-square, normal).
* ``stats.check_permutation``: a permutation (or sign-flip) test of the same hypothesis, seed 0.
  It agrees when it reaches the same decision at alpha and its p-value is within sampling error.
* ``stats.check_statsmodels``: the same test through statsmodels (pinned), for t-tests, ANOVA
  and regression.

Every result carries the diagnostics behind its assumptions (normality, equal variances,
expected counts, constant residual variance); the controller turns them into Assumption nodes.
Effect-size intervals are 95%, whatever alpha the test is judged at. Bootstrap intervals use
seed 0, 2000 resamples (within each group for group comparisons) and percentile bounds
(type-7 quantiles).
"""
from __future__ import annotations

import itertools
import math
from typing import Any, Callable

from sciai.domains.data import frames
from sciai.domains.data.frames import DataError
from sciai.graph.model import Domain
from sciai.tools.data_tools import COLUMN, DATASET, _check, _ok, _type7, canon_dataset
from sciai.tools.registry import ToolSpec, register, schema

ALTERNATIVES = ("two-sided", "greater", "less")
DEFAULT_ALPHA = 0.05
ALPHA = {"type": "number", "exclusiveMinimum": 0, "maximum": 0.5}
LEVELS = {"type": "array", "items": {"type": ["string", "number", "boolean"]}, "minItems": 2, "maxItems": 20}
SEED = 0
BOOTSTRAP = 2_000
BOOTSTRAP_MAX_ROWS = 200_000
BOOTSTRAP_BUDGET = 20_000_000   # resamples x rows; fewer resamples on big data (at least 200)
MIN_BOOTSTRAP = 200
PERMUTATIONS = 10_000
EXACT_LIMIT = 10_000          # enumerate every relabelling when there are at most this many
PERMUTATION_BUDGET = 50_000_000  # permutations x rows; fewer permutations on big data
PERMUTATION_MAX_ROWS = 200_000
MIN_PERMUTATIONS = 1_000
# Asymptotic p-values (chi-square with small expected counts, t with skewed data) legitimately
# differ from permutation ones by a fifth or so; a wrong p-value differs by far more.
PERMUTATION_REL_TOL = 0.2
MW_EXACT_MAX_N = 30           # Mann-Whitney exact distribution when n1 + n2 <= this and no ties
NORMAL_N = 30                 # at or above this many values per group the mean is ~normal (CLT)
SHAPIRO_MAX_N = 5_000
REL = 1e-6                    # agreement between solver and formula/statsmodels recomputations


# ================================================================== shared pieces
_missing = frames.is_missing


def _level_text(v: Any) -> str:
    return str(v)


def resolve_levels(present: list[str], wanted: list[Any] | None, by: str, need: int | None) -> list[str]:
    """Which group levels a test compares, in order. ``present`` are the level names found
    in the data; ``wanted`` the ones asked for (matched exactly, else case-insensitively)."""
    present = sorted(present)
    if wanted:
        out = []
        for w in wanted:
            w = _level_text(w)
            if w in present:
                out.append(w)
                continue
            hit = [k for k in present if k.lower() == w.lower()]
            if len(hit) != 1:
                raise DataError(f"level {w!r} is not in column {by!r}; its levels are: {', '.join(present[:30])}")
            out.append(hit[0])
        if len(set(out)) != len(out):
            raise DataError(f"the levels {out} repeat")
    else:
        out = present
    if need is not None and len(out) != need:
        raise DataError(f"column {by!r} has {len(out)} levels ({', '.join(out[:12])}); this test compares "
                        f"exactly {need}: pass levels=[...] to choose them")
    if len(out) < 2:
        raise DataError(f"column {by!r} has fewer than 2 levels with data")
    return out


def _alt(args: dict[str, Any]) -> str:
    return args.get("alternative") or "two-sided"


def _alpha(args: dict[str, Any]) -> float:
    return float(args.get("alpha") if args.get("alpha") is not None else DEFAULT_ALPHA)


def _t_p(t: float, df: float, alt: str) -> float:
    from scipy.stats import t as tdist

    if alt == "greater":
        return float(tdist.sf(t, df))
    if alt == "less":
        return float(tdist.cdf(t, df))
    return float(min(1.0, 2 * tdist.sf(abs(t), df)))


def _dataset_ref(desc: dict[str, Any]) -> dict[str, Any]:
    return {"name": desc.get("name"), "key": desc["key"], "source_sha": desc["source"]["sha256"]}


def _result(desc: dict[str, Any], test: str, label: str, stat_name: str, stat: float, df: Any, p: float,
            alt: str | None, alpha: float, effect: dict[str, Any] | None, extra: dict[str, Any], n: Any,
            diagnostics: list[dict[str, Any]], spec: dict[str, Any]) -> dict[str, Any]:
    if not (math.isfinite(stat) and math.isfinite(p)):
        raise DataError(f"{label}: the statistic is undefined for these data (no variation?)")
    return {"kind": "stats", "test": test, "label": label, "statistic": {"name": stat_name, "value": float(stat)},
            "df": df, "p": float(p), "alternative": alt, "alpha": alpha, "significant": bool(p < alpha),
            "effect": effect, "extra": extra, "n": n, "diagnostics": diagnostics, "spec": spec,
            "dataset": _dataset_ref(desc)}


def _canon(columns: tuple[str, ...] = (), lists: tuple[str, ...] = (),
           defaults: dict[str, Any] | None = None) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """Fingerprint form: the dataset key, real column names, and defaults filled in, so an
    omitted alternative and an explicit "two-sided" are the same call."""
    def canon(args: dict[str, Any]) -> dict[str, Any]:
        out = canon_dataset({**(defaults or {}), "alpha": DEFAULT_ALPHA,
                             **{k: v for k, v in args.items() if v is not None}}, columns, lists)
        if out.get("levels") is not None:
            out["levels"] = [_level_text(v) for v in out["levels"]]
        return out
    return canon


# ------------------------------------------------------------------ solver-side data (pandas)
def _frame_groups(desc: dict[str, Any], column: str, by: str, levels: list[Any] | None,
                  need: int | None) -> tuple[str, str, list[str], list[Any], int]:
    df = frames.load(desc)
    col, grp = frames.resolve_column(desc, column), frames.resolve_column(desc, by)
    y = frames.numeric(df, col)
    keys = df[grp].astype(str).where(df[grp].notna(), None)
    ok = y.notna() & keys.notna()
    lv = resolve_levels(list(keys[ok].unique()), levels, grp, need)
    arrays = [y[ok & (keys == k)].to_numpy(dtype=float) for k in lv]
    return col, grp, lv, arrays, int((~ok).sum())


def _frame_columns(desc: dict[str, Any], names: list[str]) -> tuple[list[str], list[Any], int]:
    """Numeric columns, rows with a missing value in any of them dropped (listwise)."""
    df = frames.load(desc)
    cols = [frames.resolve_column(desc, c) for c in names]
    if len(set(cols)) != len(cols):
        raise DataError(f"a column is used twice: {cols}")
    vals = [frames.numeric(df, c) for c in cols]
    ok = vals[0].notna()
    for v in vals[1:]:
        ok &= v.notna()
    return cols, [v[ok].to_numpy(dtype=float) for v in vals], int((~ok).sum())


# ------------------------------------------------------------------ checker-side data (plain rows)
def _records(desc: dict[str, Any]) -> list[dict[str, Any]]:
    return frames.load(desc).to_dict("records")


def _plain_groups(desc: dict[str, Any], args: dict[str, Any], need: int | None) -> tuple[list[str], list[list[float]]]:
    import numpy as np

    col, by = frames.resolve_column(desc, args["column"]), frames.resolve_column(desc, args["by"])
    frames.numeric(frames.load(desc).head(1), col)  # refuses a text column
    buckets: dict[str, list[float]] = {}
    for r in _records(desc):
        v, k = r[col], r[by]
        if _missing(v) or _missing(k):
            continue
        buckets.setdefault(_level_text(k), []).append(float(v))
    levels = resolve_levels(list(buckets), args.get("levels"), by, need)
    return levels, [np.asarray(buckets[k], dtype=float) for k in levels]


def _plain_columns(desc: dict[str, Any], names: list[str]) -> list[Any]:
    import numpy as np

    cols = [frames.resolve_column(desc, c) for c in names]
    rows = [[r[c] for c in cols] for r in _records(desc)]
    kept = [[float(v) for v in row] for row in rows if not any(_missing(v) for v in row)]
    return [np.asarray([row[i] for row in kept], dtype=float) for i in range(len(cols))]


def _plain_mean_var(x: Any) -> tuple[float, float]:
    vals = [float(v) for v in x]
    m = math.fsum(vals) / len(vals)
    var = math.fsum((v - m) ** 2 for v in vals) / (len(vals) - 1) if len(vals) > 1 else math.nan
    return m, var


# ------------------------------------------------------------------ diagnostics
def _normality(values: Any, label: str) -> dict[str, Any]:
    n = int(len(values))
    out: dict[str, Any] = {"assumption": "normality", "label": label, "n": n}
    if n < 3:
        return {**out, "ok": False, "note": "fewer than 3 values: normality can't be assessed"}
    if n > SHAPIRO_MAX_N:
        return {**out, "ok": True, "note": f"n >= {NORMAL_N}: the test is robust to non-normality "
                                            f"(Shapiro-Wilk not run above {SHAPIRO_MAX_N} values)"}
    from scipy import stats

    if float(max(values)) == float(min(values)):
        return {**out, "ok": False, "note": "all values are equal"}
    p = float(stats.shapiro(values).pvalue)
    ok = n >= NORMAL_N or p >= 0.05
    note = (f"n >= {NORMAL_N}: robust to non-normality" if n >= NORMAL_N and p < 0.05 else
            "no evidence against normality" if p >= 0.05 else "evidence against normality at 0.05")
    return {**out, "test": "Shapiro-Wilk", "p": p, "ok": ok, "note": note}


def _levene(arrays: list[Any], label: str = "equal variances across groups") -> dict[str, Any]:
    from scipy import stats

    p = float(stats.levene(*arrays, center="median").pvalue)
    if not math.isfinite(p):
        return {"assumption": "equal_variance", "label": label, "ok": False, "note": "no spread to compare"}
    return {"assumption": "equal_variance", "label": label, "test": "Levene (median-centred)", "p": p,
            "ok": p >= 0.05, "note": "no evidence of unequal variances" if p >= 0.05 else
            "evidence of unequal variances at 0.05"}


def _independence() -> dict[str, Any]:
    return {"assumption": "independence", "label": "observations are independent", "ok": None,
            "note": "not testable from the data: it follows from how the data were collected"}


# ------------------------------------------------------------------ bootstrap
def boot_reps(rows: int) -> int:
    """Resamples for a bootstrap on this many rows: 2000, fewer above 10000 rows, 0 (no
    interval) above the row limit. Solver and checker use the same count."""
    if rows > BOOTSTRAP_MAX_ROWS:
        return 0
    return max(MIN_BOOTSTRAP, min(BOOTSTRAP, BOOTSTRAP_BUDGET // max(rows, 1)))


def boot_note(rows: int) -> str:
    reps = boot_reps(rows)
    if not reps:
        return f"no interval: the bootstrap is not run above {BOOTSTRAP_MAX_ROWS} rows"
    return f"percentile bootstrap, seed {SEED}, {reps} resamples"


def boot_indices(sizes: list[int], reps: int = BOOTSTRAP, seed: int = SEED):
    """Resample indices, one array per group, for each replicate. Solver and checker draw the
    same resamples and compute the statistic on them in different ways."""
    import numpy as np

    rng = np.random.default_rng(seed)
    for _ in range(reps):
        yield [rng.integers(0, n, n) for n in sizes]


def _percentile_ci(values: list[float]) -> list[float] | None:
    import numpy as np

    vals = np.asarray([v for v in values if math.isfinite(v)], dtype=float)
    if vals.size < len(values) // 2 or vals.size == 0:
        return None
    lo, hi = np.quantile(vals, [0.025, 0.975])
    return [float(lo), float(hi)]


def _plain_percentile_ci(values: list[float]) -> list[float] | None:
    vals = sorted(v for v in values if math.isfinite(v))
    if len(vals) < len(values) // 2 or not vals:
        return None
    return [_type7(vals, 0.025), _type7(vals, 0.975)]


def _boot_effect(stat: Callable[..., float], arrays: list[Any], percentile: Callable[[list[float]], Any]) -> Any:
    reps = boot_reps(sum(len(a) for a in arrays))
    if not reps:
        return None
    return percentile([stat(*[a[i] for a, i in zip(arrays, idx)])
                       for idx in boot_indices([len(a) for a in arrays], reps)])




# ================================================================== stats.ttest
def ttest_fn(args: dict[str, Any]) -> dict[str, Any]:
    import numpy as np
    from scipy import stats

    desc = frames.require_descriptor(args["dataset"])
    alt, alpha = _alt(args), _alpha(args)
    if args.get("by"):
        col, grp, lv, (a, b), dropped = _frame_groups(desc, args["column"], args["by"], args.get("levels"), 2)
        if min(len(a), len(b)) < 2:
            raise DataError(f"each group needs at least 2 values: {lv[0]} has {len(a)}, {lv[1]} has {len(b)}")
        eq = bool(args.get("equal_var", False))
        res = stats.ttest_ind(a, b, equal_var=eq, alternative=alt)
        ci = stats.ttest_ind(a, b, equal_var=eq).confidence_interval(0.95)
        sp = math.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
        diff = float(a.mean() - b.mean())
        test = "student_t" if eq else "welch_t"
        label = f"{'Student' if eq else 'Welch'} t-test of {col} by {grp} ({lv[0]} vs {lv[1]})"
        diagnostics = [_normality(a, f"{col} is roughly normal in {lv[0]}"),
                       _normality(b, f"{col} is roughly normal in {lv[1]}"), _independence()]
        if eq:
            diagnostics.insert(2, _levene([a, b]))
        effect = {"name": f"mean difference ({lv[0]} - {lv[1]})", "value": diff,
                  "ci": [float(ci.low), float(ci.high)], "ci_method": "t interval"}
        extra = {f"mean {lv[0]}": float(a.mean()), f"mean {lv[1]}": float(b.mean()),
                 "Cohen's d": diff / sp if sp > 0 else math.nan}
        n = {lv[0]: int(len(a)), lv[1]: int(len(b))}
        spec = {"mode": "two_sample", "column": col, "by": grp, "levels": lv, "equal_var": eq, "dropped": dropped}
    else:
        if args.get("paired_with"):
            cols, (x, y), dropped = _frame_columns(desc, [args["column"], args["paired_with"]])
            d, mu, mode = x - y, 0.0, "paired"
            label = f"paired t-test of {cols[0]} - {cols[1]}"
            what = f"mean difference ({cols[0]} - {cols[1]})"
            norm_label = f"the differences {cols[0]} - {cols[1]} are roughly normal"
        else:
            cols, (d,), dropped = _frame_columns(desc, [args["column"]])
            mu, mode = float(args.get("mu") or 0.0), "one_sample"
            label = f"one-sample t-test of {cols[0]} against {frames_num(mu)}"
            what = f"mean of {cols[0]} - {frames_num(mu)}"
            norm_label = f"{cols[0]} is roughly normal"
        if len(d) < 2:
            raise DataError(f"the test needs at least 2 values; there are {len(d)}")
        res = stats.ttest_1samp(d, mu, alternative=alt)
        ci = stats.ttest_1samp(d, mu).confidence_interval(0.95)
        sd = float(d.std(ddof=1))
        test = "paired_t" if mode == "paired" else "one_sample_t"
        diagnostics = [_normality(d, norm_label), _independence()]
        effect = {"name": what, "value": float(d.mean() - mu), "ci": [float(ci.low - mu), float(ci.high - mu)],
                  "ci_method": "t interval"}
        extra = {"Cohen's d": float(d.mean() - mu) / sd if sd > 0 else math.nan}
        n = int(len(d))
        spec = {"mode": mode, "columns": cols, "mu": mu, "dropped": dropped}
    df = float(np.asarray(res.df))
    return _ok(_result(desc, test, label, "t", float(res.statistic), df, float(res.pvalue), alt, alpha,
                       effect, extra, n, diagnostics, {**spec, "alternative": alt}))


def frames_num(x: float) -> str:
    from sciai.domains.data.report import num

    return num(x)


TTEST_ARGS = {"dataset": DATASET, "column": COLUMN, "by": COLUMN, "levels": LEVELS, "paired_with": COLUMN,
              "mu": {"type": "number"}, "equal_var": {"type": "boolean"},
              "alternative": {"enum": list(ALTERNATIVES)}, "alpha": ALPHA}

register(ToolSpec(
    name="stats.ttest", domain=Domain.DATA, kind="solver",
    description="t-test: two groups (by, levels; Welch unless equal_var), paired (paired_with) or one sample (mu)",
    schema=schema(TTEST_ARGS, ["dataset", "column"]), fn=ttest_fn, always_check=True,
    canonical=_canon(("column", "by", "paired_with"), defaults={"alternative": "two-sided"}),
    dataset_arg="dataset",
))


# ================================================================== stats.mannwhitney
def _rank_biserial(a: Any, b: Any) -> float:
    from scipy.stats import rankdata

    ranks = rankdata(list(a) + list(b))
    u1 = float(ranks[: len(a)].sum()) - len(a) * (len(a) + 1) / 2
    return 2 * u1 / (len(a) * len(b)) - 1


def _has_ties(values: Any) -> bool:
    return len(set(float(v) for v in values)) < len(values)


def mannwhitney_fn(args: dict[str, Any]) -> dict[str, Any]:
    import numpy as np
    from scipy import stats

    desc = frames.require_descriptor(args["dataset"])
    alt, alpha = _alt(args), _alpha(args)
    col, grp, lv, (a, b), dropped = _frame_groups(desc, args["column"], args["by"], args.get("levels"), 2)
    if min(len(a), len(b)) < 1:
        raise DataError("each group needs at least 1 value")
    method = "exact" if len(a) + len(b) <= MW_EXACT_MAX_N and not _has_ties(np.concatenate([a, b])) else "asymptotic"
    res = stats.mannwhitneyu(a, b, alternative=alt, method=method, use_continuity=True)
    effect = {"name": f"rank-biserial correlation ({lv[0]} vs {lv[1]})", "value": _rank_biserial(a, b),
              "ci": _boot_effect(_rank_biserial, [a, b], _percentile_ci), "ci_method": boot_note(len(a) + len(b))}
    return _ok(_result(desc, "mann_whitney", f"Mann-Whitney U test of {col} by {grp} ({lv[0]} vs {lv[1]})",
                       "U", float(res.statistic), None, float(res.pvalue), alt, alpha, effect,
                       {f"median {lv[0]}": float(np.median(a)), f"median {lv[1]}": float(np.median(b))},
                       {lv[0]: int(len(a)), lv[1]: int(len(b))}, [_independence()],
                       {"mode": "two_sample", "column": col, "by": grp, "levels": lv, "method": method,
                        "continuity": method == "asymptotic", "dropped": dropped, "alternative": alt}))


register(ToolSpec(
    name="stats.mannwhitney", domain=Domain.DATA, kind="solver",
    description="Mann-Whitney U test (rank-based) comparing a column between two groups: by, levels",
    schema=schema({"dataset": DATASET, "column": COLUMN, "by": COLUMN, "levels": LEVELS,
                   "alternative": {"enum": list(ALTERNATIVES)}, "alpha": ALPHA}, ["dataset", "column", "by"]),
    fn=mannwhitney_fn, always_check=True,
    canonical=_canon(("column", "by"), defaults={"alternative": "two-sided"}), dataset_arg="dataset",
))


# ================================================================== stats.correlation
def _fisher_ci(r: float, n: int, method: str) -> list[float] | None:
    if n < 4 or abs(r) >= 1:
        return None
    se = math.sqrt((1.06 if method == "spearman" else 1.0) / (n - 3))
    z = math.atanh(r)
    q = 1.959963984540054
    return [math.tanh(z - q * se), math.tanh(z + q * se)]


def correlation_fn(args: dict[str, Any]) -> dict[str, Any]:
    from scipy import stats

    desc = frames.require_descriptor(args["dataset"])
    alt, alpha = _alt(args), _alpha(args)
    method = args.get("method", "pearson")
    cols, (x, y), dropped = _frame_columns(desc, [args["x"], args["y"]])
    if len(x) < 3:
        raise DataError(f"a correlation needs at least 3 complete pairs; there are {len(x)}")
    if x.min() == x.max() or y.min() == y.max():
        raise DataError("one of the columns is constant: its correlation is undefined")
    if method == "pearson":
        res = stats.pearsonr(x, y, alternative=alt)
        r, p = float(res.statistic), float(res.pvalue)
        ci = [float(v) for v in stats.pearsonr(x, y).confidence_interval(0.95)] if len(x) > 3 else None
        diagnostics = [_normality(x, f"{cols[0]} is roughly normal"), _normality(y, f"{cols[1]} is roughly normal"),
                       {"assumption": "linearity", "label": "the relationship is linear", "ok": None,
                        "note": "not tested; a scatter plot shows it"}, _independence()]
        ci_method = "Fisher z"
    else:
        res = stats.spearmanr(x, y, alternative=alt)
        r, p = float(res.statistic), float(res.pvalue)
        ci = _fisher_ci(r, len(x), "spearman")
        diagnostics = [_independence()]
        ci_method = "Fisher z with the Fieller variance 1.06/(n - 3)"
    name = "Pearson r" if method == "pearson" else "Spearman rho"
    effect = {"name": name, "value": r, "ci": ci, "ci_method": ci_method}
    return _ok(_result(desc, method, f"{name} between {cols[0]} and {cols[1]}", "r" if method == "pearson" else "rho",
                       r, len(x) - 2, p, alt, alpha, effect, {}, int(len(x)), diagnostics,
                       {"mode": "correlation", "columns": cols, "method": method, "dropped": dropped,
                        "alternative": alt}))


register(ToolSpec(
    name="stats.correlation", domain=Domain.DATA, kind="solver",
    description="correlation between two numeric columns x and y: method=pearson|spearman",
    schema=schema({"dataset": DATASET, "x": COLUMN, "y": COLUMN, "method": {"enum": ["pearson", "spearman"]},
                   "alternative": {"enum": list(ALTERNATIVES)}, "alpha": ALPHA}, ["dataset", "x", "y"]),
    fn=correlation_fn, always_check=True,
    canonical=_canon(("x", "y"), defaults={"alternative": "two-sided", "method": "pearson"}), dataset_arg="dataset",
))


# ================================================================== stats.chi2
def _codes(desc: dict[str, Any], row: str, col: str) -> tuple[list[str], list[str], Any, Any, int]:
    """Category codes for two columns, rows with either missing dropped (solver side)."""
    df = frames.load(desc)
    a = df[row].astype(str).where(df[row].notna(), None)
    b = df[col].astype(str).where(df[col].notna(), None)
    ok = a.notna() & b.notna()
    ra, rb = sorted(a[ok].unique()), sorted(b[ok].unique())
    ca = a[ok].map({k: i for i, k in enumerate(ra)}).to_numpy(dtype=int)
    cb = b[ok].map({k: i for i, k in enumerate(rb)}).to_numpy(dtype=int)
    return ra, rb, ca, cb, int((~ok).sum())


def _cramers_v_codes(ca: Any, cb: Any) -> float:
    import numpy as np

    r, c = int(ca.max()) + 1, int(cb.max()) + 1
    table = np.bincount(ca * c + cb, minlength=r * c).reshape(r, c).astype(float)
    table = table[table.sum(axis=1) > 0][:, table.sum(axis=0) > 0]
    if min(table.shape) < 2:
        return math.nan
    n = table.sum()
    expected = np.outer(table.sum(axis=1), table.sum(axis=0)) / n
    chi2 = float(((table - expected) ** 2 / expected).sum())
    return math.sqrt(chi2 / (n * (min(table.shape) - 1)))


def chi2_fn(args: dict[str, Any]) -> dict[str, Any]:
    import numpy as np
    from scipy import stats

    desc = frames.require_descriptor(args["dataset"])
    alpha = _alpha(args)
    row, col = frames.resolve_column(desc, args["row"]), frames.resolve_column(desc, args["column"])
    if row == col:
        raise DataError("row and column must be different columns")
    ra, rb, ca, cb, dropped = _codes(desc, row, col)
    if len(ra) < 2 or len(rb) < 2:
        raise DataError(f"a chi-square test needs at least 2 levels in each column ({row}: {len(ra)}, {col}: {len(rb)})")
    if len(ra) * len(rb) > 400:
        raise DataError(f"the table would have {len(ra)} x {len(rb)} cells; group the levels first")
    table = np.bincount(ca * len(rb) + cb, minlength=len(ra) * len(rb)).reshape(len(ra), len(rb))
    res = stats.chi2_contingency(table, correction=False)
    n = int(table.sum())
    v = math.sqrt(float(res.statistic) / (n * (min(table.shape) - 1)))
    low = int((res.expected_freq < 5).sum())
    diag = {"assumption": "expected_counts", "label": "every expected count is at least 5",
            "test": "expected counts", "ok": low == 0, "min_expected": float(res.expected_freq.min()),
            "note": "all expected counts >= 5" if low == 0 else
            f"{low} of {table.size} cells expect fewer than 5: the chi-square p-value may be off"}
    ci = _boot_paired(ca, cb)
    effect = {"name": "Cramer's V", "value": v, "ci": ci, "ci_method": boot_note(n)}
    return _ok(_result(desc, "chi2", f"chi-square test of independence of {row} and {col}", "chi2",
                       float(res.statistic), int(res.dof), float(res.pvalue), None, alpha, effect, {}, n,
                       [diag, _independence()],
                       {"mode": "contingency", "row": row, "column": col, "row_levels": ra, "column_levels": rb,
                        "table": table.tolist(), "continuity": False, "dropped": dropped}))


def _boot_paired(ca: Any, cb: Any) -> list[float] | None:
    """Cramer's V resamples whole rows (both codes together)."""
    reps = boot_reps(len(ca))
    if not reps:
        return None
    return _percentile_ci([_cramers_v_codes(ca[i], cb[i]) for (i,) in boot_indices([len(ca)], reps)])


register(ToolSpec(
    name="stats.chi2", domain=Domain.DATA, kind="solver",
    description="chi-square test of independence between two categorical columns: row, column",
    schema=schema({"dataset": DATASET, "row": COLUMN, "column": COLUMN, "alpha": ALPHA}, ["dataset", "row", "column"]),
    fn=chi2_fn, always_check=True, canonical=_canon(("row", "column")), dataset_arg="dataset",
))


# ================================================================== stats.anova
def _eta_squared(*groups: Any) -> float:
    import numpy as np

    allv = np.concatenate(groups)
    grand = allv.mean()
    ss_total = float(((allv - grand) ** 2).sum())
    ss_between = float(sum(len(g) * (g.mean() - grand) ** 2 for g in groups))
    return ss_between / ss_total if ss_total > 0 else math.nan


def anova_fn(args: dict[str, Any]) -> dict[str, Any]:
    from scipy import stats

    desc = frames.require_descriptor(args["dataset"])
    alpha = _alpha(args)
    col, grp, lv, arrays, dropped = _frame_groups(desc, args["column"], args["by"], args.get("levels"), None)
    small = [k for k, a in zip(lv, arrays) if len(a) < 2]
    if small:
        raise DataError(f"each group needs at least 2 values; too few in: {', '.join(small)}")
    res = stats.f_oneway(*arrays)
    total = sum(len(a) for a in arrays)
    effect = {"name": "eta squared", "value": _eta_squared(*arrays),
              "ci": _boot_effect(_eta_squared, arrays, _percentile_ci), "ci_method": boot_note(total)}
    diagnostics = [_normality(a, f"{col} is roughly normal in {k}") for k, a in zip(lv, arrays)]
    diagnostics += [_levene(arrays), _independence()]
    return _ok(_result(desc, "anova", f"one-way ANOVA of {col} by {grp} ({len(lv)} groups)", "F",
                       float(res.statistic), [len(lv) - 1, total - len(lv)], float(res.pvalue), None, alpha,
                       effect, {f"mean {k}": float(a.mean()) for k, a in zip(lv, arrays)},
                       {k: int(len(a)) for k, a in zip(lv, arrays)}, diagnostics,
                       {"mode": "groups", "column": col, "by": grp, "levels": lv, "dropped": dropped}))


register(ToolSpec(
    name="stats.anova", domain=Domain.DATA, kind="solver",
    description="one-way ANOVA comparing a column's mean across the groups of by (optionally only levels)",
    schema=schema({"dataset": DATASET, "column": COLUMN, "by": COLUMN, "levels": LEVELS, "alpha": ALPHA},
                  ["dataset", "column", "by"]),
    fn=anova_fn, always_check=True, canonical=_canon(("column", "by")), dataset_arg="dataset",
))


# ================================================================== stats.kruskal
def _epsilon_h(*groups: Any) -> float:
    """Rank-based effect size for Kruskal-Wallis: (H - k + 1) / (n - k)."""
    from scipy.stats import kruskal

    n, k = sum(len(g) for g in groups), len(groups)
    try:
        h = float(kruskal(*groups).statistic)
    except ValueError:  # every value equal in a resample
        return math.nan
    return (h - k + 1) / (n - k) if n > k else math.nan


def kruskal_fn(args: dict[str, Any]) -> dict[str, Any]:
    from scipy import stats

    desc = frames.require_descriptor(args["dataset"])
    alpha = _alpha(args)
    col, grp, lv, arrays, dropped = _frame_groups(desc, args["column"], args["by"], args.get("levels"), None)
    if any(len(a) < 1 for a in arrays) or sum(len(a) for a in arrays) <= len(lv):
        raise DataError("each group needs values, and more values than groups in all")
    res = stats.kruskal(*arrays)
    total = sum(len(a) for a in arrays)
    effect = {"name": "epsilon squared (rank-based)", "value": _epsilon_h(*arrays),
              "ci": _boot_effect(_epsilon_h, arrays, _percentile_ci), "ci_method": boot_note(total)}
    return _ok(_result(desc, "kruskal", f"Kruskal-Wallis test of {col} by {grp} ({len(lv)} groups)", "H",
                       float(res.statistic), len(lv) - 1, float(res.pvalue), None, alpha, effect,
                       {}, {k: int(len(a)) for k, a in zip(lv, arrays)}, [_independence()],
                       {"mode": "groups", "column": col, "by": grp, "levels": lv, "dropped": dropped}))


register(ToolSpec(
    name="stats.kruskal", domain=Domain.DATA, kind="solver",
    description="Kruskal-Wallis test (rank-based one-way ANOVA) of a column across the groups of by",
    schema=schema({"dataset": DATASET, "column": COLUMN, "by": COLUMN, "levels": LEVELS, "alpha": ALPHA},
                  ["dataset", "column", "by"]),
    fn=kruskal_fn, always_check=True, canonical=_canon(("column", "by")), dataset_arg="dataset",
))


# ================================================================== stats.regression
def _design(x_cols: list[Any]) -> Any:
    import numpy as np

    return np.column_stack([np.ones(len(x_cols[0]))] + list(x_cols))


def _breusch_pagan(X: Any, resid: Any) -> tuple[float, float]:
    """LM = n R^2 of the squared residuals regressed on the predictors (Koenker's form)."""
    import numpy as np
    from scipy.stats import chi2

    e2 = resid ** 2
    beta, *_ = np.linalg.lstsq(X, e2, rcond=None)
    fit = X @ beta
    ss_tot = float(((e2 - e2.mean()) ** 2).sum())
    r2 = 1 - float(((e2 - fit) ** 2).sum()) / ss_tot if ss_tot > 0 else 0.0
    lm = len(e2) * r2
    return lm, float(chi2.sf(lm, X.shape[1] - 1))


def regression_fn(args: dict[str, Any]) -> dict[str, Any]:
    import numpy as np
    from scipy import stats

    desc = frames.require_descriptor(args["dataset"])
    alpha = _alpha(args)
    names = [args["y"], *args["x"]]
    cols, vals, dropped = _frame_columns(desc, names)
    y, X = vals[0], _design(vals[1:])
    n, k = X.shape
    if n <= k:
        raise DataError(f"{n} complete rows for {k} coefficients: the fit needs more rows than coefficients")
    beta, _, rank, _ = np.linalg.lstsq(X, y, rcond=None)
    if rank < k:
        raise DataError("the predictors are collinear (or one is constant): drop one")
    resid = y - X @ beta
    dof = n - k
    rss = float(resid @ resid)
    tss = float(((y - y.mean()) ** 2).sum())
    if tss == 0:
        raise DataError(f"{cols[0]} is constant: there is nothing to explain")
    sigma2 = rss / dof
    cov = sigma2 * np.linalg.inv(X.T @ X)
    se = np.sqrt(np.diag(cov))
    tcrit = float(stats.t.ppf(0.975, dof))
    coefs = []
    for name, b, s in zip(["(intercept)", *cols[1:]], beta, se):
        t = float(b / s) if s > 0 else math.inf
        coefs.append({"name": name, "estimate": float(b), "se": float(s), "t": t,
                      "p": float(2 * stats.t.sf(abs(t), dof)), "ci": [float(b - tcrit * s), float(b + tcrit * s)]})
    r2 = 1 - rss / tss
    f = ((tss - rss) / (k - 1)) / sigma2 if sigma2 > 0 else math.inf
    p = float(stats.f.sf(f, k - 1, dof))
    lm, bp_p = _breusch_pagan(X, resid)
    diagnostics = [_normality(resid, "the residuals are roughly normal"),
                   {"assumption": "constant_variance", "label": "the residual variance is constant",
                    "test": "Breusch-Pagan", "p": bp_p, "statistic": lm, "ok": bp_p >= 0.05,
                    "note": "no evidence of changing variance" if bp_p >= 0.05 else
                    "evidence the residual variance changes at 0.05"},
                   {"assumption": "linearity", "label": "the relationship is linear", "ok": None,
                    "note": "not tested; a residual or fit plot shows it"}, _independence()]
    out = _result(desc, "ols", f"linear regression of {cols[0]} on {', '.join(cols[1:])}", "F", f, [k - 1, dof], p,
                  None, alpha, {"name": "R^2", "value": r2, "ci": None}, {"adjusted R^2": 1 - (1 - r2) * (n - 1) / dof},
                  n, diagnostics, {"mode": "regression", "y": cols[0], "x": cols[1:], "dropped": dropped})
    out.update({"formula": f"{cols[0]} ~ {' + '.join(cols[1:])}", "coefficients": coefs, "r2": r2,
                "residual_se": math.sqrt(sigma2)})
    return _ok(out)


register(ToolSpec(
    name="stats.regression", domain=Domain.DATA, kind="solver",
    description="ordinary least squares: y on numeric predictors x (a list); slopes with 95% CIs, R^2, F test",
    schema=schema({"dataset": DATASET, "y": COLUMN, "x": {"type": "array", "items": COLUMN, "minItems": 1, "maxItems": 10},
                   "alpha": ALPHA}, ["dataset", "y", "x"]),
    fn=regression_fn, always_check=True, canonical=_canon(("y",), ("x",)), dataset_arg="dataset",
))


# ================================================================== stats.adjust
def adjust_fn(args: dict[str, Any]) -> dict[str, Any]:
    from statsmodels.stats.multitest import multipletests

    p = [float(v) for v in args["p"]]
    labels = args.get("labels") or [f"test {i + 1}" for i in range(len(p))]
    if len(labels) != len(p):
        raise DataError("labels and p-values differ in number")
    alpha = _alpha(args)
    reject, adjusted, *_ = multipletests(p, alpha=alpha, method="holm")
    return _ok({"kind": "adjusted", "method": "holm", "labels": labels, "p": p,
                "p_adjusted": [float(v) for v in adjusted], "significant": [bool(v) for v in reject],
                "alpha": alpha, "family": args.get("family") or "the same data"})


register(ToolSpec(
    name="stats.adjust", domain=Domain.DATA, kind="solver",
    description="Holm adjustment of a family of p-values (run by the app when one dataset gets several tests)",
    schema=schema({"p": {"type": "array", "items": {"type": "number", "minimum": 0, "maximum": 1}, "minItems": 1,
                         "maxItems": 200},
                   "labels": {"type": "array", "items": {"type": "string"}}, "method": {"enum": ["holm"]},
                   "alpha": ALPHA, "family": {"type": "string"}}, ["p"]),
    fn=adjust_fn, always_check=True,
))


def check_adjust_fn(args: dict[str, Any]) -> dict[str, Any]:
    r = args["result"]
    p = [float(v) for v in r["p"]]
    m = len(p)
    order = sorted(range(m), key=lambda i: p[i])
    mine = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[i]))
        mine[i] = running
    problems = [f"{lbl}: {q} vs {mq}" for lbl, q, mq in zip(r["labels"], r["p_adjusted"], mine)
                if abs(q - mq) > 1e-12 * max(1.0, mq)]
    sig = [q <= r["alpha"] for q in mine]
    if sig != list(r["significant"]):
        problems.append(f"decisions {r['significant']} vs {sig}")
    if problems:
        return _check("fail", method="Holm step-down in plain Python", problems=problems[:10])
    return _check("pass", method="Holm step-down in plain Python", tests=m)


register(ToolSpec(
    name="stats.check_adjust", domain=Domain.DATA, kind="checker",
    description="recompute Holm-adjusted p-values with a plain step-down loop",
    schema=schema({"result": {"type": "object"}}, ["result"]), fn=check_adjust_fn,
))


# ================================================================== stats.check_formula
class _Cmp:
    def __init__(self, rel: float = REL) -> None:
        self.rel = rel
        self.problems: list[str] = []

    def num(self, name: str, mine: Any, theirs: Any, abs_tol: float = 1e-12) -> None:
        if mine is None or theirs is None:
            if (mine is None) != (theirs is None):
                self.problems.append(f"{name}: {theirs} vs {mine}")
            return
        a, b = float(mine), float(theirs)
        if math.isnan(a) and math.isnan(b):
            return
        if not abs(a - b) <= max(abs_tol, self.rel * max(abs(a), abs(b))):
            self.problems.append(f"{name}: {theirs} vs {mine}")

    def seq(self, name: str, mine: Any, theirs: Any, abs_tol: float = 1e-12) -> None:
        if mine is None or theirs is None:
            self.num(name, mine, theirs)
            return
        if len(mine) != len(theirs):
            self.problems.append(f"{name}: {theirs} vs {mine}")
            return
        for i, (a, b) in enumerate(zip(mine, theirs)):
            self.num(f"{name}[{i}]", a, b, abs_tol)

    def same(self, name: str, mine: Any, theirs: Any) -> None:
        if mine != theirs:
            self.problems.append(f"{name}: {theirs} vs {mine}")


def _plain_levene(groups: list[Any]) -> float:
    """Brown-Forsythe: one-way ANOVA on absolute deviations from each group's median."""
    import statistics

    devs = []
    for g in groups:
        med = statistics.median(float(v) for v in g)
        devs.append([abs(float(v) - med) for v in g])
    return _plain_anova(devs)[2]


def _plain_anova(groups: list[Any]) -> tuple[float, list[int], float, float]:
    from scipy.stats import f as fdist

    vals = [[float(v) for v in g] for g in groups]
    allv = [v for g in vals for v in g]
    n, k = len(allv), len(vals)
    grand = math.fsum(allv) / n
    means = [math.fsum(g) / len(g) for g in vals]
    ssb = math.fsum(len(g) * (m - grand) ** 2 for g, m in zip(vals, means))
    ssw = math.fsum((v - m) ** 2 for g, m in zip(vals, means) for v in g)
    if ssw == 0:
        return math.inf if ssb > 0 else math.nan, [k - 1, n - k], 0.0 if ssb > 0 else math.nan, \
            ssb / (ssb + ssw) if ssb + ssw > 0 else math.nan
    f = (ssb / (k - 1)) / (ssw / (n - k))
    return f, [k - 1, n - k], float(fdist.sf(f, k - 1, n - k)), ssb / (ssb + ssw)


def _check_diag(cmp: _Cmp, r: dict[str, Any], assumption: str, p: float) -> None:
    d = next((d for d in r.get("diagnostics", []) if d["assumption"] == assumption and "p" in d), None)
    if d is not None:
        cmp.num(f"{assumption} p", p, d["p"])


def _formula_ttest(desc: dict[str, Any], a: dict[str, Any], r: dict[str, Any], cmp: _Cmp) -> None:
    from scipy.stats import t as tdist

    alt = _alt(a)
    qt = None
    if a.get("by"):
        levels, (x, y) = _plain_groups(desc, a, 2)
        cmp.same("levels", levels, r["spec"]["levels"])
        n1, n2 = len(x), len(y)
        m1, v1 = _plain_mean_var(x)
        m2, v2 = _plain_mean_var(y)
        sp2 = ((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2)
        if a.get("equal_var"):
            se, df = math.sqrt(sp2 * (1 / n1 + 1 / n2)), n1 + n2 - 2
        else:
            se = math.sqrt(v1 / n1 + v2 / n2)
            df = (v1 / n1 + v2 / n2) ** 2 / ((v1 / n1) ** 2 / (n1 - 1) + (v2 / n2) ** 2 / (n2 - 1))
        diff = m1 - m2
        cmp.num("Cohen's d", diff / math.sqrt(sp2) if sp2 > 0 else math.nan, r["extra"].get("Cohen's d"))
        cmp.num(f"mean {levels[0]}", m1, r["extra"].get(f"mean {levels[0]}"))
        cmp.num(f"mean {levels[1]}", m2, r["extra"].get(f"mean {levels[1]}"))
        cmp.same("n", {levels[0]: n1, levels[1]: n2}, r["n"])
        if a.get("equal_var"):
            _check_diag(cmp, r, "equal_variance", _plain_levene([x, y]))
    else:
        if a.get("paired_with"):
            x, y = _plain_columns(desc, [a["column"], a["paired_with"]])
            d, mu = [float(p) - float(q) for p, q in zip(x, y)], 0.0
        else:
            (x,) = _plain_columns(desc, [a["column"]])
            d, mu = [float(v) for v in x], float(a.get("mu") or 0.0)
        n = len(d)
        m, v = _plain_mean_var(d)
        se, df, diff = math.sqrt(v / n), n - 1, m - mu
        cmp.num("Cohen's d", diff / math.sqrt(v) if v > 0 else math.nan, r["extra"].get("Cohen's d"))
        cmp.same("n", n, r["n"])
    t = diff / se
    qt = float(tdist.ppf(0.975, df))
    cmp.num("t", t, r["statistic"]["value"])
    cmp.num("df", df, r["df"])
    cmp.num("p", _t_p(t, df, alt), r["p"])
    cmp.num("effect", diff, r["effect"]["value"])
    cmp.seq("effect CI", [diff - qt * se, diff + qt * se], r["effect"]["ci"])


def _mw_counts(n1: int, n2: int) -> list[int]:
    """Number of arrangements giving U = 0..n1*n2: the coefficients of the Gaussian binomial
    [n1 + n2 choose n1] in q, built by exact polynomial multiplication and division."""
    poly = [1]
    for i in range(1, n1 + 1):
        # multiply by (1 - q^(n2 + i))
        shift = n2 + i
        out = poly + [0] * shift
        for j, c in enumerate(poly):
            out[j + shift] -= c
        # divide by (1 - q^i): c_k += c_{k-i}
        for j in range(i, len(out)):
            out[j] += out[j - i]
        poly = out[: n1 * n2 + 1]
    return poly


def _u_count(x: Any, y: Any) -> float:
    """U for x: pairs with x > y, plus half the ties, counted by binary search in sorted y
    (the solver sums ranks of the pooled sample)."""
    import numpy as np

    ys = np.sort(np.asarray(y, dtype=float))
    xs = np.asarray(x, dtype=float)
    lo = np.searchsorted(ys, xs, side="left")
    hi = np.searchsorted(ys, xs, side="right")
    return float(lo.sum() + 0.5 * (hi - lo).sum())


def _formula_mannwhitney(desc: dict[str, Any], a: dict[str, Any], r: dict[str, Any], cmp: _Cmp) -> None:
    from scipy.stats import norm

    alt = _alt(a)
    levels, (x, y) = _plain_groups(desc, a, 2)
    cmp.same("levels", levels, r["spec"]["levels"])
    n1, n2 = len(x), len(y)
    u1 = _u_count(x, y)
    u2 = n1 * n2 - u1
    u = u1 if alt == "greater" else u2 if alt == "less" else max(u1, u2)
    allv = sorted(float(v) for v in list(x) + list(y))
    ties = len(set(allv)) < len(allv)
    method = "exact" if n1 + n2 <= MW_EXACT_MAX_N and not ties else "asymptotic"
    cmp.same("method", method, r["spec"]["method"])
    if method == "exact":
        counts = _mw_counts(min(n1, n2), max(n1, n2))
        p = sum(counts[int(round(u)):]) / sum(counts)
    else:
        n = n1 + n2
        tie_sizes: dict[float, int] = {}
        for v in allv:
            tie_sizes[v] = tie_sizes.get(v, 0) + 1
        tie_term = sum(t ** 3 - t for t in tie_sizes.values()) / (n * (n - 1))
        sd = math.sqrt(n1 * n2 / 12 * ((n + 1) - tie_term))
        p = float(norm.sf((u - n1 * n2 / 2 - 0.5) / sd)) if sd > 0 else math.nan
    if alt == "two-sided":
        p *= 2
    p = min(max(p, 0.0), 1.0)
    cmp.num("U", u1, r["statistic"]["value"])
    cmp.num("p", p, r["p"])
    rb = 2 * u1 / (n1 * n2) - 1
    cmp.num("rank-biserial", rb, r["effect"]["value"])
    ci = _boot_effect(lambda p_, q_: 2 * _u_count(p_, q_) / (len(p_) * len(q_)) - 1, [x, y], _plain_percentile_ci)
    cmp.seq("rank-biserial CI", ci, r["effect"]["ci"], 1e-9)


def _plain_pearson(x: list[float], y: list[float]) -> float:
    mx, my = math.fsum(x) / len(x), math.fsum(y) / len(y)
    sxy = math.fsum((a - mx) * (b - my) for a, b in zip(x, y))
    sxx = math.fsum((a - mx) ** 2 for a in x)
    syy = math.fsum((b - my) ** 2 for b in y)
    return sxy / math.sqrt(sxx * syy)


def _plain_ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return ranks


def _formula_correlation(desc: dict[str, Any], a: dict[str, Any], r: dict[str, Any], cmp: _Cmp) -> None:
    alt = _alt(a)
    method = a.get("method", "pearson")
    x, y = (list(map(float, c)) for c in _plain_columns(desc, [a["x"], a["y"]]))
    if method == "spearman":
        x, y = _plain_ranks(x), _plain_ranks(y)
    rho = _plain_pearson(x, y)
    n = len(x)
    rho = max(-1.0, min(1.0, rho))
    t = rho * math.sqrt((n - 2) / (1 - rho * rho)) if abs(rho) < 1 else math.copysign(math.inf, rho)
    cmp.num("r", rho, r["statistic"]["value"], 1e-12)
    cmp.num("p", _t_p(t, n - 2, alt), r["p"])
    cmp.same("n", n, r["n"])
    cmp.seq("r CI", _fisher_ci(rho, n, method), r["effect"]["ci"])


def _formula_chi2(desc: dict[str, Any], a: dict[str, Any], r: dict[str, Any], cmp: _Cmp) -> None:
    from scipy.stats import chi2 as chi2dist

    row, col = frames.resolve_column(desc, a["row"]), frames.resolve_column(desc, a["column"])
    counts: dict[tuple[str, str], int] = {}
    for rec in _records(desc):
        u, v = rec[row], rec[col]
        if _missing(u) or _missing(v):
            continue
        key = (_level_text(u), _level_text(v))
        counts[key] = counts.get(key, 0) + 1
    ra = sorted({k[0] for k in counts})
    rb = sorted({k[1] for k in counts})
    cmp.same("row levels", ra, r["spec"]["row_levels"])
    cmp.same("column levels", rb, r["spec"]["column_levels"])
    table = [[counts.get((i, j), 0) for j in rb] for i in ra]
    cmp.same("table", table, r["spec"]["table"])
    n = sum(map(sum, table))
    rows = [sum(t) for t in table]
    cols = [sum(t[j] for t in table) for j in range(len(rb))]
    stat = math.fsum((table[i][j] - rows[i] * cols[j] / n) ** 2 / (rows[i] * cols[j] / n)
                     for i in range(len(ra)) for j in range(len(rb)))
    dof = (len(ra) - 1) * (len(rb) - 1)
    cmp.num("chi2", stat, r["statistic"]["value"])
    cmp.same("df", dof, r["df"])
    cmp.num("p", float(chi2dist.sf(stat, dof)), r["p"])
    v = math.sqrt(stat / (n * (min(len(ra), len(rb)) - 1)))
    cmp.num("Cramer's V", v, r["effect"]["value"])
    min_expected = min(rows[i] * cols[j] / n for i in range(len(ra)) for j in range(len(rb)))
    d = next(d for d in r["diagnostics"] if d["assumption"] == "expected_counts")
    cmp.num("minimum expected count", min_expected, d["min_expected"])
    cmp.same("expected counts ok", min_expected >= 5, d["ok"])
    if boot_reps(n):
        import numpy as np

        idx = {k: i for i, k in enumerate(ra)}
        jdx = {k: j for j, k in enumerate(rb)}
        # rows in data order, as the solver resamples them
        order = []
        for rec in _records(desc):
            u, v = rec[row], rec[col]
            if not (_missing(u) or _missing(v)):
                order.append((idx[_level_text(u)], jdx[_level_text(v)]))
        ca = np.asarray([o[0] for o in order])
        cb = np.asarray([o[1] for o in order])

        def v_of(sa: Any, sb: Any) -> float:
            t = np.zeros((len(ra), len(rb)))
            np.add.at(t, (sa, sb), 1)
            t = t[t.sum(axis=1) > 0][:, t.sum(axis=0) > 0]
            if min(t.shape) < 2:
                return math.nan
            e = np.outer(t.sum(axis=1), t.sum(axis=0)) / t.sum()
            return math.sqrt(float(((t - e) ** 2 / e).sum()) / (t.sum() * (min(t.shape) - 1)))

        ci = _plain_percentile_ci([v_of(ca[i], cb[i]) for (i,) in boot_indices([len(ca)], boot_reps(n))])
        cmp.seq("Cramer's V CI", ci, r["effect"]["ci"], 1e-9)


def _formula_anova(desc: dict[str, Any], a: dict[str, Any], r: dict[str, Any], cmp: _Cmp) -> None:
    levels, groups = _plain_groups(desc, a, None)
    cmp.same("levels", levels, r["spec"]["levels"])
    f, df, p, eta = _plain_anova(groups)
    cmp.num("F", f, r["statistic"]["value"])
    cmp.same("df", df, r["df"])
    cmp.num("p", p, r["p"])
    cmp.num("eta squared", eta, r["effect"]["value"])
    _check_diag(cmp, r, "equal_variance", _plain_levene(groups))
    ci = _boot_effect(lambda *g: 1 - _ssw(g) / _sst(g), groups, _plain_percentile_ci)
    cmp.seq("eta squared CI", ci, r["effect"]["ci"], 1e-9)


def _ssw(groups: Any) -> float:
    return float(sum(((g - g.mean()) ** 2).sum() for g in groups))


def _sst(groups: Any) -> float:
    import numpy as np

    allv = np.concatenate(groups)
    s = float(((allv - allv.mean()) ** 2).sum())
    return s if s > 0 else math.nan


def _formula_regression(desc: dict[str, Any], a: dict[str, Any], r: dict[str, Any], cmp: _Cmp) -> None:
    """QR decomposition (the solver uses SVD-based least squares)."""
    import numpy as np
    from scipy.linalg import solve_triangular
    from scipy.stats import f as fdist
    from scipy.stats import t as tdist

    vals = _plain_columns(desc, [a["y"], *a["x"]])
    y, X = vals[0], _design(vals[1:])
    n, k = X.shape
    q, rmat = np.linalg.qr(X)
    beta = solve_triangular(rmat, q.T @ y)
    resid = y - X @ beta
    dof = n - k
    rss = math.fsum(float(e) ** 2 for e in resid)
    my = math.fsum(map(float, y)) / n
    tss = math.fsum((float(v) - my) ** 2 for v in y)
    sigma2 = rss / dof
    rinv = solve_triangular(rmat, np.eye(k))
    se = np.sqrt(sigma2 * np.sum(rinv ** 2, axis=1))
    tcrit = float(tdist.ppf(0.975, dof))
    cmp.same("n", n, r["n"])
    for c, b, s in zip(r["coefficients"], beta, se):
        t = float(b / s)
        cmp.num(f"{c['name']} estimate", b, c["estimate"])
        cmp.num(f"{c['name']} se", s, c["se"])
        cmp.num(f"{c['name']} p", float(2 * tdist.sf(abs(t), dof)), c["p"])
        cmp.seq(f"{c['name']} CI", [b - tcrit * s, b + tcrit * s], c["ci"])
    f = ((tss - rss) / (k - 1)) / sigma2
    cmp.num("R^2", 1 - rss / tss, r["r2"])
    cmp.num("F", f, r["statistic"]["value"])
    cmp.num("p", float(fdist.sf(f, k - 1, dof)), r["p"])


def _plain_kruskal_h(groups: list[Any]) -> float:
    pooled = [float(v) for g in groups for v in g]
    ranks = _plain_ranks(pooled)
    n = len(pooled)
    h, start = 0.0, 0
    for g in groups:
        r = math.fsum(ranks[start:start + len(g)])
        h += r * r / len(g)
        start += len(g)
    h = 12 / (n * (n + 1)) * h - 3 * (n + 1)
    sizes: dict[float, int] = {}
    for v in pooled:
        sizes[v] = sizes.get(v, 0) + 1
    correction = 1 - sum(t ** 3 - t for t in sizes.values()) / (n ** 3 - n)
    return h / correction if correction > 0 else math.nan


def _formula_kruskal(desc: dict[str, Any], a: dict[str, Any], r: dict[str, Any], cmp: _Cmp) -> None:
    from scipy.stats import chi2 as chi2dist

    levels, groups = _plain_groups(desc, a, None)
    cmp.same("levels", levels, r["spec"]["levels"])
    h = _plain_kruskal_h(groups)
    k, n = len(groups), sum(len(g) for g in groups)
    cmp.num("H", h, r["statistic"]["value"])
    cmp.same("df", k - 1, r["df"])
    cmp.num("p", float(chi2dist.sf(h, k - 1)), r["p"])
    cmp.num("epsilon squared", (h - k + 1) / (n - k), r["effect"]["value"])
    ci = _boot_effect(lambda *g: (_plain_kruskal_h(list(g)) - len(g) + 1) / (sum(map(len, g)) - len(g)), groups,
                      _plain_percentile_ci)
    cmp.seq("epsilon squared CI", ci, r["effect"]["ci"], 1e-9)


FORMULAS = {"kruskal": _formula_kruskal,
            "welch_t": _formula_ttest, "student_t": _formula_ttest, "one_sample_t": _formula_ttest,
            "paired_t": _formula_ttest, "mann_whitney": _formula_mannwhitney, "pearson": _formula_correlation,
            "spearman": _formula_correlation, "chi2": _formula_chi2, "anova": _formula_anova,
            "ols": _formula_regression}


def _same_dataset(desc: dict[str, Any], r: dict[str, Any], cmp: _Cmp) -> None:
    cmp.same("dataset", desc["key"], (r.get("dataset") or {}).get("key"))
    cmp.same("significant", r["p"] < r["alpha"], r.get("significant"))


def check_formula_fn(args: dict[str, Any]) -> dict[str, Any]:
    desc = frames.require_descriptor(args["dataset"])
    r, a = args["result"], {**args["test_args"], "dataset": desc}
    cmp = _Cmp()
    _same_dataset(desc, r, cmp)
    FORMULAS[r["test"]](desc, a, r, cmp)
    method = "textbook formulas on the re-extracted rows"
    if cmp.problems:
        return _check("fail", method=method, problems=cmp.problems[:10])
    return _check("pass", method=method, test=r["test"])


register(ToolSpec(
    name="stats.check_formula", domain=Domain.DATA, kind="checker",
    description="recompute the statistic, p-value and effect size from textbook formulas",
    schema=schema({"dataset": {"type": "object"}, "test_args": {"type": "object"}, "result": {"type": "object"}},
                  ["dataset", "test_args", "result"]), fn=check_formula_fn,
))


# ================================================================== stats.check_permutation
def _perm_p_tolerance(p: float, p_perm: float, reps: int, exact: bool) -> float:
    """How far apart the two p-values may be. An exact test has no sampling error but moves in
    steps of 1/reps, so it can't get closer than one step (4 + 4 values: 1/35 at best)."""
    spread = 1 / reps if exact else 4 * math.sqrt(max(p_perm * (1 - p_perm), 1 / reps) / reps)
    return spread + PERMUTATION_REL_TOL * max(p, p_perm) + 0.005


def _orient(alt: str, center: float) -> Callable[[float], float]:
    if alt == "greater":
        return lambda s: s - center
    if alt == "less":
        return lambda s: center - s
    return lambda s: abs(s - center)


def _welch_t(a: Any, b: Any, eq: bool) -> float:
    n1, n2 = len(a), len(b)
    v1, v2 = a.var(ddof=1), b.var(ddof=1)
    if eq:
        se = math.sqrt(((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2) * (1 / n1 + 1 / n2))
    else:
        se = math.sqrt(v1 / n1 + v2 / n2)
    return float((a.mean() - b.mean()) / se) if se > 0 else 0.0


def _reps_for(n: int) -> int:
    return max(MIN_PERMUTATIONS, min(PERMUTATIONS, PERMUTATION_BUDGET // max(n, 1)))


def _two_sample_perm(values: Any, n1: int, stat: Callable[[Any], float]) -> tuple[list[float], bool]:
    """Statistic over relabellings: all of them when few, else random ones (seed 0)."""
    import numpy as np

    n = len(values)
    if math.comb(n, n1) <= EXACT_LIMIT:
        out = []
        for chosen in itertools.combinations(range(n), n1):
            mask = np.zeros(n, dtype=bool)
            mask[list(chosen)] = True
            out.append(stat(mask))
        return out, True
    rng = np.random.default_rng(SEED)
    base = np.zeros(n, dtype=bool)
    base[:n1] = True
    return [stat(rng.permutation(base)) for _ in range(_reps_for(n))], False


def _sign_flip_perm(d: Any, stat: Callable[[Any], float]) -> tuple[list[float], bool]:
    import numpy as np

    n = len(d)
    if 2 ** n <= EXACT_LIMIT:
        return [stat(d * np.asarray(signs)) for signs in itertools.product((1.0, -1.0), repeat=n)], True
    rng = np.random.default_rng(SEED)
    return [stat(d * rng.choice((1.0, -1.0), n)) for _ in range(_reps_for(n))], False


def _shuffle_perm(n: int, stat: Callable[[Any], float]) -> tuple[list[float], bool]:
    import numpy as np

    if math.factorial(n) <= EXACT_LIMIT:
        return [stat(np.asarray(p)) for p in itertools.permutations(range(n))], True
    rng = np.random.default_rng(SEED)
    return [stat(rng.permutation(n)) for _ in range(_reps_for(n))], False


def _permutation_null(desc: dict[str, Any], a: dict[str, Any], r: dict[str, Any]) -> tuple[float, list[float], bool, str]:
    """(observed oriented statistic, null statistics, exact?, description)."""
    import numpy as np
    from scipy.stats import rankdata

    test, alt = r["test"], _alt(a)
    if test in ("welch_t", "student_t", "mann_whitney"):
        _, (x, y) = _plain_groups(desc, a, 2)
        values = np.concatenate([x, y])
        n1 = len(x)
        ranks = rankdata(values)
        eq = test == "student_t"

        def raw(m: Any) -> float:
            if test == "mann_whitney":
                return float(ranks[m].sum()) - n1 * (n1 + 1) / 2
            return _welch_t(values[m], values[~m], eq)

        center = n1 * len(y) / 2 if test == "mann_whitney" else 0.0
        what = "U over relabelled groups" if test == "mann_whitney" else "t over relabelled groups"
        orient = _orient(alt, center)
        mask = np.zeros(len(values), dtype=bool)
        mask[:n1] = True
        null, exact = _two_sample_perm(values, n1, lambda m: orient(raw(m)))
        return orient(raw(mask)), null, exact, what
    if test in ("one_sample_t", "paired_t"):
        if test == "paired_t":
            x, y = _plain_columns(desc, [a["column"], a["paired_with"]])
            d = x - y
        else:
            (x,) = _plain_columns(desc, [a["column"]])
            d = x - float(a.get("mu") or 0.0)

        def tstat(v: Any) -> float:
            s = v.std(ddof=1)
            return float(v.mean() / (s / math.sqrt(len(v)))) if s > 0 else 0.0

        orient = _orient(alt, 0.0)
        null, exact = _sign_flip_perm(d, lambda v: orient(tstat(v)))
        return orient(tstat(d)), null, exact, "t over sign flips of the differences"
    if test in ("pearson", "spearman"):
        x, y = _plain_columns(desc, [a["x"], a["y"]])
        if test == "spearman":
            x, y = rankdata(x), rankdata(y)
        xc = (x - x.mean()) / math.sqrt(((x - x.mean()) ** 2).sum())
        yc = (y - y.mean()) / math.sqrt(((y - y.mean()) ** 2).sum())
        orient = _orient(alt, 0.0)
        null, exact = _shuffle_perm(len(x), lambda p: orient(float(xc @ yc[p])))
        return orient(float(xc @ yc)), null, exact, "r over shuffled pairings"
    if test == "anova":
        _, groups = _plain_groups(desc, a, None)
        values = np.concatenate(groups)
        labels = np.repeat(np.arange(len(groups)), [len(g) for g in groups])
        sizes = np.bincount(labels)
        grand = values.mean()
        sst = float(((values - grand) ** 2).sum())
        k, n = len(groups), len(values)

        def fstat(lab: Any) -> float:
            means = np.bincount(lab, weights=values) / sizes
            ssb = float((sizes * (means - grand) ** 2).sum())
            ssw = sst - ssb
            return (ssb / (k - 1)) / (ssw / (n - k)) if ssw > 0 else math.inf

        null, exact = _label_shuffle(labels, fstat)
        return fstat(labels), null, exact, "F over relabelled groups"
    if test == "kruskal":
        _, groups = _plain_groups(desc, a, None)
        values = np.concatenate(groups)
        ranks = rankdata(values)
        labels = np.repeat(np.arange(len(groups)), [len(g) for g in groups])
        sizes = np.bincount(labels)
        n = len(values)
        ties = np.unique(values, return_counts=True)[1]
        correction = 1 - float((ties ** 3 - ties).sum()) / (n ** 3 - n)

        def hstat(lab: Any) -> float:
            sums = np.bincount(lab, weights=ranks)
            return float((12 / (n * (n + 1)) * (sums ** 2 / sizes).sum() - 3 * (n + 1)) / correction)

        null, exact = _label_shuffle(labels, hstat)
        return hstat(labels), null, exact, "H over relabelled groups"
    if test == "chi2":
        ra, rb = r["spec"]["row_levels"], r["spec"]["column_levels"]
        row, col = frames.resolve_column(desc, a["row"]), frames.resolve_column(desc, a["column"])
        pairs = [(rec[row], rec[col]) for rec in _records(desc)]
        pairs = [(u, v) for u, v in pairs if not (_missing(u) or _missing(v))]
        ca = np.asarray([ra.index(_level_text(u)) for u, _ in pairs])
        cb = np.asarray([rb.index(_level_text(v)) for _, v in pairs])
        r_tot = np.bincount(ca, minlength=len(ra)).astype(float)
        c_tot = np.bincount(cb, minlength=len(rb)).astype(float)
        expected = np.outer(r_tot, c_tot) / len(ca)

        def chi(lab: Any) -> float:
            t = np.bincount(ca * len(rb) + lab, minlength=len(ra) * len(rb)).reshape(len(ra), len(rb))
            return float(((t - expected) ** 2 / expected).sum())

        null, exact = _label_shuffle(cb, chi)
        return chi(cb), null, exact, "chi-square over shuffled column labels"
    raise DataError(f"no permutation test for {test}")


def _label_shuffle(labels: Any, stat: Callable[[Any], float]) -> tuple[list[float], bool]:
    import numpy as np

    rng = np.random.default_rng(SEED)
    return [stat(rng.permutation(labels)) for _ in range(_reps_for(len(labels)))], False


PERMUTABLE = ("welch_t", "student_t", "mann_whitney", "one_sample_t", "paired_t", "pearson", "spearman", "anova",
              "kruskal", "chi2")


def check_permutation_fn(args: dict[str, Any]) -> dict[str, Any]:
    desc = frames.require_descriptor(args["dataset"])
    r, a = args["result"], {**args["test_args"], "dataset": desc}
    if r["test"] not in PERMUTABLE:
        return _check("inconclusive", method="permutation test", reason=f"no permutation test for {r['test']}")
    if desc["rows"] > PERMUTATION_MAX_ROWS:
        return _check("inconclusive", method="permutation test",
                      reason=f"{desc['rows']} rows is above the {PERMUTATION_MAX_ROWS}-row limit for this check")
    observed, null, exact, what = _permutation_null(desc, a, r)
    eps = 1e-12 * max(1.0, abs(observed))
    hits = sum(1 for s in null if s >= observed - eps)
    p_perm = hits / len(null) if exact else (hits + 1) / (len(null) + 1)
    p, alpha = float(r["p"]), float(r["alpha"])
    tol = _perm_p_tolerance(p, p_perm, len(null), exact)
    detail = {"method": f"{'exact' if exact else 'Monte Carlo'} permutation test ({what}, seed {SEED})",
              "permutations": len(null), "p_permutation": p_perm, "p_reported": p, "tolerance": tol, "alpha": alpha}
    same_decision = (p_perm < alpha) == (p < alpha)
    close = abs(p_perm - p) <= tol
    if same_decision and close:
        return _check("pass", **detail)
    if not same_decision and not close:
        return _check("fail", **detail, problems=[f"p = {p} but the permutation test gives {p_perm} "
                                                   f"(a different decision at alpha {alpha})"])
    return _check("inconclusive", **detail, reason="the permutation p-value is near the reported one but not "
                  "within sampling error, or the two straddle alpha")


register(ToolSpec(
    name="stats.check_permutation", domain=Domain.DATA, kind="checker",
    description="re-test the hypothesis by permutation (seed 0) and compare the p-value and decision",
    schema=schema({"dataset": {"type": "object"}, "test_args": {"type": "object"}, "result": {"type": "object"}},
                  ["dataset", "test_args", "result"]), fn=check_permutation_fn, cost=3,
))


# ================================================================== stats.check_statsmodels
SM_ALTERNATIVE = {"two-sided": "two-sided", "greater": "larger", "less": "smaller"}


def check_statsmodels_fn(args: dict[str, Any]) -> dict[str, Any]:
    import numpy as np
    import statsmodels
    import statsmodels.api as sm
    from statsmodels.stats.weightstats import CompareMeans, DescrStatsW

    desc = frames.require_descriptor(args["dataset"])
    r, a = args["result"], {**args["test_args"], "dataset": desc}
    cmp = _Cmp()
    _same_dataset(desc, r, cmp)
    test, alt = r["test"], SM_ALTERNATIVE[_alt(a)]
    if test in ("welch_t", "student_t"):
        _, (x, y) = _plain_groups(desc, a, 2)
        usevar = "pooled" if test == "student_t" else "unequal"
        cm = CompareMeans(DescrStatsW(x), DescrStatsW(y))
        t, p, df = cm.ttest_ind(alternative=alt, usevar=usevar)
        ci = list(cm.tconfint_diff(alpha=0.05, usevar=usevar))
        cmp.num("t", t, r["statistic"]["value"])
        cmp.num("df", df, r["df"])
        cmp.num("p", p, r["p"])
        cmp.seq("effect CI", ci, r["effect"]["ci"])
    elif test in ("one_sample_t", "paired_t"):
        if test == "paired_t":
            x, y = _plain_columns(desc, [a["column"], a["paired_with"]])
            d, mu = x - y, 0.0
        else:
            (d,) = _plain_columns(desc, [a["column"]])
            mu = float(a.get("mu") or 0.0)
        ds = DescrStatsW(d)
        t, p, df = ds.ttest_mean(mu, alternative=alt)
        lo, hi = ds.tconfint_mean(alpha=0.05)
        cmp.num("t", t, r["statistic"]["value"])
        cmp.num("df", df, r["df"])
        cmp.num("p", p, r["p"])
        cmp.seq("effect CI", [lo - mu, hi - mu], r["effect"]["ci"])
    elif test == "anova":
        _, groups = _plain_groups(desc, a, None)
        y = np.concatenate(groups)
        X = np.zeros((len(y), len(groups)))
        start = 0
        for j, g in enumerate(groups):
            X[start:start + len(g), j] = 1.0
            start += len(g)
        X = np.column_stack([np.ones(len(y)), X[:, 1:]])
        fit = sm.OLS(y, X).fit()
        cmp.num("F", fit.fvalue, r["statistic"]["value"])
        cmp.num("p", fit.f_pvalue, r["p"])
        cmp.num("eta squared", fit.rsquared, r["effect"]["value"])
    elif test == "ols":
        from statsmodels.stats.diagnostic import het_breuschpagan

        vals = _plain_columns(desc, [a["y"], *a["x"]])
        X = _design(vals[1:])
        fit = sm.OLS(vals[0], X).fit()
        conf = fit.conf_int(0.05)
        for i, c in enumerate(r["coefficients"]):
            cmp.num(f"{c['name']} estimate", fit.params[i], c["estimate"])
            cmp.num(f"{c['name']} se", fit.bse[i], c["se"])
            cmp.num(f"{c['name']} p", fit.pvalues[i], c["p"])
            cmp.seq(f"{c['name']} CI", list(conf[i]), c["ci"])
        cmp.num("R^2", fit.rsquared, r["r2"])
        cmp.num("F", fit.fvalue, r["statistic"]["value"])
        cmp.num("p", fit.f_pvalue, r["p"])
        _check_diag(cmp, r, "constant_variance", float(het_breuschpagan(fit.resid, X)[1]))
    else:
        return _check("inconclusive", reason=f"statsmodels check not offered for {test}")
    method = f"statsmodels {statsmodels.__version__}"
    if cmp.problems:
        return _check("fail", method=method, problems=cmp.problems[:10])
    return _check("pass", method=method, test=test)


register(ToolSpec(
    name="stats.check_statsmodels", domain=Domain.DATA, kind="checker",
    description="recompute t-tests, ANOVA and regression with statsmodels",
    schema=schema({"dataset": {"type": "object"}, "test_args": {"type": "object"}, "result": {"type": "object"}},
                  ["dataset", "test_args", "result"]), fn=check_statsmodels_fn, cost=2,
))
