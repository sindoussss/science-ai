"""PlotSpec: plots are data, not pixels. Tools produce a spec; the UI and export draw it.

Version 1 (Phase 1-2) is one line: ``{"x": [...], "y": [...], "label": str, "xlabel": str}``.
Version 2 adds several series per plot, axis labels with units, linear or log scales, error
bars, histograms, box plots and fitted lines::

    {"version": 2, "title": str, "xlabel": str, "ylabel": str, "xscale": "linear"|"log",
     "yscale": ..., "notes": [str], "series": [
        {"type": "line"|"scatter"|"errorbar", "label": str, "x": [...], "y": [...], "yerr": [...]?},
        {"type": "hist", "label": str, "edges": [...], "counts": [...]},
        {"type": "box", "boxes": [{"label", "whislo", "q1", "med", "q3", "whishi", "fliers": [...]}]}],
     "x": [...], "y": [...]}   # the first x/y series again, so a v1 reader still draws something

``normalize`` turns either version into v2; ``figure`` draws it with matplotlib in the app's
plain style: series told apart by marker shape and line style first, all in the ink color;
a muted gray, then the accent, only once styles run out. Missing y values are None.
"""
from __future__ import annotations

from typing import Any

SERIES_TYPES = ("line", "scatter", "errorbar", "hist", "box")
MARKERS = ("o", "s", "^", "D", "v", "x")
DASHES = ("-", "--", ":", "-.")
MAX_POINTS = 5000  # points drawn per series; larger data is thinned evenly (noted on the plot)


def make(series: list[dict[str, Any]], *, title: str = "", xlabel: str = "", ylabel: str = "",
         xscale: str = "linear", yscale: str = "linear", notes: list[str] | None = None) -> dict[str, Any]:
    value: dict[str, Any] = {"version": 2, "title": title, "xlabel": xlabel, "ylabel": ylabel, "xscale": xscale,
                             "yscale": yscale, "notes": list(notes or []), "series": series}
    first = next((s for s in series if "x" in s and "y" in s), None)
    value["x"], value["y"] = (list(first["x"]), list(first["y"])) if first else ([], [])
    return {"kind": "plotspec", "value": value}


def normalize(value: dict[str, Any]) -> dict[str, Any]:
    """Any stored spec as v2."""
    if value.get("version") == 2:
        return value
    return {"version": 2, "title": value.get("label", ""), "xlabel": value.get("xlabel", ""), "ylabel": "",
            "xscale": "linear", "yscale": "linear", "notes": [],
            "series": [{"type": "line", "label": value.get("label", ""), "x": list(value.get("x", [])),
                        "y": list(value.get("y", []))}],
            "x": list(value.get("x", [])), "y": list(value.get("y", []))}


def summary(value: dict[str, Any]) -> str:
    """One line for the graph digest and the answer text."""
    v = normalize(value)
    parts = []
    for s in v["series"]:
        if s["type"] == "hist":
            parts.append(f"histogram of {sum(s['counts'])} values in {len(s['counts'])} bins")
        elif s["type"] == "box":
            parts.append(f"{len(s['boxes'])} box{'es' if len(s['boxes']) != 1 else ''}")
        else:
            parts.append(f"{s['type']} {s.get('label') or ''} ({len(s.get('x', []))} points)".replace("  ", " "))
    title = v.get("title") or "plot"
    return f"{title}: " + "; ".join(parts)


def thin(n: int, limit: int = MAX_POINTS) -> list[int]:
    """Evenly spaced row indices, at most ``limit`` of them (all rows when there are fewer)."""
    if n <= limit:
        return list(range(n))
    step = n / limit
    return [int(i * step) for i in range(limit)]


DEFAULT_COLORS = {"background": "#FFFFFF", "ink": "#1F1E1B", "axis": "#8A877F", "grid": "#ECEAE4",
                  "muted": "#8A877F", "accent": "#C96442", "text": "#3D3B36"}


def _style(i: int, colors: dict[str, str]) -> dict[str, Any]:
    styles = len(MARKERS)
    color = colors["ink"] if i < styles else colors["muted"] if i < 2 * styles else colors["accent"]
    return {"marker": MARKERS[i % styles], "linestyle": DASHES[i % len(DASHES)], "color": color}


def figure(value: dict[str, Any], width_px: int, height_px: int, *, dpi: int = 100,
           colors: dict[str, str] | None = None, compact: bool = False, font_family: str | None = None) -> Any:
    """A matplotlib Figure (Agg, not attached to pyplot) drawing the spec."""
    from matplotlib.figure import Figure

    c = {**DEFAULT_COLORS, **(colors or {})}
    v = normalize(value)
    fig = Figure(figsize=(width_px / dpi, height_px / dpi), dpi=dpi, facecolor=c["background"])
    ax = fig.add_subplot(1, 1, 1)
    ax.set_facecolor(c["background"])
    size = 6.5 if compact else 9
    rc_font = {"fontsize": size, "color": c["text"]}
    if font_family:
        rc_font["family"] = font_family
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(c["axis"])
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=c["axis"], labelcolor=c["text"], labelsize=size - 1, width=0.8, length=3)
    if compact:  # a thumbnail: the shape of the data, no tick labels to crowd it
        ax.tick_params(labelbottom=False, labelleft=False, length=0)
    else:
        ax.grid(True, color=c["grid"], linewidth=0.6)
        ax.set_axisbelow(True)
        if v.get("xlabel"):
            ax.set_xlabel(v["xlabel"], **rc_font)
        if v.get("ylabel"):
            ax.set_ylabel(v["ylabel"], **rc_font)
        if v.get("title"):
            ax.set_title(v["title"], **{**rc_font, "fontsize": size + 1})
    labelled = False
    for i, s in enumerate(v["series"]):
        st = _style(i, c)
        kind = s.get("type", "line")
        label = s.get("label") or None
        labelled = labelled or bool(label)
        if kind == "hist":
            edges, counts = s["edges"], s["counts"]
            ax.stairs(counts, edges, fill=True, color=c["grid"], label=label)
            ax.stairs(counts, edges, color=st["color"], linewidth=0.9)
        elif kind == "box":
            stats = [{**b, "label": b.get("label", "")} for b in s["boxes"]]
            ax.bxp(stats, showfliers=True, patch_artist=False,
                   boxprops={"color": c["ink"], "linewidth": 0.9}, whiskerprops={"color": c["ink"], "linewidth": 0.9},
                   capprops={"color": c["ink"], "linewidth": 0.9}, medianprops={"color": c["ink"], "linewidth": 1.6},
                   flierprops={"marker": "o", "markersize": 3, "markerfacecolor": "none",
                               "markeredgecolor": c["axis"]})
        else:
            pts = [(x, y, (s.get("yerr") or [None] * len(s["x"]))[j]) for j, (x, y) in enumerate(zip(s["x"], s["y"]))
                   if y is not None and x is not None]
            if not pts:
                continue
            xs, ys, errs = zip(*pts)
            if kind == "scatter":
                ax.scatter(xs, ys, s=10 if compact else 16, marker=st["marker"], facecolors="none",
                           edgecolors=st["color"], linewidths=0.8, label=label)
            elif kind == "errorbar":
                ax.errorbar(xs, ys, yerr=[e or 0 for e in errs], fmt=st["marker"], color=st["color"],
                            markerfacecolor="none", markersize=4, elinewidth=0.8, capsize=2, label=label)
            else:
                ax.plot(xs, ys, linestyle=s.get("dash") or st["linestyle"], color=st["color"],
                        linewidth=1.2 if compact else 1.5, label=label)
    if v.get("xscale") == "log":
        ax.set_xscale("log")
    if v.get("yscale") == "log":
        ax.set_yscale("log")
    if labelled and not compact and len(v["series"]) > 1:
        ax.legend(frameon=False, fontsize=size - 1, labelcolor=c["text"])
    if v.get("notes") and not compact:
        fig.text(0.01, 0.01, "; ".join(v["notes"]), fontsize=size - 2, color=c["axis"])
    fig.tight_layout(pad=0.4 if compact else 0.8)
    return fig


def render_png(value: dict[str, Any], width_px: int, height_px: int, **kw: Any) -> bytes:
    import io

    from matplotlib.backends.backend_agg import FigureCanvasAgg

    fig = figure(value, width_px, height_px, **kw)
    FigureCanvasAgg(fig)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor=fig.get_facecolor())
    return buf.getvalue()


def export(value: dict[str, Any], path: str, width_px: int = 900, height_px: int = 560, **kw: Any) -> None:
    """Write PNG or SVG by the file's extension."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    fig = figure(value, width_px, height_px, **kw)
    FigureCanvasAgg(fig)
    fmt = "svg" if str(path).lower().endswith(".svg") else "png"
    fig.savefig(path, format=fmt, facecolor=fig.get_facecolor(), dpi=kw.get("dpi", 100) * (2 if fmt == "png" else 1))
