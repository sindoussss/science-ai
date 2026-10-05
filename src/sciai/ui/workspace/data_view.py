"""Data and Plot tabs of the node view.

Data: the dataset a node is (data.load, filter, derive) or read (every data, stats and plot
tool), as its name and shape, a schema table and its first 50 rows. The rows are read off the
UI thread, one preview at a time, and the newest request wins.

Plot: a plot node drawn at the tab's full size, with PNG and SVG export.
"""
from __future__ import annotations

from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable

from PyQt6.QtCore import QObject, QSize, Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QScrollArea,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from sciai.domains.data import frames
from sciai.domains.data.report import recipe_step_text
from sciai.graph import plotspec
from sciai.graph.model import Node
from sciai.tools.registry import get as get_tool
from sciai.ui.chat.inline import Column, TableCard
from sciai.ui.icons import icon
from sciai.ui.plot_image import PlotView, plot_font
from sciai.ui.theme.theme import Theme

PREVIEW_ROWS = 50
PREVIEW_ROW_H = 26
_previews: OrderedDict[str, tuple[list[str], list[list[str]]]] = OrderedDict()  # descriptor key -> rows
PREVIEW_CACHE = 16
_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="sciai-preview")


def dataset_of(node: Node | None, lookup: Callable[[str], Node | None]) -> dict[str, Any] | None:
    """The dataset descriptor a node is, or the one it read; None for every other node."""
    if node is None:
        return None
    r = node.result or {}
    if r.get("kind") == "dataset":
        return r
    if not node.tool_name or not node.tool_inputs:
        return None
    try:
        arg = get_tool(node.tool_name).dataset_arg
    except KeyError:
        return None
    ref = node.tool_inputs.get(arg) if arg else None
    if isinstance(ref, dict) and ref.get("kind") == "dataset":
        return ref
    if isinstance(ref, str):
        src = lookup(ref)
        if src is not None and (src.result or {}).get("kind") == "dataset":
            return src.result
    return None


def plot_of(node: Node | None) -> dict[str, Any] | None:
    r = (node.result or {}) if node is not None else {}
    return r.get("value") if r.get("kind") == "plotspec" else None


def shape_text(desc: dict[str, Any]) -> str:
    rows, cols = int(desc.get("rows", 0)), len(desc.get("columns") or [])
    src = desc.get("source") or {}
    return (f"{rows:,} row{'s' if rows != 1 else ''} × {cols} column{'s' if cols != 1 else ''} · "
            f"{src.get('format', '')} · sha {str(src.get('sha256', ''))[:12]}")


def schema_rows(desc: dict[str, Any]) -> list[list[str]]:
    out = []
    for c in desc.get("columns") or []:
        levels = c.get("levels") or []
        out.append([c["name"], c.get("type", ""), c.get("unit") or "", str(c.get("missing", 0)),
                    ", ".join(str(v) for v in levels)])
    return out


class _Done(QObject):
    """Carries a finished preview from the worker thread to the UI thread (queued signal)."""

    done = pyqtSignal(int, object, object)  # token, (columns, rows) | None, error | None


class DataView(QWidget):
    def __init__(self, theme: Theme) -> None:
        super().__init__()
        self.theme = theme
        self.desc: dict[str, Any] | None = None
        self._token = 0
        self._done = _Done(self)
        self._done.done.connect(self._preview_ready)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        body = QWidget()
        lay = QVBoxLayout(body)
        lay.setContentsMargins(16, 12, 16, 16)
        lay.setSpacing(6)
        self.name = QLabel()
        self.name.setObjectName("dataName")
        self.shape = QLabel()
        self.shape.setObjectName("secondary")
        self.recipe = QLabel()
        self.recipe.setObjectName("secondary")
        self.recipe.setWordWrap(True)
        for w in (self.name, self.shape, self.recipe):
            w.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            lay.addWidget(w)
        lay.addSpacing(6)
        self.schema_host = QVBoxLayout()
        self.schema_host.setContentsMargins(0, 0, 0, 0)
        lay.addLayout(self.schema_host)
        self.schema: TableCard | None = None
        lay.addSpacing(10)
        self.preview_label = QLabel()
        self.preview_label.setObjectName("secondary")
        lay.addWidget(self.preview_label)
        self.table = QTableWidget()
        self.table.setObjectName("dataPreview")
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.ContiguousSelection)
        self.table.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.table.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.table.setWordWrap(False)
        self.table.verticalHeader().setDefaultSectionSize(PREVIEW_ROW_H)
        self.table.verticalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Fixed)
        self.table.horizontalHeader().setHighlightSections(False)
        self.table.verticalHeader().setHighlightSections(False)
        self.table.setShowGrid(True)
        lay.addWidget(self.table)
        lay.addStretch(1)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setWidget(body)
        outer.addWidget(scroll)
        self.show_dataset(None)

    def show_dataset(self, desc: dict[str, Any] | None) -> None:
        if desc is not None and self.desc is not None and desc.get("key") == self.desc.get("key"):
            return  # same data: keep the rows already shown (or loading)
        self.desc = desc
        self._token += 1
        if self.schema is not None:
            # removeWidget before dropping the parent: the layout otherwise keeps an item
            # pointing at a widget deleteLater has freed, and the next layout pass calls
            # minimumSizeHint() on it. That is a segfault, not an exception.
            self.schema_host.removeWidget(self.schema)
            self.schema.setParent(None)
            self.schema.deleteLater()
            self.schema = None
        self.table.clear()
        self.table.setRowCount(0)
        self.table.setColumnCount(0)
        self.table.hide()
        if desc is None:
            self.name.setText("No dataset")
            self.shape.setText("Select a dataset, or a step that read one, to see its columns and rows.")
            self.recipe.hide()
            self.preview_label.hide()
            return
        self.name.setText(desc.get("name") or "dataset")
        self.shape.setText(shape_text(desc))
        steps = desc.get("recipe") or []
        self.recipe.setText("Derived, not stored: " + "; then ".join(recipe_step_text(s) for s in steps))
        self.recipe.setVisible(bool(steps))
        self.schema = TableCard(self.theme, [Column("column", "code", 60, flex=True), Column("type", "text", 40),
                                             Column("unit", "text", 30, shrink=True),
                                             Column("missing", "text", 40),
                                             Column("levels", "text", 50, shrink=True)], schema_rows(desc))
        self.schema_host.addWidget(self.schema)
        self.preview_label.show()
        key = desc.get("key") or ""
        cached = _previews.get(key)
        if cached is not None:
            _previews.move_to_end(key)
            self._fill(*cached)
            return
        self.preview_label.setText("Reading the first rows…")
        token, done = self._token, self._done

        def work() -> None:
            try:
                result, err = frames.preview_rows(desc, PREVIEW_ROWS), None
            except Exception as e:  # noqa: BLE001 - shown in the tab
                result, err = None, e
            try:
                done.done.emit(token, result, err)
            except RuntimeError:
                pass  # the view was closed while the rows were read

        _pool.submit(work)

    def _preview_ready(self, token: int, result: Any, err: Exception | None) -> None:
        if token != self._token or self.desc is None:
            return  # a newer selection replaced this one
        if err is not None:
            self.preview_label.setText(f"The rows could not be read: {err}")
            return
        _previews[self.desc.get("key") or ""] = result
        while len(_previews) > PREVIEW_CACHE:
            _previews.popitem(last=False)
        self._fill(*result)

    def _fill(self, columns: list[str], rows: list[list[str]]) -> None:
        total = int((self.desc or {}).get("rows", len(rows)))
        self.preview_label.setText(f"First {len(rows)} of {total:,} rows" if total > len(rows)
                                   else f"All {len(rows)} rows")
        t = self.table
        t.setColumnCount(len(columns))
        t.setRowCount(len(rows))
        t.setHorizontalHeaderLabels(columns)
        for i, row in enumerate(rows):
            for j, v in enumerate(row):
                item = QTableWidgetItem(v)
                if v and _numeric(v):
                    item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                t.setItem(i, j, item)
        t.resizeColumnsToContents()
        for j in range(len(columns)):
            t.setColumnWidth(j, min(220, max(56, t.columnWidth(j) + 8)))
        bar = t.horizontalScrollBar().sizeHint().height()
        t.setFixedHeight(t.horizontalHeader().sizeHint().height() + PREVIEW_ROW_H * len(rows) + bar + 2)
        t.show()

    def preview_shown(self) -> int:
        return self.table.rowCount() if self.table.isVisible() else 0


def _numeric(v: str) -> bool:
    try:
        float(v)
    except ValueError:
        return False
    return True


class PlotTab(QWidget):
    def __init__(self, theme: Theme) -> None:
        super().__init__()
        self.theme = theme
        self.value: dict[str, Any] | None = None
        self.name = "plot"
        lay = QVBoxLayout(self)
        lay.setContentsMargins(16, 12, 16, 16)
        lay.setSpacing(8)
        row = QHBoxLayout()
        row.setSpacing(6)
        self.export_png = QPushButton("Export PNG")
        self.export_svg = QPushButton("Export SVG")
        for b, fmt in ((self.export_png, "png"), (self.export_svg, "svg")):
            b.setObjectName("outline")
            b.setIcon(icon("download", theme.hex("text_secondary")))
            b.setIconSize(QSize(14, 14))
            b.clicked.connect(lambda _=False, f=fmt: self._export(f))
            row.addWidget(b)
        row.addStretch(1)
        self.summary = QLabel()
        self.summary.setObjectName("secondary")
        self.summary.setWordWrap(True)
        lay.addLayout(row)
        self.view = PlotView(theme)
        lay.addWidget(self.view, 1)
        lay.addWidget(self.summary)

    def show_plot(self, value: dict[str, Any] | None, name: str = "plot") -> None:
        self.value, self.name = value, name
        self.view.set_spec(value)
        self.summary.setText(plotspec.summary(value) if value else "")
        for b in (self.export_png, self.export_svg):
            b.setEnabled(value is not None)

    def _export(self, fmt: str) -> None:
        if self.value is None:
            return
        filt = "PNG image (*.png)" if fmt == "png" else "SVG image (*.svg)"
        path, _ = QFileDialog.getSaveFileName(self, f"Export {fmt.upper()}", f"{self.name}.{fmt}", filt)
        if path:
            if not path.lower().endswith(f".{fmt}"):
                path += f".{fmt}"
            export_plot(self.value, path)


def export_plot(value: dict[str, Any], path: str) -> None:
    """PNG (at 2x) or SVG by the extension, on a white background in the app's plain style."""
    plotspec.export(value, path, font_family=plot_font())


__all__ = ["DataView", "PlotTab", "dataset_of", "export_plot", "plot_of", "schema_rows", "shape_text"]
