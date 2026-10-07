"""Measured UI metrics for the injected-fault scenario (shared by the metrics test and the screenshot script).

Everything here reads real widget geometry after layout; nothing is estimated.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

from PyQt6.QtCore import QRect
from PyQt6.QtWidgets import (
    QAbstractButton,
    QAbstractScrollArea,
    QApplication,
    QLabel,
    QLineEdit,
    QTabBar,
    QWidget,
)

from sciai.graph.model import NodeType
from sciai.tools.faults import FaultyRunner
from sciai.ui.canvas.node_item import NodeItem
from sciai.ui.controller_thread import RootConfirmer
from sciai.ui.main_window import MainWindow
from sciai.ui.shell import ElidedLabel
from sciai.ui.theme.theme import Theme, load_bundled_fonts
from tests.acceptance.test_phase1 import (
    INTEGRAL_Q,
    INTEGRAL_ROUTE,
    finish_integral,
    retry_integral,
)
from tests.fakes.fake_llm import ScriptedLLM

NODE_TITLE_PX = 13
TITLE_FULL_CHARS = 22


def fault_script() -> ScriptedLLM:
    """The recipe's integral fails its check, the ladder retries it, and the retry passes."""
    return ScriptedLLM([INTEGRAL_ROUTE, retry_integral("meijerg"), finish_integral])


def fault_runner(inner) -> FaultyRunner:  # noqa: ANN001
    return FaultyRunner(inner, "sympy.integrate", "pi**2 - 2", calls={1})


_APP: QApplication | None = None  # keep a reference: a collected QApplication deletes every widget


def setup_app() -> QApplication:
    global _APP
    app = QApplication.instance() or QApplication([])
    _APP = app
    app.setStyle("Fusion")
    load_bundled_fonts()
    app.setStyleSheet(Theme.load("light").qss())
    return app


def pump(app: QApplication, cond: Callable[[], bool], timeout: float = 30.0) -> None:
    end = time.time() + timeout
    while not cond():
        if time.time() > end:
            raise TimeoutError("UI condition not reached")
        app.processEvents()
        time.sleep(0.01)
    for _ in range(10):
        app.processEvents()


def run_fault_window(rt, size: tuple[int, int]) -> MainWindow:  # noqa: ANN001
    """Open the window at ``size``, run the injected-fault task and select the failed derivative
    (the node pill appears; the workspace stays on the Graph tab)."""
    app = setup_app()
    win = MainWindow(rt, Theme.load("light"), RootConfirmer())
    win.resize(*size)
    win.show()
    pump(app, lambda: win.session_id is not None)
    win.chat.input.setPlainText(INTEGRAL_Q)
    win.chat._submit()
    pump(app, lambda: not win.executor.is_busy
         and any(n.type == NodeType.FINAL for n in win.graph.nodes.values()))
    bad = failed_diff(win)
    win._select_and_show(bad.id)
    pump(app, lambda: True)
    return win


def failed_diff(win: MainWindow):  # noqa: ANN201
    return next(n for n in win.graph.nodes.values()
                if n.tool_name == "sympy.integrate" and n.status.value == "failed")


NODE_TABS = ["Code", "Execution Log", "Messages", "Environment", "Review"]


@dataclass
class Metrics:
    window: str
    chat_w: int
    sidebar_w: int
    workspace_w: int
    regions: tuple[int, int, int]
    tab_gaps: list[int]
    chat_widest: bool
    scale: float
    title_px: float
    nodes_visible: int
    composer: list[str]
    pills: list[str]
    node_tabs: list[str]
    live: str
    clipped: list[str] = field(default_factory=list)
    titles_not_full: list[str] = field(default_factory=list)


def _inside_scroll_area(w: QWidget) -> bool:
    p = w.parentWidget()
    while p is not None:
        if isinstance(p, QAbstractScrollArea):
            return True
        p = p.parentWidget()
    return False


def _name(w: QWidget) -> str:
    if isinstance(w, QTabBar):
        text = "/".join(w.tabText(i) for i in range(w.count()))
    else:
        text = w.text() if hasattr(w, "text") and callable(w.text) else ""
    return f"{type(w).__name__}#{w.objectName() or '-'} {text[:40]!r}"


def clipped_text(root: QWidget) -> list[str]:
    """Visible text widgets whose text does not fit, or that a parent cuts off.

    Intentional elision (ElidedLabel, list items, the canvas's own elided text) is not counted.
    Vertical cut-off inside a scroll area is scrolling, not clipping.
    """
    out: list[str] = []
    for w in root.findChildren(QWidget):
        if not w.isVisible() or w.width() <= 0 or isinstance(w, ElidedLabel):
            continue
        need: tuple[int, int] | None = None
        if isinstance(w, QLabel) and w.text():
            if w.wordWrap():
                need = (0, w.heightForWidth(w.width()))
            else:
                need = (w.sizeHint().width(), w.sizeHint().height())
        elif isinstance(w, QAbstractButton) and w.text():
            need = (w.sizeHint().width(), 0)
        elif isinstance(w, QLineEdit):
            need = (0, w.sizeHint().height())
        elif isinstance(w, QTabBar):
            # every tab at its natural size must fit, and none may be pushed past the bar's edge
            need = (w.sizeHint().width(), 0)
            for i in range(w.count()):
                if not w.rect().contains(w.tabRect(i)):
                    out.append(f"QTabBar tab {w.tabText(i)!r} cut")
        else:
            continue
        if need is not None and (need[0] > w.width() + 1 or need[1] > w.height() + 1):
            out.append(f"{_name(w)} needs {need[0]}x{need[1]}, has {w.width()}x{w.height()}")
            continue
        vis: QRect = w.visibleRegion().boundingRect()
        if vis.isEmpty() and _inside_scroll_area(w):
            continue  # scrolled out of view entirely
        if vis.width() < w.width() - 1:
            out.append(f"{_name(w)} cut horizontally ({vis.width()} of {w.width()} px visible)")
        elif vis.height() < w.height() - 1 and not _inside_scroll_area(w):
            out.append(f"{_name(w)} cut vertically ({vis.height()} of {w.height()} px visible)")
    return out


def open_pin_editor(win: MainWindow) -> None:
    """Click the selected node again (as a user would) to drop pin 1 near its right edge."""
    from PyQt6.QtCore import QPointF

    g = win.graph
    item = g.items_[g.selected]
    w, h = item.size()
    g._clicked(item, QPointF(w * 0.8, h * 0.5))


def composer_parts(win: MainWindow) -> list[str]:
    """Which of the reference composer's parts are present and visible."""
    c = win.chat.composer
    parts = {"status strip": c.status_strip, "+": c.attach_btn, "tools": c.tools_btn, "mic": c.mic_btn,
             "send": c.send_btn, "input": c.input}
    found = [name for name, w in parts.items() if w.isVisible() and w.width() > 0 and w.height() > 0]
    if c.input.placeholderText() != "Ask anything":
        found.remove("input")
    return found


def visible_clipping(win: MainWindow) -> list[str]:
    from sciai.ui.chat.inline import TableCard

    out = clipped_text(win)
    for card in win.findChildren(TableCard):
        if card.isVisible():
            out += card.elided()
    return out


def measure(win: MainWindow) -> Metrics:
    """Measure the Graph tab, then the node's Code and Review tabs; clipping is checked in all three."""
    app = setup_app()
    ws = win.workspace
    ws.show_graph()
    pump(app, lambda: True)
    g = win.graph
    scale = g.scale_factor()
    view_rect = g.mapToScene(g.viewport().rect()).boundingRect()
    items = [it for it in g.items_.values() if it.node.type != NodeType.CHECK]
    titles_not_full = [it.node.title for it in items
                       if len(it.node.title) <= TITLE_FULL_CHARS and not it.title_layout()[1]]
    clipped = [f"[graph] {c}" for c in visible_clipping(win)]
    tab_gaps: list[int] = []
    for idx, name in ((0, "code"), (4, "review")):
        ws.show_node_tab()
        ws.node_panel.tabs.setCurrentIndex(idx)
        pump(app, lambda: True)
        clipped += [f"[{name}] {c}" for c in visible_clipping(win)]
        tab_gaps = ws.node_panel.tabs.label_gaps()
    ws.show_graph()
    pump(app, lambda: True)
    # popovers: the running-tools list and a pin note
    win.chat.toggle_tools()
    pump(app, lambda: True)
    clipped += [f"[tools popover] {c}" for c in visible_clipping(win)]
    win.chat.toggle_tools()
    open_pin_editor(win)
    pump(app, lambda: True)
    clipped += [f"[pin popover] {c}" for c in visible_clipping(win)]
    win.graph.pin_editor.hide()
    pump(app, lambda: True)
    chat_w, side_w, ws_w = win.chat.width(), win.sidebar_card.width(), win.workspace_card.width()
    pills = [b.text() for b in (ws.graph_tab, ws.node_tab) if b.isVisible()]
    return Metrics(
        window=f"{win.width()}x{win.height()}",
        chat_w=chat_w, sidebar_w=side_w, workspace_w=ws_w, regions=win.region_widths(),
        tab_gaps=tab_gaps,
        chat_widest=chat_w > max(side_w, ws_w),
        scale=round(scale, 3),
        title_px=round(NODE_TITLE_PX * scale, 2),
        nodes_visible=sum(1 for it in items if view_rect.contains(it.sceneBoundingRect())),
        composer=composer_parts(win),
        pills=pills,
        node_tabs=ws.node_panel.tab_labels(),
        live=ws.live_btn.text(),
        clipped=clipped,
        titles_not_full=titles_not_full,
    )


def title_fit_samples(item: NodeItem, samples: list[str]) -> list[str]:
    """Realistic 22-character titles that would NOT be shown in full on a node."""
    keep = item.node.title
    bad = []
    for s in samples:
        item.node.title = s
        if not item.title_layout()[1]:
            bad.append(s)
    item.node.title = keep
    return bad


TABLE_HEAD = ("| window | regions sidebar / chat / workspace | chat pane | workspace card | tab gaps (min) | "
              "graph scale | node title px | nodes fully in view | composer parts | clipped text |\n"
              "|---|---|---|---|---|---|---|---|---|---|")


def table_row(m: Metrics) -> str:
    s, c, w = m.regions
    W = s + c + w
    pct = f"{s} / {c} / {w} ({100 * s / W:.0f} / {100 * c / W:.0f} / {100 * w / W:.0f}%)"
    return (f"| {m.window} | {pct} | {m.chat_w} | {m.workspace_w} | {min(m.tab_gaps)} | "
            f"{m.scale:.3f} | {m.title_px:.2f} | {m.nodes_visible} | {len(m.composer)}/6 | {len(m.clipped)} |")
