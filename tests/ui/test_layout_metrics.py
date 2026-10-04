"""Measured layout checks for the injected-fault scenario at two window sizes (offscreen)."""
from __future__ import annotations

import pytest

pytest.importorskip("PyQt6")

from tests.ui.metrics import (  # noqa: E402
    NODE_TABS,
    TABLE_HEAD,
    fault_runner,
    fault_script,
    measure,
    run_fault_window,
    table_row,
    title_fit_samples,
)

SIZES = [(1200, 720), (1440, 900)]
SAMPLE_TITLES_22 = ["Second derivative test", "Integrate over [0, pi]", "Solve quadratic system",
                    "Mean free path of N2 g", "Differentiate x^2 sinx", "Taylor series at x = 0"]
_rows: list[str] = []


@pytest.fixture
def fault_window(make_rt, runner):
    wins = []

    def open_(size):
        rt = make_rt(fault_script(), run=fault_runner(runner))
        win = run_fault_window(rt, size)
        wins.append(win)
        return win

    yield open_
    for w in wins:
        w.close()


@pytest.mark.parametrize("size", SIZES, ids=[f"{w}x{h}" for w, h in SIZES])
def test_layout_metrics(fault_window, size):
    win = fault_window(size)
    m = measure(win)
    _rows.append(table_row(m))
    print("\n" + TABLE_HEAD + "\n" + "\n".join(_rows))
    if m.clipped:
        print("clipped:\n  " + "\n  ".join(m.clipped))

    assert m.sidebar_w == 220
    assert 340 <= m.workspace_w <= 640
    assert m.workspace_w == 420  # default
    assert m.chat_w >= 440
    assert m.chat_widest
    assert 0.92 <= m.scale <= 1.3
    assert m.title_px >= 12
    assert m.composer == ["status strip", "+", "tools", "mic", "send", "input"]
    assert m.pills == ["Graph", "n2 · f'(x)"]
    assert m.node_tabs == NODE_TABS
    assert m.live in ("Live", "Idle")
    assert m.clipped == []
    assert m.titles_not_full == []
    item = next(it for it in win.graph.items_.values() if it.node.type.value != "check")
    assert title_fit_samples(item, SAMPLE_TITLES_22) == []


def test_fit_button_may_go_below_floor_until_next_layout(fault_window):
    from tests.ui.metrics import pump, setup_app

    win = fault_window((1200, 720))
    win.main_split.last_px = 340 + 12  # narrowest workspace: this graph no longer fits at the floor
    win.main_split._apply_px()
    pump(setup_app(), lambda: win.workspace_card.width() == 340)
    g = win.graph
    assert g.scale_factor() >= 0.92
    g.fit_all()
    assert g.scale_factor() < 0.92
    g.add_edge(*next((s, d, k) for s, d, k in g.edges))  # duplicate: no layout
    assert g.scale_factor() < 0.92
    g._rebuild()  # next layout
    assert g.scale_factor() >= 0.92


def test_wheel_never_zooms_out_past_floor(fault_window):
    from PyQt6.QtCore import QPoint, QPointF, Qt
    from PyQt6.QtGui import QWheelEvent

    win = fault_window((1440, 900))
    g = win.graph
    for _ in range(10):
        ev = QWheelEvent(QPointF(50, 50), QPointF(50, 50), QPoint(0, 0), QPoint(0, -120),
                         Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier, Qt.ScrollPhase.NoScrollPhase,
                         False)
        g.wheelEvent(ev)
    assert g.scale_factor() >= 0.92 - 1e-9


def test_clipping_detector_catches_real_clipping():
    from PyQt6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QWidget

    from tests.ui.metrics import clipped_text, setup_app

    setup_app()
    host = QWidget()
    lay = QHBoxLayout(host)
    btn = QPushButton("Download script")
    btn.setFixedWidth(60)
    ok = QLabel("fits")
    lay.addWidget(btn)
    lay.addWidget(ok)
    host.resize(300, 60)
    host.show()
    found = clipped_text(host)
    host.close()
    assert len(found) == 1 and "Download script" in found[0]


def test_ctrl_g_collapses_the_workspace_and_chat_takes_the_room(fault_window):
    from PyQt6.QtCore import Qt
    from PyQt6.QtTest import QTest

    from tests.ui.metrics import pump, setup_app

    win = fault_window((1200, 720))
    app = setup_app()
    chat_before = win.chat.width()
    QTest.keyClick(win, Qt.Key.Key_G, Qt.KeyboardModifier.ControlModifier)
    pump(app, lambda: not win.workspace_pane.isVisible())
    pump(app, lambda: win.chat.width() > chat_before + 300)
    QTest.keyClick(win, Qt.Key.Key_G, Qt.KeyboardModifier.ControlModifier)
    pump(app, lambda: win.workspace_pane.isVisible() and win.workspace_card.width() == 420)
    assert win.chat.width() == chat_before


def test_workspace_drag_is_clamped_to_340_640_and_kept_on_resize(fault_window):
    from tests.ui.metrics import pump, setup_app

    win = fault_window((1440, 900))
    app = setup_app()
    sp = win.main_split
    total = sum(sp.sizes())
    sp.moveSplitter(total - 900, 1)  # try to make the workspace far too wide
    pump(app, lambda: True)
    assert win.workspace_card.width() <= 640
    sp.moveSplitter(total - 100, 1)  # and far too narrow
    pump(app, lambda: True)
    assert win.workspace_card.width() >= 340
    sp.moveSplitter(total - 520, 1)
    pump(app, lambda: True)
    kept = win.workspace_card.width()
    win.resize(1300, 800)
    pump(app, lambda: win.width() == 1300)
    assert win.workspace_card.width() == kept  # the chat absorbs window resizes
    assert win.chat.width() >= 440


def test_view_graph_focuses_the_answer(fault_window):
    from tests.ui.metrics import pump, setup_app

    win = fault_window((1200, 720))
    app = setup_app()
    final = next(n for n in win.graph.nodes.values() if n.type.value == "final")
    win.workspace.show_node_tab()
    win.view_graph(final.id)
    pump(app, lambda: True)
    assert win.workspace.current_tab() == "Graph"
    assert win.graph.selected == final.id
    dimmed = [nid for nid, it in win.graph.items_.items() if it.opacity() < 1]
    assert dimmed  # the first, failed derivative is not part of the answer
    assert all(win.graph.items_[a].opacity() == 1 for a in win.rt.repo.ancestors(final.id)
               if a in win.graph.items_)
