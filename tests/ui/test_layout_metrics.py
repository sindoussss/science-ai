"""Measured layout checks for the injected-fault scenario at two window sizes (offscreen)."""
from __future__ import annotations

import pytest

pytest.importorskip("PyQt6")

from tests.ui.metrics import (  # noqa: E402
    NODE_TABS,
    TABLE_HEAD,
    measure,
    table_row,
    title_fit_samples,
)

SIZES = [(1200, 720), (1440, 900)]


def expected_sidebar(w: int) -> int:
    return max(200, min(240, round(0.17 * w)))


def expected_workspace(w: int) -> int:
    return max(440, min(640, round(0.40 * w)))
SAMPLE_TITLES_22 = ["Second derivative test", "Integrate over [0, pi]", "Solve quadratic system",
                    "Mean free path of N2 g", "Differentiate x^2 sinx", "Taylor series at x = 0"]
_rows: list[str] = []


@pytest.mark.parametrize("size", SIZES, ids=[f"{w}x{h}" for w, h in SIZES])
def test_layout_metrics(fault_window, size):
    win = fault_window(size)
    m = measure(win)
    _rows.append(table_row(m))
    print("\n" + TABLE_HEAD + "\n" + "\n".join(_rows))
    if m.clipped:
        print("clipped:\n  " + "\n  ".join(m.clipped))

    side, chat, ws = m.regions
    W = size[0]
    assert side + chat + ws == W
    assert side == expected_sidebar(W)
    assert abs(ws - expected_workspace(W)) <= 1
    assert chat >= 440 and m.chat_w >= 440
    assert m.tab_gaps == [18, 18, 18, 18]
    assert 0.92 <= m.scale <= 1.3
    assert m.title_px >= 12
    assert m.composer == ["status strip", "+", "tools", "mic", "send", "input"]
    assert m.pills == ["Graph", "n2 · the integral"]
    assert m.node_tabs == NODE_TABS
    assert m.live in ("Live", "Idle")
    assert m.clipped == []
    assert m.titles_not_full == []
    item = next(it for it in win.graph.items_.values() if it.node.type.value != "check")
    assert title_fit_samples(item, SAMPLE_TITLES_22) == []


def test_fit_button_may_go_below_floor_until_next_layout(fault_window):
    from tests.ui.metrics import pump, setup_app

    win = fault_window((1200, 720))
    g = win.graph
    g.setFixedHeight(240)  # a short canvas: this graph no longer fits at the floor
    pump(setup_app(), lambda: g.height() == 240)
    g._rebuild()
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
    pump(app, lambda: win.workspace_pane.isVisible() and win.region_widths()[2] == expected_workspace(1200))
    assert win.chat.width() == chat_before


def test_workspace_drag_is_clamped_to_440_640_and_kept_on_resize(fault_window):
    from tests.ui.metrics import pump, setup_app

    win = fault_window((1440, 900))
    app = setup_app()
    sp = win.main_split
    total = sum(sp.sizes())
    sp.moveSplitter(total - 900, 1)  # try to make the workspace far too wide
    pump(app, lambda: True)
    assert win.region_widths()[2] <= 640
    sp.moveSplitter(total - 100, 1)  # and far too narrow
    pump(app, lambda: True)
    assert win.region_widths()[2] >= 440
    tabs = win.node_panel.tabs
    win.workspace.show_node_tab()
    pump(app, lambda: True)
    assert min(tabs.label_gaps()) >= 12  # narrowest workspace: labels still apart and unelided
    assert [tabs.buttons[i].text() for i in tabs.visible_indexes()] == NODE_TABS
    sp.moveSplitter(total - 560, 1)
    pump(app, lambda: True)
    share = win.region_widths()[2] / win.width()
    win.resize(1300, 800)
    pump(app, lambda: win.width() == 1300)
    assert abs(win.region_widths()[2] - round(share * 1300)) <= 1  # the dragged share is kept
    assert win.region_widths()[1] >= 440


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
