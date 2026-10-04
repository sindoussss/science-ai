"""Measured layout checks for the injected-fault scenario at two window sizes (offscreen)."""
from __future__ import annotations

import pytest

pytest.importorskip("PyQt6")

from tests.ui.metrics import (  # noqa: E402
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

    assert m.scale >= 0.92
    assert m.scale <= 1.3
    assert m.title_px >= 12
    assert m.chat_frac <= 0.30
    assert m.canvas_frac >= 0.70
    assert m.sidebar_w == 220
    assert m.inspector_w == 340
    assert m.clipped == []
    assert m.titles_not_full == []
    item = next(it for it in win.graph.items_.values() if it.node.type.value != "check")
    assert title_fit_samples(item, SAMPLE_TITLES_22) == []


def test_fit_button_may_go_below_floor_until_next_layout(fault_window):
    win = fault_window((1200, 720))
    g = win.graph
    g.fit_all()
    assert g.scale_factor() < 0.92  # this graph does not fit at the floor in a 1200px window
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


def test_sidebar_collapses_to_rail_and_back(fault_window):
    from tests.ui.metrics import pump, setup_app

    win = fault_window((1200, 720))
    app = setup_app()
    center_before = win.center_card.width()
    win.sidebar.set_collapsed(True)
    pump(app, lambda: win.sidebar_card.width() == 52)
    assert win.center_card.width() > center_before
    win.sidebar.set_collapsed(False)
    pump(app, lambda: win.sidebar_card.width() == 220)
    assert win.center_card.width() == center_before
