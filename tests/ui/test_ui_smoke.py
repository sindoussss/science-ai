"""UI smoke tests (offscreen): the window builds, a task fills the graph, status styling and pins work."""
from __future__ import annotations

import pytest

pytest.importorskip("pytestqt")

from sciai.graph.model import NodeType, Status  # noqa: E402
from sciai.ui.controller_thread import RootConfirmer  # noqa: E402
from sciai.ui.main_window import MainWindow  # noqa: E402
from sciai.ui.theme.theme import STATUS_ICON, Theme, load_bundled_fonts  # noqa: E402
from tests.acceptance.test_phase1 import DIFF, FORMAL, QUESTION, finish_both, subs_step  # noqa: E402
from tests.fakes.fake_llm import ScriptedLLM  # noqa: E402


@pytest.fixture
def window(qtbot, make_rt, tmp_path):
    rt = make_rt(ScriptedLLM([FORMAL, DIFF, subs_step, finish_both]))
    theme = Theme.load("light")
    from PyQt6.QtWidgets import QApplication
    app = QApplication.instance()
    app.setStyle("Fusion")
    load_bundled_fonts()
    app.setStyleSheet(theme.qss())
    win = MainWindow(rt, theme, RootConfirmer())
    qtbot.addWidget(win)
    win.show()
    qtbot.waitUntil(lambda: win.session_id is not None, timeout=10000)
    yield win
    win.close()


def test_light_theme_is_default_and_tokens_complete():
    light, dark = Theme.load("light"), Theme.load("dark")
    assert light.tokens["name"] == "light"
    assert set(light.tokens["color"]) == set(dark.tokens["color"])
    for status in STATUS_ICON:
        assert f"status_{status}" in light.tokens["color"]


def test_task_fills_graph_and_node_panel(window, qtbot, tmp_path):
    window.chat.input.setPlainText(QUESTION)
    window.chat._submit()
    qtbot.waitUntil(lambda: any(n.type == NodeType.FINAL for n in window.graph.nodes.values()), timeout=30000)
    qtbot.waitUntil(lambda: not window.executor.is_busy, timeout=30000)
    finals = [n for n in window.graph.nodes.values() if n.type == NodeType.FINAL]
    assert finals[0].status == Status.VERIFIED
    assert "-pi**2" in window.chat.transcript_text()
    assert window.node_panel.node is not None
    assert window.node_panel.tab_labels() == ["Code", "Execution Log", "Messages", "Environment", "Review"]
    window.grab().save(str(tmp_path / "window.png"))

    # Pin a note on the derivative node: triggers a deterministic re-check (no model call).
    diff = next(n for n in window.graph.nodes.values() if n.tool_name == "sympy.diff")
    window.graph.pin_editor.open_for(diff.id, "n2", 1, 0.5, 0.5)
    window.graph.pin_editor.text.setPlainText("recheck this derivative")
    window.graph.pin_editor._send()
    qtbot.waitUntil(lambda: "Re-check of" in window.chat.transcript_text(), timeout=30000)
    assert "Note 1 on n2: recheck this derivative" in window.chat.transcript_text()  # a system line
    assert window.graph.items_[diff.id].pins == [(1, 0.5, 0.5)]
    msgs = window.rt.repo.messages(window.session_id, diff.id)
    assert msgs[-1]["pin_number"] == 1


def test_locked_node_cannot_be_deleted(window, qtbot):
    window.chat.input.setPlainText(QUESTION)
    window.chat._submit()
    qtbot.waitUntil(lambda: any(n.type == NodeType.FINAL for n in window.graph.nodes.values()), timeout=30000)
    qtbot.waitUntil(lambda: not window.executor.is_busy, timeout=30000)
    diff = next(n for n in window.graph.nodes.values() if n.tool_name == "sympy.diff")
    window.node_panel.lock_toggled.emit(diff.id, True)
    qtbot.waitUntil(lambda: window.rt.repo.get_node(diff.id).locked, timeout=10000)
    from sciai.graph.engine import GraphRuleError
    with pytest.raises(GraphRuleError):
        window.rt.engine.delete(diff.id)


def test_restyle_single_primary_and_delete_in_menu(window):
    from PyQt6.QtWidgets import QPushButton
    primaries = [b for b in window.findChildren(QPushButton) if b.objectName() == "primary"]
    assert [b.text() for b in primaries] == ["Download script"]
    assert window.node_panel.delete_action.text().startswith("Delete")
    assert not any(b.text() == "Delete" for b in window.node_panel.findChildren(QPushButton))
    from PyQt6.QtCore import Qt
    assert window.graph.horizontalScrollBarPolicy() == Qt.ScrollBarPolicy.ScrollBarAlwaysOff
    assert window.graph.verticalScrollBarPolicy() == Qt.ScrollBarPolicy.ScrollBarAlwaysOff
