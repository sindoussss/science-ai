"""Phase 2 UI: the confirm dialog's checklist, assumption nodes, the answer card's assumptions,
and the reject action, run end to end on the projectile problem."""
from __future__ import annotations

import pytest

pytest.importorskip("pytestqt")

from sciai.domains.physics.assumptions import DEFAULTS  # noqa: E402
from sciai.graph.model import Node, NodeType, Status  # noqa: E402
from sciai.ui.canvas.node_item import assumption_chip  # noqa: E402
from sciai.ui.confirm_dialog import ConfirmProblemDialog  # noqa: E402
from sciai.ui.controller_thread import RootConfirmer  # noqa: E402
from sciai.ui.main_window import MainWindow  # noqa: E402
from sciai.ui.theme.theme import Theme, load_bundled_fonts  # noqa: E402
from tests.acceptance.test_phase2 import (  # noqa: E402
    CONST_G,
    PROJECTILE_FORMAL,
    PROJECTILE_Q,
    finish_range,
    range_step,
)
from tests.fakes.fake_llm import ScriptedLLM  # noqa: E402


def root_node(checklist, givens=None, unsourced=()):
    return Node(session_id="s", layer="reasoning", type=NodeType.PROBLEM, title="Problem", content="statement",
                tool_inputs={"checklist": checklist, "givens": givens or {}, "unsourced_givens": list(unsourced)})


def test_confirm_dialog_returns_unticked_assumptions(qtbot):
    items = [{"text": t, "source": "default"} for t in DEFAULTS["projectile"]] + [{"text": "no spin",
                                                                                     "source": "model"}]
    dlg = ConfirmProblemDialog(root_node(items, {"v0": {"value": 20, "unit": "m/s", "kind": "speed"}}, ["v0"]))
    qtbot.addWidget(dlg)
    assert [b.isChecked() for b in dlg.boxes] == [True] * 4  # every item starts ticked
    assert dlg.boxes[3].text().endswith("suggested")
    dlg.boxes[0].setChecked(False)
    dlg.statement.setPlainText("edited")
    assert dlg.answer() == {"statement": "edited", "rejected": ["no air resistance"]}


def test_assumption_chip_labels():
    n = Node(session_id="s", layer="reasoning", type=NodeType.ASSUMPTION, title="Assumption", content="x")
    assert assumption_chip(n) == ("inconclusive", "assumed")
    n.status = Status.FAILED
    assert assumption_chip(n) == ("failed", "rejected")


@pytest.fixture
def window(qtbot, make_rt):
    rt = make_rt(ScriptedLLM([PROJECTILE_FORMAL, CONST_G, range_step, finish_range]))
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


def test_projectile_in_the_window(window, qtbot, monkeypatch, tmp_path):
    window.chat.input.setPlainText(PROJECTILE_Q)
    window.chat._submit()
    qtbot.waitUntil(lambda: any(n.type == NodeType.FINAL for n in window.graph.nodes.values()), timeout=30000)
    qtbot.waitUntil(lambda: not window.executor.is_busy, timeout=30000)
    text = window.chat.transcript_text()
    assert "35.3" in text and "Assuming: no air resistance; constant g = 9.80665 m/s^2" in text
    assumed = [n for n in window.graph.nodes.values() if n.type == NodeType.ASSUMPTION]
    assert len(assumed) == 3 and all(window.graph.items_[n.id] for n in assumed)
    window.grab().save(str(tmp_path / "projectile.png"))
    import os
    if os.environ.get("SCIAI_SHOT_DIR"):
        window.graph.grab().save(os.path.join(os.environ["SCIAI_SHOT_DIR"], "graph.png"))
        window.grab().save(os.path.join(os.environ["SCIAI_SHOT_DIR"], "window.png"))

    # The node panel offers Reject (not Demote) on an assumption; rejecting invalidates the answer.
    air = next(n for n in assumed if n.content == "no air resistance")
    window._select_and_show(air.id)
    panel = window.node_panel
    assert not panel.reject_btn.isHidden() and panel.demote_btn.isHidden()
    from PyQt6.QtWidgets import QMessageBox
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.Yes)
    panel.reject_btn.click()
    qtbot.waitUntil(lambda: window.rt.repo.get_node(air.id).status == Status.FAILED, timeout=10000)
    qtbot.waitUntil(lambda: not window.executor.is_busy, timeout=10000)
    final = next(n for n in window.graph.nodes.values() if n.type == NodeType.FINAL)
    qtbot.waitUntil(lambda: window.graph.nodes[final.id].status == Status.INVALIDATED, timeout=10000)
    assert not panel.reject_btn.isEnabled()
