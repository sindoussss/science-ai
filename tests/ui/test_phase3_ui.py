"""Phase 3 UI: data files dropped or added are imported, the Files menu lists them with Remove,
the Data and Plot tabs, plots drawn by matplotlib in the chat, the graph and the Plot tab, and
assumption nodes that read "doubtful". Run end to end with a scripted model and the real sandbox."""
from __future__ import annotations

import os
import time

import pytest

pytest.importorskip("pytestqt")

from PyQt6.QtCore import QMimeData, QPoint, QPointF, Qt, QUrl  # noqa: E402
from PyQt6.QtGui import QDropEvent  # noqa: E402
from PyQt6.QtWidgets import QApplication, QMessageBox  # noqa: E402


from sciai.graph.model import NodeType, Status  # noqa: E402
from sciai.ui import main_window as mw  # noqa: E402
from sciai.ui import sidebar as sb  # noqa: E402
from sciai.ui.canvas.node_item import assumption_chip, assumption_source  # noqa: E402
from sciai.ui.chat.inline import PlotCard  # noqa: E402
from sciai.ui.controller_thread import RootConfirmer  # noqa: E402
from sciai.ui.main_window import MainWindow  # noqa: E402
from sciai.ui.theme.theme import Theme, load_bundled_fonts  # noqa: E402
from sciai.ui.workspace.data_view import export_plot  # noqa: E402
from tests.acceptance.test_phase3 import FORMAL, FORMAL_GOAL, finish, write_trial  # noqa: E402
from tests.fakes.fake_llm import ScriptedLLM, last_handle  # noqa: E402

SHOTS = os.environ.get("SCIAI_SHOT_DIR")


@pytest.fixture
def cfg(cfg, tmp_path):
    cfg.data.data_dir = str(tmp_path / "datasets")
    return cfg


@pytest.fixture
def trial(tmp_path):
    return write_trial(tmp_path / "trial.csv")


@pytest.fixture
def files_dir(tmp_path, monkeypatch):
    d = tmp_path / "files"
    monkeypatch.setattr(sb, "FILES_DIR", d)
    monkeypatch.setattr(mw, "FILES_DIR", d)
    return d


@pytest.fixture
def open_window(qtbot, make_rt, files_dir):
    wins = []

    def open_(llm, size=(1440, 900)):
        rt = make_rt(llm)
        theme = Theme.load("light")
        app = QApplication.instance()
        app.setStyle("Fusion")
        load_bundled_fonts()
        app.setStyleSheet(theme.qss())
        win = MainWindow(rt, theme, RootConfirmer())
        qtbot.addWidget(win)
        win.resize(*size)
        win.show()
        qtbot.waitUntil(lambda: win.session_id is not None, timeout=10000)
        wins.append(win)
        return win

    yield open_
    for w in wins:
        w.close()


def drop(win, target, paths):
    """A drop on a child widget, handed to the window's application-level filter the way Qt hands
    it a real one (Qt discards synthetic drag events when no drag is in progress, so they can't
    be sent through the event loop offscreen). Returns (taken, event)."""
    mime = QMimeData()
    mime.setUrls([QUrl.fromLocalFile(str(p)) for p in paths])
    ev = QDropEvent(QPointF(10, 10), Qt.DropAction.CopyAction, mime, Qt.MouseButton.NoButton,
                    Qt.KeyboardModifier.NoModifier)
    ev.ignore()
    return win._drops.eventFilter(target, ev), ev


def ask(win, qtbot, text):
    win.chat.input.setPlainText(text)
    win.chat._submit()
    qtbot.waitUntil(lambda: not win.executor.is_busy and not win._task_running, timeout=60000)


def node(win, tool):
    return next(n for n in sorted(win.graph.nodes.values(), key=lambda n: n.created_at) if n.tool_name == tool)


def test_dropped_files_import_data_and_copy_the_rest(open_window, qtbot, trial, tmp_path, files_dir):
    win = open_window(ScriptedLLM([]))
    note = tmp_path / "notes.txt"
    note.write_text("lab notes")
    text_drag = QMimeData()
    text_drag.setText("plain text")
    ev = QDropEvent(QPointF(10, 10), Qt.DropAction.CopyAction, text_drag, Qt.MouseButton.NoButton,
                    Qt.KeyboardModifier.NoModifier)
    assert win._drops.eventFilter(win.chat.input.viewport(), ev) is False  # text drops reach the composer
    taken, ev = drop(win, win.chat.input.viewport(), [trial, note])  # files onto the composer: added
    assert taken and ev.isAccepted()
    qtbot.waitUntil(lambda: not win.executor.is_busy, timeout=30000)
    text = win.chat.transcript_text()
    assert "Imported trial.csv: 40 rows × 7 columns. It was read twice, by pandas and by the Python csv " \
           "module, and both reads agree." in text
    assert win.chat.input.toPlainText() == ""
    assert (files_dir / "notes.txt").read_text() == "lab notes"
    data, others = win.sidebar.files_menu_entries()
    assert [sb.dataset_label(r) for r in data] == ["trial.csv  ·  40 × 7"] and [p.name for p in others] == ["notes.txt"]
    # the import line is kept with the session
    assert any("Imported trial.csv" in m["text"] for m in win.rt.repo.messages(win.session_id))

    assert drop(win, win.graph.viewport(), [trial])[0]  # the same bytes again, dropped on the graph
    qtbot.waitUntil(lambda: not win.executor.is_busy, timeout=30000)
    assert "trial.csv is already imported (40 rows × 7 columns); these are the same bytes" in \
        win.chat.transcript_text()

    empty = tmp_path / "empty.csv"
    empty.write_text("")
    win.add_paths([str(empty)])
    qtbot.waitUntil(lambda: not win.executor.is_busy, timeout=30000)
    assert "notice: empty.csv was not imported: " in win.chat.transcript_text()


def test_a_question_typed_during_an_import_is_a_task_not_a_steer(open_window, qtbot, trial, monkeypatch):
    win = open_window(ScriptedLLM([FORMAL_GOAL]))
    real = win.rt.import_file

    def slow(*a, **k):
        time.sleep(0.6)
        return real(*a, **k)

    monkeypatch.setattr(win.rt, "import_file", slow)
    win.add_paths([str(trial)])
    assert win.executor.is_busy and not win.chat.composer.busy  # an import shows no Stop button
    ask(win, qtbot, "In trial.csv, is the score different between groups A and B?")
    text = win.chat.transcript_text()
    assert "Steer:" not in text
    assert text.index("Imported trial.csv") < text.index("Welch t-test")  # queued behind the import


def test_data_tab_assumptions_and_remove(open_window, qtbot, trial, monkeypatch, tmp_path):
    measured = {}
    win = open_window(ScriptedLLM([FORMAL_GOAL]))
    win.add_paths([str(trial)])
    ask(win, qtbot, "In trial.csv, is the score different between groups A and B?")
    test = node(win, "stats.ttest")
    win._select_and_show(test.id)
    win.workspace.show_node_tab()
    panel = win.node_panel
    assert panel.tab_labels() == ["Code", "Data", "Execution Log", "Messages", "Environment", "Review"]
    panel.show_tab("Data")
    view = panel.data_view
    qtbot.waitUntil(lambda: view.preview_shown() == 40, timeout=20000)
    assert view.name.text() == "trial.csv" and view.shape.text().startswith("40 rows × 7 columns · csv · sha ")
    assert [r[0] for r in view.schema.rows] == ["id", "group", "score [points]", "dose [mg]", "response", "site",
                                                "sex"]
    assert view.schema.rows[1][4] == "A, B" and view.schema.rows[2][2] == "" and view.schema.rows[3][2] == "mg"  # "points" is not a unit
    assert view.preview_label.text() == "All 40 rows"
    assert view.table.item(0, 1).text() == "A"
    tabs = panel.tabs
    sp = win.main_split
    for target in (900, 100):  # the widest and the narrowest workspace (640 and 440 px)
        sp.moveSplitter(sum(sp.sizes()) - target, 1)
        qtbot.wait(30)
        measured[target] = (win.region_widths()[2], tabs.row.width(), tabs.label_gaps(), tabs.side,
                            tabs.needed_width(12), [tabs.buttons[i].text() for i in tabs.visible_indexes()])
        assert min(tabs.label_gaps()) >= 12
        for i in tabs.visible_indexes():  # six tabs, none elided or clipped
            assert tabs.buttons[i].text() == (tabs.labels[i] if target == 900 else tabs.short.get(i, tabs.labels[i]))
            assert tabs.buttons[i].width() >= tabs.buttons[i].sizeHint().width()
        assert tabs.label_rect(tabs.visible_indexes()[-1]).right() < tabs.row.width()
    print("tab row at 640 / 440:", measured)

    # the dataset node itself also has the Data tab; a plain step does not
    dataset = win.rt.repo.get_node(test.tool_inputs["dataset"])
    win._select_and_show(dataset.id)
    assert "Data" in panel.tab_labels() and "Plot" not in panel.tab_labels()
    assert tabs.tabText(tabs.currentIndex()) == "Data"  # stays on Data between data nodes
    final = next(n for n in win.graph.nodes.values() if n.type == NodeType.FINAL)
    win._select_and_show(final.id)
    assert "Data" not in panel.tab_labels() and tabs.tabText(tabs.currentIndex()) == "Code"

    # diagnostics read as assumptions from a diagnostic, or stated and not tested
    assumed = [n for n in win.graph.nodes.values() if n.type == NodeType.ASSUMPTION]
    assert sorted(assumption_source(a) for a in assumed) == ["diagnostic", "diagnostic", "stated, not tested"]
    assert all(assumption_chip(a) == ("inconclusive", "assumed") for a in assumed)
    if SHOTS:
        win._select_and_show(test.id)
        panel.show_tab("Data")
        qtbot.waitUntil(lambda: view.preview_shown() == 40, timeout=20000)
        win.grab().save(os.path.join(SHOTS, "f-node-data-1440x900.png"))
        win.workspace.show_graph()
        qtbot.wait(50)
        win.grab().save(os.path.join(SHOTS, "g-graph-data-1440x900.png"))

    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.Yes)
    win._remove_dataset("trial.csv")
    qtbot.waitUntil(lambda: "Removed trial.csv" in win.chat.transcript_text(), timeout=20000)
    assert "Removed trial.csv. Everything built on it was invalidated." in win.chat.transcript_text()
    qtbot.waitUntil(lambda: win.graph.nodes[test.id].status == Status.INVALIDATED, timeout=5000)
    assert win.sidebar.files_menu_entries()[0] == []


def test_fit_plot_in_chat_graph_and_plot_tab(open_window, qtbot, trial, tmp_path):
    regress = {"action": "call_tool", "tool": "stats.regression", "title": "response on dose",
               "args": {"dataset": "trial.csv", "y": "response", "x": ["dose"]}}

    def plot(prompt):
        h = last_handle(prompt, "stats.regression")
        return {"action": "call_tool", "tool": "plot.fit", "title": "response vs dose",
                "args": {"dataset": "trial.csv", "node": h}}

    win = open_window(ScriptedLLM([FORMAL, regress, plot, finish("stats.regression", text="Fit: ")]))
    win.add_paths([str(trial)])
    ask(win, qtbot, "How does response depend on dose in trial.csv?")
    qtbot.waitUntil(lambda: any(c.isVisible() for c in win.chat.findChildren(PlotCard)), timeout=5000)
    cards = [c for c in win.chat.findChildren(PlotCard) if c.isVisible()]
    assert len(cards) == 1
    card = cards[0]
    chat = win.chat  # the card sits in the chat's one content column, like the table and answer cards
    assert card.mapTo(chat, QPoint(0, 0)).x() == chat.pad() and card.width() == chat.width() - 2 * chat.pad()
    qtbot.waitUntil(lambda: card.view.image() is not None, timeout=5000)
    img = card.view.image()
    assert abs(img.width() / img.devicePixelRatio() - card.view.width()) <= 1
    assert card.view.value["title"] == ""  # the card caption is the title
    assert card.title.endswith("response vs dose")

    fig = node(win, "plot.fit")
    win._select_and_show(fig.id)
    win.workspace.show_node_tab()
    panel = win.node_panel
    assert panel.tab_labels() == ["Code", "Plot", "Execution Log", "Messages", "Environment", "Review"]
    panel.show_tab("Plot")
    qtbot.wait(50)
    view = panel.plot_tab.view
    assert view.image() is not None and view.width() >= 380 and view.height() >= 300
    assert panel.plot_tab.summary.text().startswith("response")
    # a resize re-renders once the size settles, not on every step
    old = view.image()
    view.resize(view.width() - 40, view.height())  # delivered at once to a visible widget
    assert view._settle.isActive() and view.image() is old
    qtbot.waitUntil(lambda: view.image() is not old, timeout=10000)
    win.resize(1300, 860)
    qtbot.waitUntil(lambda: abs(view.image().width() / view.image().devicePixelRatio() - view.width()) <= 1,
                    timeout=10000)

    png, svg = tmp_path / "fit.png", tmp_path / "fit.svg"
    export_plot(fig.result["value"], str(png))
    export_plot(fig.result["value"], str(svg))
    assert png.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n" and b"<svg" in svg.read_bytes()[:400]

    item = win.graph.items_[fig.id]  # the graph node's thumbnail
    assert item.size()[1] == 112
    if SHOTS:
        bar = win.chat.scroll.verticalScrollBar()
        bar.setValue(bar.maximum())
        qtbot.wait(200)
        win.grab().save(os.path.join(SHOTS, "h-node-plot-1300x860.png"))
        win.workspace.show_graph()
        qtbot.wait(50)
        win.grab().save(os.path.join(SHOTS, "i-graph-plot-1300x860.png"))


def test_doubtful_assumption_reads_doubtful(open_window, qtbot, tmp_path):
    import numpy as np

    path = tmp_path / "skew.csv"
    rng = np.random.default_rng(3)
    rows = [f"{g},{v:.4f}" for g, v in [("A", x) for x in rng.exponential(1, 12) ** 3] +
            [("B", x) for x in rng.exponential(1, 12) ** 3 + 2]]
    path.write_text("group,score\n" + "\n".join(rows) + "\n")
    goal = {"tool": "stats.ttest", "args": {"dataset": "skew.csv", "column": "score", "by": "group"}}
    win = open_window(ScriptedLLM([{**FORMAL_GOAL, "goal": goal}]))
    win.add_paths([str(path)])
    ask(win, qtbot, "Is the score different between groups in skew.csv?")
    doubtful = [n for n in win.graph.nodes.values() if n.type == NodeType.ASSUMPTION
                and assumption_chip(n)[1] == "doubtful"]
    assert doubtful
    from sciai.ui.canvas.node_item import body_prefix

    assert all("flagged" not in body_prefix(n, "n1") for n in doubtful)  # the chip says it once
    win._select_and_show(doubtful[0].id)
    assert win.node_panel.warning.text().startswith("Note: locked; doubtful: ")
    assert "Caution: score is roughly normal in" in win.chat.transcript_text()
