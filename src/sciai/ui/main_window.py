"""Chat-centered main window: sidebar card | chat (flat on the window) | 1px divider | workspace card."""
from __future__ import annotations

import shutil
import threading
from pathlib import Path
from typing import Any, Callable

from PyQt6.QtCore import QPoint, Qt, QTimer
from PyQt6.QtGui import QKeySequence, QShortcut
from PyQt6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QDialog,
    QMainWindow,
    QMenu,
    QMessageBox,
    QVBoxLayout,
    QWidget,
)

from sciai.controller.loop import TaskResult
from sciai.graph.events import Event
from sciai.graph.model import CascadeReport, Node, NodeType
from sciai.runtime import Runtime
from sciai.tools.registry import all_tools
from sciai.ui.chat.inline import Column
from sciai.ui.chat_bar import ChatPane, fmt_elapsed
from sciai.ui.confirm_dialog import ConfirmProblemDialog
from sciai.ui.controller_thread import EngineExecutor, EventBridge, RootConfirmer
from sciai.ui.mathtext import answer_text, prose_text, value_text
from sciai.ui.scrollbars import install_everywhere as install_overlay_scrollbars
from sciai.ui.shell import AppRoot, Card, HairlineSplitter
from sciai.ui.sidebar import FILES_DIR, Sidebar
from sciai.ui.theme.theme import Theme
from sciai.ui.workspace.workspace import Workspace

# Proportions at any window width (Yeri's polish pass): sidebar card 17% (200-240px),
# workspace 40% (440-640px, measured from the divider to the window's right edge), chat the
# rest and never under 440px. Dragging the divider changes the workspace's share.
SIDEBAR_FRACTION, SIDEBAR_MIN_W, SIDEBAR_MAX_W = 0.17, 200, 240
WORKSPACE_FRACTION, WORKSPACE_MIN_W, WORKSPACE_MAX_W = 0.40, 440, 640
CHAT_MIN_W = 440
GUTTER = 12  # window edge to cards, and sidebar card to chat
WORKSPACE_INSET = 10  # the workspace card sits 10px in from the divider
HANDLE_W = 5  # splitter grab area; the 1px divider is drawn at its center
MIN_SIZE = (1200, 720)
# Short check names for the chat table (the Review tab has the long ones).
CHECK_SHORT = {"symbolic_vs_numeric": "numeric", "alt_algorithm": "alt. method", "known_value": "known value",
               "units": "units", "inputs_verified": "inputs"}


def short_args(inputs: dict[str, Any] | None) -> str:
    if not inputs:
        return ""
    parts = []
    for k, v in inputs.items():
        if k in ("template", "answer_nodes"):
            continue
        s = v if isinstance(v, str) else str(v)
        parts.append(f"{k}={s}")
    return ", ".join(parts)


def fmt_duration(ms: float) -> str:
    return f"{ms:.0f} ms" if ms < 1000 else fmt_elapsed(ms / 1000)


class MainWindow(QMainWindow):
    def __init__(self, rt: Runtime, theme: Theme, confirmer: RootConfirmer) -> None:
        super().__init__()
        self.rt, self.theme, self.confirmer = rt, theme, confirmer
        self.setWindowTitle("Science AI")
        self.setMinimumSize(*MIN_SIZE)
        self.resize(1440, 900)
        self.executor = EngineExecutor()
        self.bridge = EventBridge(rt.engine.bus)
        self._callbacks: dict[int, Callable[[Any, Exception | None], None]] = {}
        self._env: dict[str, Any] = {"model": rt.cfg.model.name, "num_ctx": rt.cfg.model.num_ctx,
                                     "tools": [t.name for t in all_tools()]}
        self.session_id: str | None = None
        self._task_nodes: list[str] = []  # nodes added during the running task, in order
        self._task_running = False
        # Runs are attached to their node right after node_added; look the durations up shortly after.
        # Owned by the window (not QTimer.singleShot) so it can be stopped before the database closes.
        self._durations_timer = QTimer(self)
        self._durations_timer.setSingleShot(True)
        self._durations_timer.setInterval(150)
        self._durations_timer.timeout.connect(self._refresh_durations)

        self.sidebar = Sidebar(rt.repo, theme)
        self.chat = ChatPane(theme)
        self.workspace = Workspace(rt.repo, theme, self._env_info)
        self.graph = self.workspace.graph
        self.node_panel = self.workspace.node_panel

        root = AppRoot(theme)
        rl = QHBoxLayout(root)
        rl.setContentsMargins(GUTTER, GUTTER, 0, GUTTER)  # the right gutter belongs to the workspace pane
        rl.setSpacing(GUTTER)
        self.sidebar_card = root.track(Card(self.sidebar))
        self.sidebar_card.setFixedWidth(SIDEBAR_MIN_W)
        rl.addWidget(self.sidebar_card)

        # Chat is flat on the window background; the splitter handle is the 1px divider.
        self.chat.setMinimumWidth(CHAT_MIN_W)
        self.workspace_card = root.track(Card(self.workspace))
        self.workspace_pane = QWidget()
        self.workspace_pane.setObjectName("flat")
        wl = QVBoxLayout(self.workspace_pane)
        wl.setContentsMargins(WORKSPACE_INSET, 0, GUTTER, 0)
        wl.addWidget(self.workspace_card)
        self._ws_fraction = WORKSPACE_FRACTION
        self.workspace_pane.setMinimumWidth(WORKSPACE_MIN_W - self._handle_right())
        self.workspace_pane.setMaximumWidth(WORKSPACE_MAX_W - self._handle_right())
        self.main_split = HairlineSplitter(Qt.Orientation.Horizontal, theme, last_px=480)
        self.main_split.setHandleWidth(HANDLE_W)
        self.main_split.addWidget(self.chat)
        self.main_split.addWidget(self.workspace_pane)
        self.main_split.setStretchFactor(0, 1)
        self.main_split.setStretchFactor(1, 0)
        rl.addWidget(self.main_split, 1)
        self.setCentralWidget(root)
        self.main_split.splitterMoved.connect(self._divider_moved)
        QShortcut(QKeySequence("Ctrl+G"), self, activated=self.toggle_workspace,
                  context=Qt.ShortcutContext.WindowShortcut)

        self.executor.done.connect(self._job_done)
        self.executor.busy.connect(self.chat.set_busy)
        self.bridge.event.connect(self._on_event)
        self.confirmer.ask.connect(self._confirm_root)
        self.chat.submitted.connect(self._submit)
        self.chat.stop.connect(self.rt.controller.stop)
        self.chat.view_graph.connect(self.view_graph)
        self.chat.composer.attach_btn.clicked.connect(self._add_file)
        self.chat.composer.tools_btn.clicked.connect(self._show_tools_menu)
        self.chat.composer.mic_btn.clicked.connect(
            lambda: self.chat.add_system("Voice input isn't available in this local build."))
        self.workspace.stop.connect(self.rt.controller.stop)
        self.graph.node_selected.connect(self._select)
        self.graph.pin_sent.connect(self._pin_sent)
        np = self.node_panel
        np.select_node.connect(self._select_and_show)
        np.lock_toggled.connect(lambda nid, v: self._engine_job(lambda: rt.engine.lock(nid, v), nid))
        np.demote.connect(
            lambda nid: self._engine_job(lambda: rt.engine.demote(nid, automatic=False, reason="demoted by user"), nid))
        np.delete.connect(self._delete)
        np.reject_assumption.connect(self._reject_assumption)
        np.note.connect(self._node_note)
        self.sidebar.new_session.connect(self._new_session)
        self.sidebar.open_session.connect(self._open_session)
        self.sidebar.open_node.connect(lambda sid, nid: self._open_session(sid, select=nid))
        self.sidebar.toggle_workspace.connect(self.toggle_workspace)
        self.sidebar.settings.connect(self._open_environment)
        self.sidebar.add_file.connect(self._add_file)

        install_overlay_scrollbars(self)

        sessions = rt.repo.list_sessions()
        if sessions:
            self._open_session(sessions[0]["id"])
        else:
            self._new_session()
        if rt.flagged_on_start:
            self.chat.add_notice(f"{rt.flagged_on_start} unverified result(s) from earlier sessions are flagged "
                                 "for re-check and will not be reused silently.")
        threading.Thread(target=self._poll_env, daemon=True).start()

    # --------------------------------------------------------------- shell
    @staticmethod
    def _handle_right() -> int:
        """Pixels of the splitter handle right of the divider line (the line is at its center)."""
        return HANDLE_W - HANDLE_W // 2

    def region_widths(self) -> tuple[int, int, int]:
        """(sidebar, chat, workspace) in px, summing to the window width: the sidebar is its card,
        the workspace runs from the divider line to the right edge, the chat is everything between
        (the gutters included). These are the 17% / 43% / 40% regions of the spec."""
        side = self.sidebar_card.width()
        if not self.workspace_pane.isVisible():
            return side, self.width() - side, 0
        handle = self.main_split.handle(1)
        line_x = handle.mapTo(self, handle.rect().topLeft()).x() + HANDLE_W // 2
        return side, line_x - side, self.width() - line_x

    def _apply_proportions(self) -> None:
        w = self.width()
        side = int(max(SIDEBAR_MIN_W, min(SIDEBAR_MAX_W, round(SIDEBAR_FRACTION * w))))
        self.sidebar_card.setFixedWidth(side)
        chat_left = GUTTER + side + GUTTER
        ws = max(WORKSPACE_MIN_W, min(WORKSPACE_MAX_W, round(self._ws_fraction * w)))
        ws = max(WORKSPACE_MIN_W, min(ws, w - chat_left - CHAT_MIN_W))  # the chat keeps 440px
        self.main_split.last_px = ws - self._handle_right()
        self.main_split._apply_px()

    def _divider_moved(self, _pos: int, _index: int) -> None:
        _s, _c, ws = self.region_widths()
        if self.width() > 0 and ws > 0:
            self._ws_fraction = ws / self.width()

    def resizeEvent(self, event) -> None:  # noqa: N802, ANN001
        super().resizeEvent(event)
        self._apply_proportions()

    def toggle_workspace(self) -> None:
        show = not self.workspace_pane.isVisible()
        self.workspace_pane.setVisible(show)
        self.sidebar.workspace_action.setChecked(show)
        if show:
            self._apply_proportions()

    def _open_environment(self) -> None:
        if not self.workspace_pane.isVisible():
            self.toggle_workspace()
        self.workspace.show_environment()

    # ---------------------------------------------------------------- jobs
    def _submit_job(self, fn: Callable[[], Any], cb: Callable[[Any, Exception | None], None] | None = None) -> None:
        job = self.executor.submit(fn)
        if cb is not None:
            self._callbacks[job] = cb

    def _job_done(self, job: int, result: Any, error: Exception | None) -> None:
        cb = self._callbacks.pop(job, None)
        if cb is not None:
            cb(result, error)
        elif error is not None:
            self.chat.add_notice(str(error))

    def _engine_job(self, fn: Callable[[], Any], node_id: str | None = None) -> None:
        def done(_r: Any, err: Exception | None) -> None:
            if err is not None:
                QMessageBox.warning(self, "Not allowed", str(err))
            self.sidebar.refresh(self.session_id)
            if node_id:
                self._show(node_id)
        self._submit_job(fn, done)

    # ------------------------------------------------------------ sessions
    def _new_session(self) -> None:
        def make() -> tuple[str, Any]:
            sid = self.rt.engine.new_session("New session")
            return sid, self.rt.engine.snapshot()
        self._submit_job(make, lambda r, err: self._load_view(*r))

    def _open_session(self, sid: str, select: str | None = None) -> None:
        def open_() -> tuple[str, Any]:
            self.rt.engine.open_session(sid)
            return sid, self.rt.engine.snapshot()
        self._submit_job(open_, lambda r, err: self._load_view(*r, select=select))

    def _session_title(self, sid: str | None) -> str:
        return next((r["title"] for r in self.rt.repo.list_sessions() if r["id"] == sid), "New session")

    def _load_view(self, sid: str, snapshot: Any, select: str | None = None) -> None:
        """Runs on the UI thread with a snapshot taken on the engine thread."""
        self.session_id = sid
        nodes, edges = snapshot
        pins: dict[str, list[tuple[int, float, float]]] = {}
        for n, _ in nodes:
            for m in self.rt.repo.messages(sid, n.id):
                if m["pin_number"]:
                    pins.setdefault(n.id, []).append((m["pin_number"], m["pin_x"], m["pin_y"]))
        self.graph.load(nodes, edges, pins)
        self.chat.clear()
        self.chat.set_title(self._session_title(sid))
        for m in self.rt.repo.messages(sid):
            if m["node_id"] and m["author"] == "user":
                pin = f"Note {m['pin_number']}" if m["pin_number"] else "Note"
                self.chat.add_system(f"{pin} on {self.graph.handles.get(m['node_id'], 'a step')}: {m['text']}")
            elif m["author"] == "user":
                self.chat.add_user(m["text"])
            elif m["author"] == "system":
                self.chat.add_system(m["text"])
            else:
                self.chat.add_assistant(m["text"])
        for notice in self.rt.repo.notices(sid):
            self.chat.add_notice(notice["text"])
        self.rt.repo.mark_notices_seen(sid)
        self.sidebar.refresh(sid)
        self.workspace.show_node(None, self.graph.handles)
        if select and select in self.graph.nodes:
            self._select_and_show(select)
            self.workspace.show_node_tab()

    # --------------------------------------------------------------- chat
    def _submit(self, text: str) -> None:
        if self.executor.is_busy:
            self.rt.controller.steer(text)
            self.chat.add_system(f"Steer: {text}")
            return
        self.chat.add_user(text)
        sid = self.session_id
        if sid and self._session_title(sid) == "New session":
            self.rt.repo.rename_session(sid, text[:60])
            self.chat.set_title(text[:60])
        self.sidebar.refresh(sid)
        self._task_nodes = []
        self._task_running = True
        self.chat.start_run()
        self.workspace.set_live(True)
        self.sidebar.set_running(sid, 0)
        self._submit_job(lambda: self.rt.controller.run(text), self._task_done)

    def _task_done(self, result: TaskResult | None, err: Exception | None) -> None:
        self._task_running = False
        self._refresh_durations()
        self.chat.finish_run()
        self.workspace.set_live(False)
        self.sidebar.set_running(None)
        if err is not None:
            self.chat.add_notice(str(err))
            return
        assert result is not None
        caption = f"{result.llm_calls} model call(s) · {result.steps} step(s)"
        if result.status in ("answered", "reused"):
            self._report_steps(result)
            self.chat.add_answer(self._answer_math(result), result.answer, verified=result.verified,
                                 steps=result.steps, calls=result.llm_calls, reused=result.status == "reused",
                                 final_id=result.final_node, assumptions=result.assumptions)
            if result.final_node:
                self._select_and_show(result.final_node)
        elif result.status == "needs_user":
            self.chat.add_assistant(result.question or "", caption)
        elif result.status == "escalated":
            self._report_steps(result)
            self.chat.add_assistant(f"{result.detail} The conflicting results are outlined in red on the graph.",
                                    caption)
            self.graph.highlight(result.conflict)
        else:
            self.chat.add_assistant(f"Stopped ({result.status}). {result.detail}", caption)
        self.sidebar.refresh(self.session_id)

    def _report_steps(self, result: TaskResult) -> None:
        """Summary line with chips, a table of this task's steps and their checks, and any plots."""
        repo, handles = self.rt.repo, self.graph.handles
        nodes = [n for n in (repo.get_node(i) for i in self._task_nodes) if n is not None]
        steps = [n for n in nodes if n.tool_name and n.type in (NodeType.TOOL_RESULT, NodeType.PLOT)]
        checks = [n for n in nodes if n.type == NodeType.CHECK]
        if not steps:
            return
        tools = sorted({n.tool_name for n in steps if n.tool_name})
        failed = [n for n in steps if n.status.value == "failed"]
        invalid = [n for n in steps if n.status.value == "invalidated"]
        parts = [f"Ran {len(steps)} tool calls ({', '.join(f'`{t}`' for t in tools)})"
                 + (f" and {len(checks)} independent checks." if checks else ".")]

        def names(ns: list[Node]) -> str:
            return ", ".join(handles.get(n.id, "?") for n in ns)

        if failed:
            one = len(failed) == 1
            line = f"{names(failed)} failed {'its' if one else 'their'} check"
            if invalid:
                line += f"; {names(invalid)}, built on {'it' if one else 'them'}, " \
                        f"{'was' if len(invalid) == 1 else 'were'} invalidated"
            parts.append(line + ". The affected steps were redone.")
        if result.verified:
            parts.append("Every value in the answer comes from a verified step.")
        self.chat.add_assistant(" ".join(parts))
        rows = []
        for n in steps:
            ev = repo.evidence_for(n.id)
            check = CHECK_SHORT.get(ev[-1].method, ev[-1].method.replace("_", " ")) if ev else "—"
            rows.append([handles.get(n.id, "?"), n.tool_name or "", value_text(n.display_result()), check,
                         n.status.value])
        self.chat.add_table([Column("step", "code", 40), Column("tool", "code", shrink=True),
                             Column("result", "code", 60, flex=True), Column("check", "text", 40, shrink=True),
                             Column("status", "status")], rows)
        for n in steps:
            if n.result and n.result.get("kind") == "plotspec" and n.status.value not in ("failed", "invalidated"):
                self.chat.add_plot(n.result["value"], f"{handles.get(n.id, '')} · {n.title}")

    def _answer_math(self, result: TaskResult) -> str:
        final = self.rt.repo.get_node(result.final_node) if result.final_node else None
        if final is None:
            return prose_text(result.answer)
        by_handle = {h: nid for nid, h in self.graph.handles.items()}
        return answer_text(final, lambda h: self.rt.repo.get_node(by_handle[h]) if h in by_handle else None)

    def view_graph(self, final_id: str) -> None:
        if not self.workspace_pane.isVisible():
            self.toggle_workspace()
        self.workspace.show_graph()
        ancestors = [a for a in self.rt.repo.ancestors(final_id) if a in self.graph.nodes]
        self.graph.focus_answer(final_id, ancestors)
        self._show(final_id)

    def _node_note(self, node_id: str, text: str) -> None:
        self.rt.repo.add_message(self.session_id or "", "user", text, node_id=node_id, sent=True)
        handle = self.graph.handles.get(node_id, "")
        self.chat.add_system(f"Note on {handle}: {text}")
        self.rt.controller.steer(f"Note on {handle}: {text}")
        self._show(node_id)

    def _pin_sent(self, node_id: str, text: str, number: int, rx: float, ry: float) -> None:
        self.rt.repo.add_message(self.session_id or "", "user", text, node_id=node_id, pin_number=number,
                                 pin_xy=(rx, ry), sent=True)
        handle = self.graph.handles.get(node_id, "")
        self.chat.add_system(f"Note {number} on {handle}: {text}")
        self.rt.controller.steer(f"Pinned note {number} on {handle}: {text}")

        def done(msg: Any, err: Exception | None) -> None:
            if err:
                self.chat.add_notice(str(err))
            else:
                self.chat.add_assistant(str(msg))
            self._show(node_id)
        self._submit_job(lambda: self.rt.controller.recheck(node_id, text), done)

    def _delete(self, node_id: str) -> None:
        handle = self.graph.handles.get(node_id, "")
        if QMessageBox.question(self, "Delete node", f"Delete {handle}? Everything built on it is invalidated.") \
                != QMessageBox.StandardButton.Yes:
            return
        self._engine_job(lambda: self.rt.engine.delete(node_id))

    def _add_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Add a file")
        if not path:
            return
        FILES_DIR.mkdir(parents=True, exist_ok=True)
        dst = FILES_DIR / Path(path).name
        shutil.copy2(path, dst)
        self.chat.add_system(f"Added {dst.name} to Files")

    def _show_tools_menu(self) -> None:
        menu = QMenu(self)
        for t in all_tools():
            a = menu.addAction(t.name)
            a.setEnabled(False)
        btn = self.chat.composer.tools_btn
        menu.exec(btn.mapToGlobal(QPoint(0, -menu.sizeHint().height() - 4)))

    # ------------------------------------------------------------- events
    def _on_event(self, ev: Event) -> None:
        p = ev.payload
        if ev.kind == "node_added" and p["node"].session_id is not None:
            node: Node = p["node"]
            self.graph.add_node(node, p["handle"], p.get("depends_on", []))
            if self._task_running:
                self._task_nodes.append(node.id)
                if node.tool_name and not p.get("reused"):
                    self.chat.add_tool_call(f"{node.tool_name} · {short_args(node.tool_inputs)}", "…", key=node.id)
                    self.sidebar.set_running(self.session_id, self.chat.tool_count())
                    self._durations_timer.start()
        elif ev.kind == "node_updated":
            self.graph.update_node(p["node"])
            cur = self.node_panel.node
            if cur is not None and cur.lineage_id == p["node"].lineage_id:
                self._show(p["node"].id)
        elif ev.kind == "edge_added":
            self.graph.add_edge(p["src"], p["dst"], p["edge_kind"])
        elif ev.kind == "node_removed":
            self.graph.remove_node(p["node_id"])
            if self.node_panel.node is not None and self.node_panel.node.id == p["node_id"]:
                self.workspace.show_node(None, self.graph.handles)
        elif ev.kind == "cascade":
            self._cascade_notice(p["report"])
        elif ev.kind == "thought":
            self.chat.add_thought(p["text"])
        elif ev.kind == "loop_state":
            self.chat.set_step(p["step"])

    def _refresh_durations(self) -> None:
        """Tool runs are attached to their node just after it is added; fill in the real durations."""
        for nid in self._task_nodes:
            runs = self.rt.repo.runs_for(nid)
            if runs:
                self.chat.update_tool_call(nid, fmt_duration(runs[-1]["duration_ms"]))

    def _cascade_notice(self, report: CascadeReport) -> None:
        if report.locked_warnings:
            self.chat.add_notice(f"{len(report.locked_warnings)} locked node(s) were invalidated; they stay "
                                 "in the graph with a warning.")
        if not report.crosses_sessions:
            return
        titles = {r["id"]: r["title"] for r in self.rt.repo.list_sessions()}
        parts = []
        for sid, answers in report.affected_sessions.items():
            what = f"{len(answers)} answer(s)" if answers else "intermediate results"
            parts.append(f"“{titles.get(sid, sid[:8])}” ({what})")
        self.chat.add_notice("This invalidation also affects other sessions: " + ", ".join(parts) + ".")

    # ----------------------------------------------------------- selection
    def _select(self, node_id: str) -> None:
        self._show(node_id)

    def _select_and_show(self, node_id: str) -> None:
        self.graph.select(node_id)
        self._show(node_id)

    def _show(self, node_id: str) -> None:
        node: Node | None = self.rt.repo.get_node(node_id)
        self.workspace.show_node(node, self.graph.handles)

    # ------------------------------------------------------------ dialogs
    def _confirm_root(self, node: Node) -> None:
        dialog = ConfirmProblemDialog(node, self)
        ok = dialog.exec() == QDialog.DialogCode.Accepted
        self.confirmer.answer(dialog.answer() if ok else False)

    def _reject_assumption(self, node_id: str) -> None:
        node = self.rt.repo.get_node(node_id)
        text = f"“{node.content}”" if node is not None else "this assumption"
        if QMessageBox.question(self, "Reject assumption",
                                f"Reject {text}? Every result built on it is invalidated, in every session.") \
                != QMessageBox.StandardButton.Yes:
            return
        self._engine_job(lambda: self.rt.engine.reject_assumption(node_id), node_id)

    # ------------------------------------------------------------------ env
    def _poll_env(self) -> None:
        status = getattr(self.rt.llm, "status", None)
        if callable(status):
            self._env.update(status())

    def _env_info(self) -> dict[str, Any]:
        threading.Thread(target=self._poll_env, daemon=True).start()
        return dict(self._env)

    def closeEvent(self, event) -> None:  # noqa: N802, ANN001
        self._durations_timer.stop()
        self.rt.controller.stop()
        self.confirmer.answer(False)
        self.executor.shutdown()
        self.bridge.close()
        super().closeEvent(event)

