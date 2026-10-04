"""Three-pane main window: sidebar | live graph + chat | inspector."""
from __future__ import annotations

import threading
from typing import Any, Callable

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QInputDialog,
    QMainWindow,
    QMessageBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from sciai.controller.loop import TaskResult
from sciai.graph.events import Event
from sciai.graph.model import CascadeReport, Node
from sciai.runtime import Runtime
from sciai.tools.registry import all_tools
from sciai.ui.canvas.graph_view import GraphView
from sciai.ui.canvas.legend import Legend
from sciai.ui.chat_bar import ChatPane
from sciai.ui.controller_thread import EngineExecutor, EventBridge, RootConfirmer
from sciai.ui.inspector.inspector import Hairline, Inspector
from sciai.ui.mathtext import answer_text, prose_text
from sciai.ui.shell import AppRoot, Card, HairlineSplitter
from sciai.ui.sidebar import Sidebar
from sciai.ui.theme.theme import Theme


class MainWindow(QMainWindow):
    def __init__(self, rt: Runtime, theme: Theme, confirmer: RootConfirmer) -> None:
        super().__init__()
        self.rt, self.theme, self.confirmer = rt, theme, confirmer
        self.setWindowTitle("Science AI")
        self.resize(1440, 880)
        self.executor = EngineExecutor()
        self.bridge = EventBridge(rt.engine.bus)
        self._callbacks: dict[int, Callable[[Any, Exception | None], None]] = {}
        self._env: dict[str, Any] = {"model": rt.cfg.model.name, "num_ctx": rt.cfg.model.num_ctx,
                                     "tools": [t.name for t in all_tools()]}
        self.session_id: str | None = None

        self.sidebar = Sidebar(rt.repo, theme)
        self.graph = GraphView(theme)
        self.chat = ChatPane(theme)
        self.inspector = Inspector(rt.repo, theme, self._env_info)
        top = QWidget()
        top.setObjectName("cardBody")
        tl = QVBoxLayout(top)
        tl.setContentsMargins(0, 0, 0, 0)
        tl.setSpacing(0)
        tl.addWidget(self.graph, 1)
        tl.addWidget(Hairline())
        tl.addWidget(Legend(theme))
        # Canvas column: graph on top, chat below (~25% of the height, resizable).
        self.center_split = HairlineSplitter(Qt.Orientation.Vertical, theme)
        self.center_split.addWidget(top)
        self.center_split.addWidget(self.chat)
        self.center_split.setStretchFactor(0, 3)
        self.center_split.setStretchFactor(1, 1)
        self.center_split.setSizes([3000, 1000])  # proportional: 75 / 25

        gap = theme.space("gap")
        root = AppRoot(theme)
        rl = QHBoxLayout(root)
        rl.setContentsMargins(gap, gap, gap, gap)
        rl.setSpacing(0)
        split = QSplitter(Qt.Orientation.Horizontal)
        split.setHandleWidth(gap)
        split.setChildrenCollapsible(False)
        for pane, min_w in ((self.sidebar, 200), (self.center_split, 480), (self.inspector, 320)):
            card = root.track(Card(pane))
            card.setMinimumWidth(max(min_w, card.minimumSizeHint().width()))  # never below content
            split.addWidget(card)
        split.setSizes([224, 820, 340])
        split.setStretchFactor(1, 1)
        rl.addWidget(split)
        self.setCentralWidget(root)

        self.executor.done.connect(self._job_done)
        self.executor.busy.connect(self.chat.set_busy)
        self.bridge.event.connect(self._on_event)
        self.confirmer.ask.connect(self._confirm_root)
        self.chat.submitted.connect(self._submit)
        self.chat.stop.connect(self.rt.controller.stop)
        self.graph.node_selected.connect(self._select)
        self.graph.pin_sent.connect(self._pin_sent)
        self.inspector.select_node.connect(self._select_and_show)
        self.inspector.lock_toggled.connect(lambda nid, v: self._engine_job(lambda: rt.engine.lock(nid, v), nid))
        self.inspector.demote.connect(
            lambda nid: self._engine_job(lambda: rt.engine.demote(nid, automatic=False, reason="demoted by user"), nid))
        self.inspector.delete.connect(self._delete)
        self.inspector.note.connect(self._node_note)
        self.sidebar.new_session.connect(self._new_session)
        self.sidebar.open_session.connect(self._open_session)
        self.sidebar.open_node.connect(lambda sid, nid: self._open_session(sid, select=nid))

        sessions = rt.repo.list_sessions()
        if sessions:
            self._open_session(sessions[0]["id"])
        else:
            self._new_session()
        if rt.flagged_on_start:
            self.chat.add_notice(f"{rt.flagged_on_start} unverified result(s) from earlier sessions are flagged "
                                 "for re-check and will not be reused silently.")
        threading.Thread(target=self._poll_env, daemon=True).start()

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
        for m in self.rt.repo.messages(sid):
            if m["author"] == "user":
                self.chat.add_user(m["text"])
            elif m["author"] == "system":
                self.chat.add_meta(m["text"])
            else:
                self.chat.add_assistant(m["text"])
        for notice in self.rt.repo.notices(sid):
            self.chat.add_notice(notice["text"])
        self.rt.repo.mark_notices_seen(sid)
        self.sidebar.refresh(sid)
        self.inspector.show_node(None, self.graph.handles)
        if select and select in self.graph.nodes:
            self._select_and_show(select)

    # --------------------------------------------------------------- chat
    def _submit(self, text: str) -> None:
        if self.executor.is_busy:
            self.rt.controller.steer(text)
            self.chat.add_user(text, label="(steer)")
            return
        self.chat.add_user(text)
        sid = self.session_id
        sessions = {r["id"]: r["title"] for r in self.rt.repo.list_sessions()}
        if sid and sessions.get(sid) == "New session":
            self.rt.repo.rename_session(sid, text[:60])
            self.sidebar.refresh(sid)
        self._submit_job(lambda: self.rt.controller.run(text), self._task_done)

    def _task_done(self, result: TaskResult | None, err: Exception | None) -> None:
        if err is not None:
            self.chat.add_notice(str(err))
            return
        assert result is not None
        caption = f"{result.llm_calls} model call(s) · {result.steps} step(s)"
        if result.status in ("answered", "reused"):
            self.chat.add_answer(self._answer_math(result), result.answer, verified=result.verified,
                                 steps=result.steps, calls=result.llm_calls, reused=result.status == "reused")
            if result.final_node:
                self._select_and_show(result.final_node)
        elif result.status == "needs_user":
            self.chat.add_assistant(result.question or "", caption)
        elif result.status == "escalated":
            self.chat.add_assistant(f"{result.detail} The conflicting results are outlined in red on the graph.",
                                    caption)
            self.graph.highlight(result.conflict)
        else:
            self.chat.add_assistant(f"Stopped ({result.status}). {result.detail}", caption)
        self.sidebar.refresh(self.session_id)

    def _answer_math(self, result: TaskResult) -> str:
        final = self.rt.repo.get_node(result.final_node) if result.final_node else None
        if final is None:
            return prose_text(result.answer)
        by_handle = {h: nid for nid, h in self.graph.handles.items()}
        return answer_text(final, lambda h: self.rt.repo.get_node(by_handle[h]) if h in by_handle else None)

    def _node_note(self, node_id: str, text: str) -> None:
        self.rt.repo.add_message(self.session_id or "", "user", text, node_id=node_id, sent=True)
        self.rt.controller.steer(f"Note on {self.graph.handles.get(node_id, '')}: {text}")
        self._show(node_id)

    def _pin_sent(self, node_id: str, text: str, number: int, rx: float, ry: float) -> None:
        self.rt.repo.add_message(self.session_id or "", "user", text, node_id=node_id, pin_number=number,
                                 pin_xy=(rx, ry), sent=True)
        handle = self.graph.handles.get(node_id, "")
        self.chat.add_user(text, label=f"(note {number} on {handle})")
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

    # ------------------------------------------------------------- events
    def _on_event(self, ev: Event) -> None:
        p = ev.payload
        if ev.kind == "node_added" and p["node"].session_id is not None:
            self.graph.add_node(p["node"], p["handle"], p.get("depends_on", []))
        elif ev.kind == "node_updated":
            self.graph.update_node(p["node"])
            if self.inspector.node is not None and self.inspector.node.lineage_id == p["node"].lineage_id:
                self._show(p["node"].id)
        elif ev.kind == "edge_added":
            self.graph.add_edge(p["src"], p["dst"], p["edge_kind"])
        elif ev.kind == "node_removed":
            self.graph.remove_node(p["node_id"])
            if self.inspector.node is not None and self.inspector.node.id == p["node_id"]:
                self.inspector.show_node(None, self.graph.handles)
        elif ev.kind == "cascade":
            self._cascade_notice(p["report"])
        elif ev.kind == "thought":
            self.chat.add_thought(p["text"])

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
        self.inspector.show_node(node, self.graph.handles)

    # ------------------------------------------------------------ dialogs
    def _confirm_root(self, node: Node) -> None:
        text, ok = QInputDialog.getMultiLineText(
            self, "Confirm the problem",
            "This is how the question was formalized. Edit it if needed, then confirm.", node.content)
        self.confirmer.answer(text if ok else False)

    # ------------------------------------------------------------------ env
    def _poll_env(self) -> None:
        status = getattr(self.rt.llm, "status", None)
        if callable(status):
            self._env.update(status())

    def _env_info(self) -> dict[str, Any]:
        threading.Thread(target=self._poll_env, daemon=True).start()
        return dict(self._env)

    def closeEvent(self, event) -> None:  # noqa: N802, ANN001
        self.rt.controller.stop()
        self.confirmer.answer(False)
        self.executor.shutdown()
        self.bridge.close()
        super().closeEvent(event)
