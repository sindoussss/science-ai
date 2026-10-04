"""Right pane: everything about the selected node."""
from __future__ import annotations

import datetime as dt
import json
from typing import Any, Callable

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMenu,
    QPlainTextEdit,
    QPushButton,
    QTabWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from sciai.graph.model import Node, Status
from sciai.store.repository import Repository
from sciai.tools.registry import get as get_tool
from sciai.ui.chat_bar import PRODUCT_NAME
from sciai.ui.inspector.review import ReviewPanel
from sciai.ui.layout_util import clear_layout
from sciai.ui.theme.theme import STATUS_ICON, Theme


def _ts(t: float) -> str:
    return dt.datetime.fromtimestamp(t).strftime("%H:%M:%S")


def node_script(node: Node) -> str:
    if node.tool_name and node.tool_inputs is not None:
        try:
            spec = get_tool(node.tool_name)
            if spec.script is not None:
                return f"# Reproduces node {node.title!r} ({node.tool_name})\n" + spec.script(node.tool_inputs)
        except KeyError:
            pass
        return (f"# Tool call for node {node.title!r}\n# tool: {node.tool_name}\n"
                f"args = {json.dumps(node.tool_inputs, indent=2)}\n")
    return f"# Node {node.title!r} has no tool call\n"


class FlowRow(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.lay = QHBoxLayout(self)
        self.lay.setContentsMargins(0, 0, 0, 0)
        self.lay.setSpacing(6)

    def clear(self) -> None:
        clear_layout(self.lay)


class Inspector(QWidget):
    select_node = pyqtSignal(str)
    lock_toggled = pyqtSignal(str, bool)
    demote = pyqtSignal(str)
    delete = pyqtSignal(str)
    note = pyqtSignal(str, str)

    def __init__(self, repo: Repository, theme: Theme, env_provider: Callable[[], dict[str, Any]]) -> None:
        super().__init__()
        self.setObjectName("cardBody")
        self.repo, self.theme, self.env_provider = repo, theme, env_provider
        self.node: Node | None = None
        self.handles: dict[str, str] = {}
        self._versions: list[Node] = []
        self._vidx = 0

        lay = QVBoxLayout(self)
        pad = theme.space("pad")
        lay.setContentsMargins(pad + 2, pad, pad + 2, pad)
        lay.setSpacing(10)

        # Header: handle + title, version switcher, status chip.
        head = QHBoxLayout()
        head.setSpacing(6)
        self.handle_lbl = QLabel()
        self.handle_lbl.setObjectName("muted")
        self.handle_lbl.setFont(theme.mono_font())
        self.prev_v = QToolButton(text="‹")
        self.next_v = QToolButton(text="›")
        self.version_lbl = QLabel()
        self.version_lbl.setObjectName("muted")
        for b in (self.prev_v, self.next_v):
            b.setObjectName("iconButton")
        self.prev_v.clicked.connect(lambda: self._step_version(-1))
        self.next_v.clicked.connect(lambda: self._step_version(1))
        head.addWidget(self.handle_lbl)
        head.addStretch(1)
        head.addWidget(self.prev_v)
        head.addWidget(self.version_lbl)
        head.addWidget(self.next_v)
        lay.addLayout(head)

        self.title = QLabel("No node selected")
        self.title.setObjectName("nodeTitle")
        self.title.setWordWrap(True)
        lay.addWidget(self.title)

        row = QHBoxLayout()
        row.setSpacing(8)
        self.badge = QLabel()
        self.lock_lbl = QLabel()
        self.lock_lbl.setObjectName("muted")
        row.addWidget(self.badge)
        row.addWidget(self.lock_lbl)
        row.addStretch(1)
        lay.addLayout(row)
        self.warning = QLabel()
        self.warning.setStyleSheet(f"color:{theme.hex('warning')};")
        self.warning.setWordWrap(True)
        self.warning.hide()
        lay.addWidget(self.warning)

        deps_row = QHBoxLayout()
        deps_row.setSpacing(6)
        dep_label = QLabel("Depends on")
        dep_label.setObjectName("muted")
        deps_row.addWidget(dep_label)
        self.deps = FlowRow()
        deps_row.addWidget(self.deps, 1)
        lay.addLayout(deps_row)

        # Actions: one primary, quiet outlines, destructive actions behind "…".
        actions = QHBoxLayout()
        actions.setSpacing(6)
        self.download = QPushButton("Download script")
        self.download.setObjectName("primary")
        self.lock_btn = QPushButton("Lock")
        self.demote_btn = QPushButton("Demote")
        self.more = QToolButton(text="…")
        self.more.setObjectName("iconButton")
        self.more.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        menu = QMenu(self.more)
        self.delete_action = menu.addAction("Delete node…")
        self.more.setMenu(menu)
        actions.addWidget(self.download)
        actions.addWidget(self.lock_btn)
        actions.addWidget(self.demote_btn)
        actions.addStretch(1)
        actions.addWidget(self.more)
        lay.addLayout(actions)
        self.download.clicked.connect(self._download)
        self.lock_btn.clicked.connect(lambda: self.node and self.lock_toggled.emit(self.node.id, not self.node.locked))
        self.demote_btn.clicked.connect(lambda: self.node and self.demote.emit(self.node.id))
        self.delete_action.triggered.connect(lambda: self.node and self.delete.emit(self.node.id))

        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)
        self.code = QPlainTextEdit(readOnly=True)
        self.code.setObjectName("code")
        self.log = QPlainTextEdit(readOnly=True)
        self.log.setObjectName("code")
        chat = QWidget()
        cl = QVBoxLayout(chat)
        cl.setContentsMargins(0, 8, 0, 0)
        cl.setSpacing(8)
        self.chat = QPlainTextEdit(readOnly=True)
        self.chat.setObjectName("code")
        self.chat_input = QLineEdit()
        self.chat_input.setPlaceholderText("Note to the controller about this node")
        self.chat_input.returnPressed.connect(self._send_note)
        cl.addWidget(self.chat, 1)
        cl.addWidget(self.chat_input)
        self.env = QPlainTextEdit(readOnly=True)
        self.env.setObjectName("code")
        self.review = ReviewPanel(theme)
        for w, name in ((self.code, "Code"), (self.log, "Log"), (chat, "Chat"), (self.env, "Env"),
                        (self.review, "Review")):
            self.tabs.addTab(w, name)
        self.tabs.currentChanged.connect(lambda i: self._refresh_env() if self.tabs.tabText(i) == "Env" else None)
        lay.addWidget(self.tabs, 1)
        self._set_enabled(False)

    def _set_enabled(self, on: bool) -> None:
        for b in (self.download, self.lock_btn, self.demote_btn, self.more):
            b.setEnabled(on)

    # ------------------------------------------------------------------ show
    def show_node(self, node: Node | None, handles: dict[str, str]) -> None:
        self.handles = handles
        self.node = node
        self._set_enabled(node is not None)
        if node is None:
            self.title.setText("No node selected")
            self.handle_lbl.clear()
            self.version_lbl.clear()
            self.badge.clear()
            self.badge.setStyleSheet("")
            self.lock_lbl.clear()
            self.warning.hide()
            self.deps.clear()
            for w in (self.code, self.log, self.chat):
                w.clear()
            self.review.clear()
            self._versions = []
            self.prev_v.setEnabled(False)
            self.next_v.setEnabled(False)
            return
        self._versions = self.repo.lineage(node.lineage_id) or [node]
        self._vidx = next((i for i, v in enumerate(self._versions) if v.id == node.id), len(self._versions) - 1)
        self._render(self._versions[self._vidx])

    def _step_version(self, delta: int) -> None:
        i = self._vidx + delta
        if 0 <= i < len(self._versions):
            self._vidx = i
            self._render(self._versions[i])

    def _render(self, node: Node) -> None:
        t = self.theme
        self.node = node
        n = len(self._versions)
        self.version_lbl.setText(f"v{node.version}" + (f" of {n}" if n > 1 else ""))
        self.prev_v.setEnabled(self._vidx > 0)
        self.next_v.setEnabled(self._vidx < n - 1)
        self.handle_lbl.setText(self.handles.get(node.id, "earlier version"))
        self.title.setText(node.title)
        self.title.setToolTip(node.content or node.title)
        st = node.status.value
        self.badge.setText(f"{STATUS_ICON.get(st, '')} {st}")
        self.badge.setStyleSheet(t.chip_css(st))
        self.lock_lbl.setText("🔒 locked" if node.locked else "")
        warn = [w for w in [node.warning, *node.flags] if w]
        self.warning.setText("⚠ " + "; ".join(warn))
        self.warning.setVisible(bool(warn))
        self.lock_btn.setText("Unlock" if node.locked else "Lock")
        self.demote_btn.setEnabled(node.status == Status.VERIFIED)

        self.deps.clear()
        dep_ids = self.repo.dependencies(node.id)
        if not dep_ids:
            none = QLabel("nothing")
            none.setObjectName("muted")
            self.deps.lay.addWidget(none)
        for d in dep_ids:
            chip = QPushButton(self.handles.get(d, d[:6]))
            chip.setObjectName("chip")
            chip.setCursor(Qt.CursorShape.PointingHandCursor)
            chip.clicked.connect(lambda _=False, x=d: self.select_node.emit(x))
            self.deps.lay.addWidget(chip)
        self.deps.lay.addStretch(1)

        self.code.setPlainText(node_script(node) + "\n# Result\n# " + _result_text(node.result))

        lines = []
        for r in self.repo.runs_for(node.id):
            out = r["error"] or r["output"]
            lines.append(f"{_ts(r['created_at'])}  ran {r['tool_name']} in {r['duration_ms']:.0f} ms")
            lines.append(f"          → {_clip(out)}")
        for e in self.repo.status_history(node.id):
            frm = e["from_status"]
            arrow = f"{frm} → {e['to_status']}" if frm else e["to_status"]
            lines.append(f"{_ts(e['created_at'])}  {arrow}  ({e['reason']})")
        self.log.setPlainText("\n".join(lines) or "No log entries.")

        msgs = self.repo.messages(node.session_id, node.id)
        self.chat.setPlainText("\n\n".join(
            f"{_ts(m['created_at'])}  {'You' if m['author'] == 'user' else PRODUCT_NAME}"
            f"{'  · pin ' + str(m['pin_number']) if m['pin_number'] else ''}\n{m['text']}" for m in msgs)
            or "No messages for this node.")

        self.review.show_node(node, self.repo.evidence_for(node.id))
        if self.tabs.tabText(self.tabs.currentIndex()) == "Env":
            self._refresh_env()

    def _refresh_env(self) -> None:
        info = self.env_provider()
        vram = info.get("vram_bytes")
        lines = [f"Model            {info.get('model')}",
                 f"Ollama           {'reachable' if info.get('reachable') else 'not reachable'}",
                 f"Loaded           {'yes' if info.get('loaded') else 'no'}",
                 f"VRAM             {vram / 1024**3:.2f} GB" if vram else "VRAM             unknown",
                 f"Context window   {info.get('num_ctx')} tokens", "", "Tools"]
        lines += [f"  {t}" for t in info.get("tools", [])]
        self.env.setPlainText("\n".join(lines))

    # --------------------------------------------------------------- actions
    def _download(self) -> None:
        if self.node is None:
            return
        name = (self.node.tool_name or "node").replace(".", "_") + ".py"
        path, _ = QFileDialog.getSaveFileName(self, "Download script", name, "Python (*.py)")
        if path:
            with open(path, "w", encoding="utf-8") as f:
                f.write(node_script(self.node))

    def _send_note(self) -> None:
        text = self.chat_input.text().strip()
        if text and self.node is not None:
            self.chat_input.clear()
            self.note.emit(self.node.id, text)


def _clip(v: Any, n: int = 120) -> str:
    s = str(v)
    return s if len(s) <= n else s[: n - 1] + "…"


def _result_text(result: dict[str, Any] | None) -> str:
    if not result:
        return "none"
    if "value" in result:
        return str(result["value"])
    return _clip(json.dumps(result, default=str), 400)


class Hairline(QFrame):
    def __init__(self) -> None:
        super().__init__()
        self.setObjectName("hairline")
