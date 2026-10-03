"""Right pane: everything about the selected node."""
from __future__ import annotations

import datetime as dt
import json
from typing import Any, Callable

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from sciai.graph.model import Node, Status
from sciai.store.repository import Repository
from sciai.tools.registry import get as get_tool
from sciai.ui.theme.theme import STATUS_ICON, Theme

LADDER = ("retried", "backtracked", "escalated")


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
        while self.lay.count():
            w = self.lay.takeAt(0).widget()
            if w is not None:
                w.deleteLater()


class Inspector(QWidget):
    select_node = pyqtSignal(str)
    lock_toggled = pyqtSignal(str, bool)
    demote = pyqtSignal(str)
    delete = pyqtSignal(str)
    note = pyqtSignal(str, str)

    def __init__(self, repo: Repository, theme: Theme, env_provider: Callable[[], dict[str, Any]]) -> None:
        super().__init__()
        self.setObjectName("inspector")
        self.repo, self.theme, self.env_provider = repo, theme, env_provider
        self.node: Node | None = None
        self.handles: dict[str, str] = {}
        self._versions: list[Node] = []

        lay = QVBoxLayout(self)
        lay.setContentsMargins(14, 12, 14, 12)
        lay.setSpacing(8)
        head = QHBoxLayout()
        self.title = QLabel("No node selected")
        self.title.setObjectName("nodeTitle")
        self.title.setWordWrap(True)
        self.versions = QComboBox()
        self.versions.setFixedWidth(64)
        self.versions.currentIndexChanged.connect(self._version_changed)
        head.addWidget(self.title, 1)
        head.addWidget(self.versions)
        lay.addLayout(head)

        row = QHBoxLayout()
        self.badge = QLabel()
        self.warning = QLabel()
        self.warning.setStyleSheet(f"color:{theme.hex('warning')};")
        self.warning.setWordWrap(True)
        row.addWidget(self.badge)
        row.addWidget(self.warning, 1)
        lay.addLayout(row)

        dep_label = QLabel("Depends on")
        dep_label.setObjectName("muted")
        lay.addWidget(dep_label)
        self.deps = FlowRow()
        lay.addWidget(self.deps)

        self.tabs = QTabWidget()
        self.code = QPlainTextEdit(readOnly=True)
        self.code.setObjectName("mono")
        self.log = QPlainTextEdit(readOnly=True)
        self.log.setObjectName("mono")
        chat = QWidget()
        cl = QVBoxLayout(chat)
        cl.setContentsMargins(0, 6, 0, 0)
        self.chat = QPlainTextEdit(readOnly=True)
        self.chat_input = QLineEdit()
        self.chat_input.setPlaceholderText("Note to the controller about this node")
        self.chat_input.returnPressed.connect(self._send_note)
        cl.addWidget(self.chat)
        cl.addWidget(self.chat_input)
        self.env = QPlainTextEdit(readOnly=True)
        self.env.setObjectName("mono")
        self.review = QPlainTextEdit(readOnly=True)
        self.review.setObjectName("mono")
        for w, name in ((self.code, "Code"), (self.log, "Log"), (chat, "Chat"), (self.env, "Env"),
                        (self.review, "Review")):
            self.tabs.addTab(w, name)
        self.tabs.currentChanged.connect(lambda i: self._refresh_env() if self.tabs.tabText(i) == "Env" else None)
        lay.addWidget(self.tabs, 1)

        actions = QHBoxLayout()
        self.download = QPushButton("Download script")
        self.lock_btn = QPushButton("Lock")
        self.demote_btn = QPushButton("Demote")
        self.delete_btn = QPushButton("Delete")
        for b in (self.download, self.lock_btn, self.demote_btn, self.delete_btn):
            actions.addWidget(b)
            b.setEnabled(False)
        lay.addLayout(actions)
        self.download.clicked.connect(self._download)
        self.lock_btn.clicked.connect(lambda: self.node and self.lock_toggled.emit(self.node.id, not self.node.locked))
        self.demote_btn.clicked.connect(lambda: self.node and self.demote.emit(self.node.id))
        self.delete_btn.clicked.connect(lambda: self.node and self.delete.emit(self.node.id))

    # ------------------------------------------------------------------ show
    def show_node(self, node: Node | None, handles: dict[str, str]) -> None:
        self.handles = handles
        self.node = node
        for b in (self.download, self.lock_btn, self.demote_btn, self.delete_btn):
            b.setEnabled(node is not None)
        if node is None:
            self.title.setText("No node selected")
            self.badge.clear()
            self.warning.clear()
            self.deps.clear()
            for w in (self.code, self.log, self.chat, self.review):
                w.clear()
            self.versions.blockSignals(True)
            self.versions.clear()
            self.versions.blockSignals(False)
            return
        self._versions = self.repo.lineage(node.lineage_id) or [node]
        self.versions.blockSignals(True)
        self.versions.clear()
        for v in self._versions:
            self.versions.addItem(f"v{v.version}", v.id)
        idx = next((i for i, v in enumerate(self._versions) if v.id == node.id), len(self._versions) - 1)
        self.versions.setCurrentIndex(idx)
        self.versions.blockSignals(False)
        self._render(self._versions[idx] if self._versions else node)

    def _version_changed(self, idx: int) -> None:
        if 0 <= idx < len(self._versions):
            self._render(self._versions[idx])

    def _render(self, node: Node) -> None:
        t = self.theme
        self.node = node
        handle = self.handles.get(node.id, node.id[:8])
        self.title.setText(f"{handle} · {node.title}")
        st = node.status.value
        self.badge.setText(f"{STATUS_ICON.get(st, '')} {st}")
        border = "dashed" if node.status == Status.INVALIDATED else "solid"
        self.badge.setStyleSheet(f"color:{t.hex('status_' + st)}; background:{t.hex('status_' + st + '_bg')};"
                                 f"border:1px {border} {t.hex('status_' + st)}; border-radius:9px; padding:1px 8px;"
                                 "font-weight:600;")
        warn = [w for w in [node.warning, *node.flags] if w]
        self.warning.setText("⚠ " + "; ".join(warn) if warn else "")
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
            chip.clicked.connect(lambda _=False, x=d: self.select_node.emit(x))
            self.deps.lay.addWidget(chip)
        self.deps.lay.addStretch(1)

        call = {"tool": node.tool_name, "args": node.tool_inputs} if node.tool_name else {"content": node.content}
        self.code.setPlainText(json.dumps(call, indent=2, default=str) + "\n\n# Result\n" +
                               json.dumps(node.result, indent=2, default=str) + "\n\n" + node_script(node))

        lines = []
        for r in self.repo.runs_for(node.id):
            lines.append(f"[{_ts(r['created_at'])}] {r['tool_name']} ({r['duration_ms']:.0f} ms)")
            lines.append(f"  in : {r['inputs']}")
            lines.append(f"  out: {r['error'] or r['output']}")
        for e in self.repo.status_history(node.id):
            lines.append(f"[{_ts(e['created_at'])}] {e['from_status'] or '-'} -> {e['to_status']}: {e['reason']}")
        self.log.setPlainText("\n".join(lines) or "No log entries.")

        msgs = self.repo.messages(node.session_id, node.id)
        self.chat.setPlainText("\n".join(
            f"[{_ts(m['created_at'])}] {m['author']}{' (pin ' + str(m['pin_number']) + ')' if m['pin_number'] else ''}: "
            f"{m['text']}" for m in msgs) or "No messages for this node.")

        stage = node.ladder_stage.value
        ladder = " -> ".join(f"[{s}]" if s == stage else s for s in LADDER)
        rev = [f"Status: {st}", f"Ladder: {ladder}" + ("   (not on the ladder)" if stage == "none" else "")]
        fired = node.risk.get("fired", [])
        rev.append("Risk rules fired:" if fired else "Risk rules fired: none")
        rev += [f"  - {f['rule']}: {f['reason']}" for f in fired]
        if node.risk.get("pending_checks"):
            rev.append(f"Pending checks: {', '.join(node.risk['pending_checks'])}")
        if node.risk.get("unverifiable"):
            rev.append("No independent check exists for this tool.")
        ev = self.repo.evidence_for(node.id)
        rev.append("Evidence:" if ev else "Evidence: none")
        for e in ev:
            rev.append(f"  - {e.outcome.upper()} via {e.method} ({e.tool_name}) at {_ts(e.created_at)}")
            detail = {k: v for k, v in e.detail.items() if k not in ("kind", "outcome")}
            if detail:
                rev.append(f"      {json.dumps(detail, default=str)[:400]}")
        self.review.setPlainText("\n".join(rev))
        if self.tabs.tabText(self.tabs.currentIndex()) == "Env":
            self._refresh_env()

    def _refresh_env(self) -> None:
        info = self.env_provider()
        vram = info.get("vram_bytes")
        lines = [f"Model: {info.get('model')}",
                 f"Ollama: {'reachable' if info.get('reachable') else 'not reachable'}",
                 f"Loaded: {info.get('loaded', False)}",
                 f"VRAM: {vram / 1024**3:.2f} GB" if vram else "VRAM: unknown",
                 f"Context window (num_ctx): {info.get('num_ctx')}", "", "Tools:"]
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


class Hairline(QFrame):
    def __init__(self) -> None:
        super().__init__()
        self.setObjectName("hairline")
