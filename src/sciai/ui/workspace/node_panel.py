"""Node tab of the workspace, structured like the reference's notebook cell view.

Header: "[n2]" cell tag, tool chip, status chip | version stepper "‹ v2 ›", download, close.
Sub-tabs: Code | Execution Log | Messages | Environment | Review.
"""
from __future__ import annotations

import datetime as dt
import json
from typing import Any, Callable

from PyQt6.QtCore import QSize, Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMenu,
    QPushButton,
    QScrollArea,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from sciai.graph.model import Node, Status
from sciai.store.repository import Repository
from sciai.tools.registry import get as get_tool
from sciai.ui.chat_bar import PRODUCT_NAME
from sciai.ui.icons import icon
from sciai.ui.layout_util import FlowLayout, clear_layout
from sciai.ui.theme.theme import STATUS_ICON, Theme
from sciai.ui.workspace.code_view import CodeView
from sciai.ui.workspace.review import ReviewPanel
from sciai.ui.workspace.subtabs import SubTabs

TAB_NAMES = ("Code", "Execution Log", "Messages", "Environment", "Review")


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


def _clip(v: Any, n: int = 160) -> str:
    s = str(v)
    return s if len(s) <= n else s[: n - 1] + "…"


def _result_text(result: dict[str, Any] | None) -> str:
    if not result:
        return "No output."
    if "value" in result:
        return str(result["value"])
    return _clip(json.dumps(result, default=str), 600)


def _scroll(body: QWidget) -> QScrollArea:
    sa = QScrollArea()
    sa.setWidgetResizable(True)
    sa.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    sa.setWidget(body)
    return sa


def _secondary(text: str, wrap: bool = False) -> QLabel:
    lbl = QLabel(text)
    lbl.setObjectName("secondary")
    lbl.setWordWrap(wrap)
    return lbl


class NodePanel(QWidget):
    select_node = pyqtSignal(str)
    lock_toggled = pyqtSignal(str, bool)
    demote = pyqtSignal(str)
    delete = pyqtSignal(str)
    note = pyqtSignal(str, str)
    closed = pyqtSignal()

    def __init__(self, repo: Repository, theme: Theme, env_provider: Callable[[], dict[str, Any]]) -> None:
        super().__init__()
        self.setObjectName("cardBody")
        self.repo, self.theme, self.env_provider = repo, theme, env_provider
        self.node: Node | None = None
        self.handles: dict[str, str] = {}
        self._versions: list[Node] = []
        self._vidx = 0
        t = theme
        ink, muted = t.hex("text"), t.hex("text_secondary")

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)

        head_w = QWidget()
        head = QHBoxLayout(head_w)
        head.setContentsMargins(16, 8, 10, 4)
        head.setSpacing(6)
        self.cell_tag = QLabel()
        self.cell_tag.setObjectName("cellTag")
        self.tool_chip = QLabel()
        self.tool_chip.setObjectName("toolChip")
        self.badge = QLabel()
        head.addWidget(self.cell_tag)
        head.addWidget(self.tool_chip)
        head.addWidget(self.badge)
        head.addStretch(1)

        def icon_btn(name: str, tip: str, color: str = muted) -> QToolButton:
            b = QToolButton()
            b.setObjectName("iconButton")
            b.setIcon(icon(name, color))
            b.setIconSize(QSize(14, 14))
            b.setToolTip(tip)
            return b

        self.prev_v = icon_btn("chevron_left", "Previous version", ink)
        self.version_lbl = QLabel()
        self.version_lbl.setObjectName("versionTag")
        self.next_v = icon_btn("chevron_right", "Next version", ink)
        self.download_icon = icon_btn("download", "Download script")
        self.close_btn = icon_btn("close", "Close node")
        self.prev_v.clicked.connect(lambda: self._step_version(-1))
        self.next_v.clicked.connect(lambda: self._step_version(1))
        self.download_icon.clicked.connect(self._download)
        self.close_btn.clicked.connect(self.closed.emit)
        for w in (self.prev_v, self.version_lbl, self.next_v):
            head.addWidget(w)
        head.addSpacing(6)
        head.addWidget(self.download_icon)
        head.addWidget(self.close_btn)
        lay.addWidget(head_w)

        self.warning = QLabel()
        self.warning.setStyleSheet(f"color:{t.hex('warning')}; padding: 0 16px 6px 16px;")
        self.warning.setWordWrap(True)
        self.warning.hide()
        lay.addWidget(self.warning)

        self.tabs = SubTabs(t)
        lay.addWidget(self.tabs, 1)

        # Code ---------------------------------------------------------------------------
        code_page = QWidget()
        cl = QVBoxLayout(code_page)
        cl.setContentsMargins(0, 12, 0, 0)
        cl.setSpacing(0)
        top = QHBoxLayout()
        top.setContentsMargins(16, 0, 16, 8)
        self.download = QPushButton("Download script")
        self.download.setObjectName("primary")
        self.download.setIcon(icon("download", t.hex("accent_text")))
        self.download.setIconSize(QSize(14, 14))
        self.download.clicked.connect(self._download)
        self.download.ensurePolished()
        self.download.setMinimumWidth(self.download.sizeHint().width())
        top.addWidget(self.download)
        top.addStretch(1)
        cl.addLayout(top)
        inputs = QWidget()
        inputs.setObjectName("inputsRow")
        il = QHBoxLayout(inputs)
        il.setContentsMargins(16, 8, 16, 8)
        il.setSpacing(10)
        in_lbl = _secondary("Inputs")
        il.addWidget(in_lbl, 0, Qt.AlignmentFlag.AlignTop)
        self.inputs_host = QWidget()
        self.inputs = FlowLayout(self.inputs_host, spacing=6)
        il.addWidget(self.inputs_host, 1)
        cl.addWidget(inputs)
        self.code = CodeView(t)
        cl.addWidget(self.code, 1)
        self.output_toggle = QToolButton()
        self.output_toggle.setObjectName("link")
        self.output_toggle.setText("output")
        self.output_toggle.setCheckable(True)
        self.output_toggle.setChecked(True)
        self.output_toggle.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.output_toggle.setIconSize(QSize(12, 12))
        self.output_toggle.toggled.connect(self._output_toggled)
        trow = QHBoxLayout()
        trow.setContentsMargins(12, 6, 12, 0)
        trow.addWidget(self.output_toggle)
        trow.addStretch(1)
        cl.addLayout(trow)
        self.output = QLabel()
        self.output.setObjectName("outputText")
        self.output.setWordWrap(True)
        self.output.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.output_box = QWidget()
        ob = QVBoxLayout(self.output_box)
        ob.setContentsMargins(16, 2, 16, 12)
        ob.addWidget(self.output)
        cl.addWidget(self.output_box)
        self._output_toggled(True)

        # Execution Log ------------------------------------------------------------------
        log_body = QWidget()
        self.log_box = QVBoxLayout(log_body)
        self.log_box.setContentsMargins(16, 12, 16, 12)
        self.log_box.setSpacing(2)
        # Messages -----------------------------------------------------------------------
        msg_page = QWidget()
        ml = QVBoxLayout(msg_page)
        ml.setContentsMargins(0, 0, 0, 10)
        msg_body = QWidget()
        self.msg_box = QVBoxLayout(msg_body)
        self.msg_box.setContentsMargins(16, 12, 16, 12)
        self.msg_box.setSpacing(10)
        ml.addWidget(_scroll(msg_body), 1)
        self.chat_input = QLineEdit()
        self.chat_input.setPlaceholderText("Note to the controller about this node")
        self.chat_input.returnPressed.connect(self._send_note)
        wrap = QHBoxLayout()
        wrap.setContentsMargins(12, 0, 12, 0)
        wrap.addWidget(self.chat_input)
        ml.addLayout(wrap)
        # Environment --------------------------------------------------------------------
        env_body = QWidget()
        self.env_grid = QGridLayout(env_body)
        self.env_grid.setContentsMargins(16, 12, 16, 12)
        self.env_grid.setHorizontalSpacing(16)
        self.env_grid.setVerticalSpacing(8)
        # Review -------------------------------------------------------------------------
        review_page = QWidget()
        rl = QVBoxLayout(review_page)
        rl.setContentsMargins(16, 12, 12, 0)
        rl.setSpacing(0)
        actions = QHBoxLayout()
        actions.setSpacing(6)
        self.lock_btn = QPushButton("Lock")
        self.demote_btn = QPushButton("Demote")
        for b in (self.lock_btn, self.demote_btn):
            b.setObjectName("outline")
        self.more = QToolButton()
        self.more.setObjectName("iconButton")
        self.more.setIcon(icon("more", muted))
        self.more.setToolTip("More actions")
        self.more.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        menu = QMenu(self.more)
        self.delete_action = menu.addAction("Delete node…")
        self.more.setMenu(menu)
        actions.addWidget(self.lock_btn)
        actions.addWidget(self.demote_btn)
        actions.addStretch(1)
        actions.addWidget(self.more)
        rl.addLayout(actions)
        self.review = ReviewPanel(t)
        rl.addWidget(self.review, 1)
        self.lock_btn.clicked.connect(lambda: self.node and self.lock_toggled.emit(self.node.id, not self.node.locked))
        self.demote_btn.clicked.connect(lambda: self.node and self.demote.emit(self.node.id))
        self.delete_action.triggered.connect(lambda: self.node and self.delete.emit(self.node.id))

        for w, name in ((code_page, "Code"), (_scroll(log_body), "Execution Log"), (msg_page, "Messages"),
                        (_scroll(env_body), "Environment"), (review_page, "Review")):
            self.tabs.addTab(w, name)
        self.tabs.currentChanged.connect(
            lambda i: self._refresh_env() if self.tabs.tabText(i) == "Environment" else None)
        self._set_enabled(False)

    # ------------------------------------------------------------------ helpers
    def _output_toggled(self, on: bool) -> None:
        self.output_box.setVisible(on)
        self.output_toggle.setIcon(icon("chevron_down" if on else "chevron_right", self.theme.hex("text_secondary")))

    def _set_enabled(self, on: bool) -> None:
        for b in (self.download, self.download_icon, self.lock_btn, self.demote_btn, self.more):
            b.setEnabled(on)

    def tab_labels(self) -> list[str]:
        return [self.tabs.tabText(i) for i in range(self.tabs.count())]

    # --------------------------------------------------------------------- show
    def show_node(self, node: Node | None, handles: dict[str, str]) -> None:
        self.handles = handles
        self.node = node
        self._set_enabled(node is not None)
        if node is None:
            self.cell_tag.clear()
            self.tool_chip.hide()
            self.badge.clear()
            self.badge.setStyleSheet("")
            self.version_lbl.clear()
            self.warning.hide()
            clear_layout(self.inputs)
            self.code.clear()
            self.output.clear()
            for box in (self.log_box, self.msg_box):
                clear_layout(box)
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
        self.version_lbl.setText(f"v{node.version}")
        self.version_lbl.setToolTip(f"version {self._vidx + 1} of {n}")
        self.prev_v.setEnabled(self._vidx > 0)
        self.next_v.setEnabled(self._vidx < n - 1)
        self.cell_tag.setText(f"[{self.handles.get(node.id, '—')}]")
        self.cell_tag.setToolTip(node.title)
        ns = (node.tool_name or "").split(".")[0]
        self.tool_chip.setText(ns)
        self.tool_chip.setVisible(bool(ns))
        st = node.status.value
        self.badge.setText(f"{STATUS_ICON.get(st, '')} {st}")
        self.badge.setStyleSheet(t.chip_css(st))
        warn = [w for w in [node.warning, *node.flags] if w]
        if node.locked:
            warn.insert(0, "locked")
        self.warning.setText("⚠ " + "; ".join(warn))
        self.warning.setVisible(bool(warn))
        self.lock_btn.setText("Unlock" if node.locked else "Lock")
        self.demote_btn.setEnabled(node.status == Status.VERIFIED)

        clear_layout(self.inputs)
        dep_ids = self.repo.dependencies(node.id)
        if not dep_ids:
            self.inputs.addWidget(_secondary("none"))
        for d in dep_ids:
            chip = QPushButton(self.handles.get(d, d[:6]))
            chip.setObjectName("chip")
            chip.setCursor(Qt.CursorShape.PointingHandCursor)
            dep = self.repo.get_node(d)
            chip.setToolTip(dep.title if dep else d)
            chip.clicked.connect(lambda _=False, x=d: self.select_node.emit(x))
            self.inputs.addWidget(chip)
        self.inputs_host.updateGeometry()

        self.code.setPlainText(node_script(node))
        self.output.setText(_result_text(node.result))

        clear_layout(self.log_box)
        rows = 0
        for r in self.repo.runs_for(node.id):
            out = r["error"] or r["output"]
            self._log_row(r["created_at"], f"ran `{r['tool_name']}` in {r['duration_ms']:.0f} ms", _clip(out))
            rows += 1
        for e in self.repo.status_history(node.id):
            frm = e["from_status"]
            arrow = f"{frm} → {e['to_status']}" if frm else e["to_status"]
            self._log_row(e["created_at"], arrow, e["reason"] or "")
            rows += 1
        if not rows:
            self.log_box.addWidget(_secondary("No tool calls or status changes yet."))
        self.log_box.addStretch(1)

        clear_layout(self.msg_box)
        msgs = self.repo.messages(node.session_id, node.id)
        for m in msgs:
            who = "You" if m["author"] == "user" else PRODUCT_NAME
            pin = f" · note {m['pin_number']}" if m["pin_number"] else ""
            head = _secondary(f"{_ts(m['created_at'])}  {who}{pin}")
            body = QLabel(m["text"])
            body.setWordWrap(True)
            body.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            box = QVBoxLayout()
            box.setSpacing(2)
            box.addWidget(head)
            box.addWidget(body)
            self.msg_box.addLayout(box)
        if not msgs:
            self.msg_box.addWidget(_secondary("No messages or pinned notes for this node.", wrap=True))
        self.msg_box.addStretch(1)

        evidence = self.repo.evidence_for(node.id)
        self.review.show_node(node, evidence)
        outcomes = {e.outcome for e in evidence}
        if "fail" in outcomes:
            mark, color = "x_circle", t.hex("status_failed")
        elif "pass" in outcomes:
            mark, color = "check_circle", t.hex("status_verified")
        else:
            mark, color = "dash_circle", t.hex("text_faint")
        self.tabs.setTabIcon(4, icon(mark, color))
        if self.tabs.tabText(self.tabs.currentIndex()) == "Environment":
            self._refresh_env()

    def _log_row(self, when: float, what: str, detail: str) -> None:
        from sciai.ui.chat.inline import InlineText

        row = QHBoxLayout()
        row.setSpacing(10)
        ts = QLabel(_ts(when))
        ts.setObjectName("monoSecondary")
        row.addWidget(ts, 0, Qt.AlignmentFlag.AlignTop)
        col = QVBoxLayout()
        col.setSpacing(0)
        col.addWidget(InlineText(self.theme, what, size_key="size_ui_px"))
        if detail:
            d = QLabel(detail)
            d.setObjectName("secondary")
            d.setWordWrap(True)
            col.addWidget(d)
        row.addLayout(col, 1)
        self.log_box.addLayout(row)
        self.log_box.addSpacing(6)

    def _refresh_env(self) -> None:
        info = self.env_provider()
        clear_layout(self.env_grid)
        vram = info.get("vram_bytes")
        rows = [("Model", str(info.get("model"))),
                ("Ollama", "reachable" if info.get("reachable") else "not reachable"),
                ("Loaded", "yes" if info.get("loaded") else "no"),
                ("VRAM", f"{vram / 1024**3:.2f} GB" if vram else "unknown"),
                ("Context window", f"{info.get('num_ctx')} tokens")]
        for i, (k, v) in enumerate(rows):
            self.env_grid.addWidget(_secondary(k), i, 0, Qt.AlignmentFlag.AlignTop)
            val = QLabel(v)
            val.setWordWrap(True)
            self.env_grid.addWidget(val, i, 1)
        self.env_grid.addWidget(_secondary("Tools"), len(rows), 0, Qt.AlignmentFlag.AlignTop)
        host = QWidget()
        flow = FlowLayout(host, spacing=4)
        for name in info.get("tools", []):
            chip = QLabel(name)
            chip.setObjectName("codeChip")
            flow.addWidget(chip)
        self.env_grid.addWidget(host, len(rows), 1)
        self.env_grid.setColumnStretch(1, 1)
        self.env_grid.setRowStretch(len(rows) + 1, 1)

    # ------------------------------------------------------------------ actions
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
