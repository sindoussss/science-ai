"""The chat region: session title, a flat thread (no assistant bubbles) and the composer card.

Thread blocks: user bubbles (right), assistant text with inline code chips, table
cards, plot cards, a collapsed "Thinking · N steps" row, the final answer card and
small system lines. The composer has a status strip that opens the tools popover.
"""
from __future__ import annotations

import time

from PyQt6.QtCore import QEvent, QObject, QSize, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QKeyEvent, QMouseEvent
from PyQt6.QtWidgets import (
    QApplication,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from sciai.ui.canvas.pins import add_soft_shadow
from sciai.ui.chat.inline import Column, InlineText, PlotCard, TableCard
from sciai.ui.icons import icon
from sciai.ui.layout_util import clear_layout
from sciai.ui.shell import CapsLabel, ElidedLabel
from sciai.ui.theme.theme import Theme

PRODUCT_NAME = "Science AI"
COLUMN_MAX_W = 720
SIDE_PAD = 24


def fmt_elapsed(seconds: float) -> str:
    s = int(round(seconds))
    return f"{s // 60}m {s % 60}s" if s >= 60 else f"{s}s"


def _plural(n: int, word: str) -> str:
    return f"{n} {word}" if n == 1 else f"{n} {word}s"


class Bubble(QFrame):
    """User message: right-aligned soft bubble."""

    MAX_W = 520

    def __init__(self, theme: Theme, text: str) -> None:
        super().__init__()
        self.setObjectName("bubbleUser")
        self.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Preferred)
        self.setMaximumWidth(self.MAX_W)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(14, 8, 14, 8)
        body = QLabel(text)
        body.setObjectName("chatText")
        body.setWordWrap(True)
        body.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        lay.addWidget(body)
        # A word-wrapped QLabel reports its narrowest wrap as its size hint; size it from the real width.
        body.ensurePolished()
        widest = max((body.fontMetrics().horizontalAdvance(line) for line in text.splitlines() or [""]), default=0)
        body.setMinimumWidth(min(widest + 4, self.MAX_W - 28))
        self.text = text


class AnswerCard(QFrame):
    view_graph = pyqtSignal(str)

    def __init__(self, theme: Theme, math: str, *, verified: bool, steps: int, calls: int, reused: bool,
                 final_id: str | None) -> None:
        super().__init__()
        self.setObjectName("answerCard")
        self.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Preferred)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(16, 12, 16, 14)
        lay.setSpacing(8)
        who = QLabel("Answer")
        who.setObjectName("secondary")
        lay.addWidget(who)
        body = QLabel(math)
        body.setObjectName("answerMath")
        body.setWordWrap(True)
        body.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        lay.addWidget(body)
        row = QHBoxLayout()
        row.setSpacing(8)
        head = "✓ Verified" if verified else "● Not fully verified"
        extra = " · from knowledge base" if reused else ""
        self.chip_text = f"{head} · {_plural(steps, 'step')} · {_plural(calls, 'model call')}{extra}"
        chip = QLabel(self.chip_text)
        chip.setStyleSheet(theme.chip_css("verified" if verified else "proposed"))
        row.addWidget(chip)
        row.addStretch(1)
        self.view_btn = QPushButton("View graph")
        self.view_btn.setObjectName("outline")
        self.view_btn.setIcon(icon("graph", theme.hex("text")))
        self.view_btn.setEnabled(final_id is not None)
        self.view_btn.clicked.connect(lambda: final_id and self.view_graph.emit(final_id))
        row.addWidget(self.view_btn)
        lay.addLayout(row)
        body.ensurePolished()
        chip.ensurePolished()
        widest = max(body.fontMetrics().horizontalAdvance(math),
                     chip.sizeHint().width() + 8 + self.view_btn.sizeHint().width())
        body.setMinimumWidth(min(widest + 4, COLUMN_MAX_W - 32))


class ThinkingGroup(QWidget):
    """Model thoughts for one task, collapsed behind "Thinking · N steps"."""

    def __init__(self, theme: Theme) -> None:
        super().__init__()
        self.lines: list[str] = []
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(2)
        self.toggle = QToolButton()
        self.toggle.setObjectName("link")
        self.toggle.setCheckable(True)
        self.toggle.toggled.connect(self._toggled)
        self.body = QLabel()
        self.body.setObjectName("secondary")
        self.body.setWordWrap(True)
        self.body.setContentsMargins(14, 0, 0, 0)
        self.body.hide()
        lay.addWidget(self.toggle, 0, Qt.AlignmentFlag.AlignLeft)
        lay.addWidget(self.body)
        self._refresh()

    def add(self, text: str) -> None:
        self.lines.append(text)
        self._refresh()

    def _refresh(self) -> None:
        n = len(self.lines)
        arrow = "▾" if self.toggle.isChecked() else "▸"
        self.toggle.setText(f"{arrow} Thinking · {_plural(n, 'step')}")
        self.body.setText("\n".join(f"{i + 1}. {t}" for i, t in enumerate(self.lines)))

    def _toggled(self, on: bool) -> None:
        self.body.setVisible(on)
        self._refresh()


class SystemLine(QLabel):
    def __init__(self, text: str) -> None:
        super().__init__(f"·  {text}")
        self.setObjectName("secondary")
        self.setWordWrap(True)


class Notice(QFrame):
    def __init__(self, theme: Theme, text: str) -> None:
        super().__init__()
        self.setObjectName("notice")
        lay = QHBoxLayout(self)
        lay.setContentsMargins(12, 8, 12, 8)
        lbl = QLabel(f"⚠ {text}")
        lbl.setWordWrap(True)
        lbl.setStyleSheet(f"color:{theme.hex('warning')};")
        lay.addWidget(lbl)


class ToolsPopover(QFrame):
    """List of tool calls in the current run, opened from the status strip (like "REMOTE · 8")."""

    MAX_ROWS = 10

    def __init__(self, theme: Theme, parent: QWidget) -> None:
        super().__init__(parent)
        self.theme = theme
        self.setObjectName("popover")
        self.setFixedWidth(360)
        add_soft_shadow(self)
        self.lay = QVBoxLayout(self)
        self.lay.setContentsMargins(14, 10, 14, 10)
        self.lay.setSpacing(2)
        self.head = CapsLabel("Tools · 0", theme)
        self.lay.addWidget(self.head)
        self.rows_box = QVBoxLayout()
        self.rows_box.setSpacing(0)
        self.lay.addLayout(self.rows_box)
        self.hide()

    def set_rows(self, rows: list[tuple[str, str]]) -> None:
        self.head.set_text(f"Tools · {len(rows)}")
        clear_layout(self.rows_box)
        shown = rows[-self.MAX_ROWS:]
        if len(rows) > len(shown):
            more = QLabel(f"{len(rows) - len(shown)} earlier calls")
            more.setObjectName("secondary")
            self.rows_box.addWidget(more)
        if not rows:
            empty = QLabel("No tool calls yet")
            empty.setObjectName("secondary")
            self.rows_box.addWidget(empty)
        for label, elapsed in shown:
            row = QWidget()
            hl = QHBoxLayout(row)
            hl.setContentsMargins(0, 3, 0, 3)
            hl.setSpacing(6)
            bolt = QLabel()
            bolt.setPixmap(icon("bolt", self.theme.hex("text")).pixmap(QSize(14, 14)))
            hl.addWidget(bolt)
            name = ElidedLabel(label)
            hl.addWidget(name, 1)
            t = QLabel(elapsed)
            t.setObjectName("secondary")
            hl.addWidget(t)
            self.rows_box.addWidget(row)
        self.adjustSize()


class Composer(QFrame):
    """Status strip on top, then the white input card: "+", tools | mic, send (terracotta)."""

    submitted = pyqtSignal(str)
    stop = pyqtSignal()
    strip_clicked = pyqtSignal()

    def __init__(self, theme: Theme) -> None:
        super().__init__()
        self.theme = theme
        self.setObjectName("composer")
        self.busy = False
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        self.status_strip = QToolButton()
        self.status_strip.setObjectName("statusStrip")
        self.status_strip.setIcon(icon("bolt", theme.hex("status_strip_text")))
        self.status_strip.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.status_strip.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.status_strip.setCursor(Qt.CursorShape.PointingHandCursor)
        self.status_strip.clicked.connect(self.strip_clicked.emit)
        self.status_strip.setText("No tool calls yet")
        lay.addWidget(self.status_strip)

        card = QFrame()
        card.setObjectName("composerCard")
        cl = QVBoxLayout(card)
        cl.setContentsMargins(16, 10, 10, 10)
        cl.setSpacing(6)
        self.input = QPlainTextEdit()
        self.input.setObjectName("composerInput")
        self.input.setPlaceholderText("Ask anything")
        self.input.setFixedHeight(48)
        self.input.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.input.installEventFilter(self)
        cl.addWidget(self.input)
        row = QHBoxLayout()
        row.setSpacing(4)
        ink = theme.hex("text")

        def icon_btn(name: str, tip: str) -> QToolButton:
            b = QToolButton()
            b.setObjectName("iconButton")
            b.setIcon(icon(name, ink))
            b.setIconSize(QSize(18, 18))
            b.setToolTip(tip)
            return b

        self.attach_btn = icon_btn("plus", "Attach a file")
        self.tools_btn = icon_btn("tools", "Tools")
        self.mic_btn = icon_btn("mic", "Voice input")
        self.send_btn = QPushButton()
        self.send_btn.setObjectName("send")
        self.send_btn.setFixedSize(36, 36)
        self.send_btn.setIconSize(QSize(18, 18))
        self.send_btn.clicked.connect(self._send_clicked)
        row.addWidget(self.attach_btn)
        row.addWidget(self.tools_btn)
        row.addStretch(1)
        row.addWidget(self.mic_btn)
        row.addSpacing(6)
        row.addWidget(self.send_btn)
        cl.addLayout(row)
        lay.addWidget(card)
        self.set_busy(False)

    def eventFilter(self, obj: QObject, event: QEvent) -> bool:  # noqa: N802
        if obj is self.input and event.type() == QEvent.Type.KeyPress:
            ke: QKeyEvent = event  # type: ignore[assignment]
            if ke.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter) and \
                    not ke.modifiers() & Qt.KeyboardModifier.ShiftModifier:
                self.submit()
                return True
        return False

    def submit(self) -> None:
        text = self.input.toPlainText().strip()
        if text:
            self.input.clear()
            self.submitted.emit(text)

    def _send_clicked(self) -> None:
        if self.busy and not self.input.toPlainText().strip():
            self.stop.emit()
        else:
            self.submit()

    def set_busy(self, busy: bool) -> None:
        self.busy = busy
        t = self.theme
        self.send_btn.setIcon(icon("stop" if busy else "arrow_up", t.hex("send_text")))
        self.send_btn.setToolTip("Stop (type to steer instead)" if busy else "Send")
        self.input.setPlaceholderText("Steer the run, or press Stop" if busy else "Ask anything")


class ChatPane(QWidget):
    submitted = pyqtSignal(str)
    stop = pyqtSignal()
    view_graph = pyqtSignal(str)

    def __init__(self, theme: Theme) -> None:
        super().__init__()
        self.theme = theme
        self.setObjectName("flat")
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 14, 0, 12)
        outer.setSpacing(8)

        self.title = ElidedLabel("New session")
        self.title.setObjectName("sessionTitle")
        self.title.setContentsMargins(SIDE_PAD, 0, SIDE_PAD, 0)
        outer.addWidget(self.title)

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        inner = QWidget()
        inner.setObjectName("flat")
        h = QHBoxLayout(inner)
        h.setContentsMargins(SIDE_PAD, 4, SIDE_PAD, 8)
        self.column = QWidget()
        self.column.setObjectName("flat")
        self.column.setMaximumWidth(COLUMN_MAX_W)
        self.messages = QVBoxLayout(self.column)
        self.messages.setContentsMargins(0, 0, 0, 0)
        self.messages.setSpacing(14)
        self.messages.addStretch(1)
        h.addStretch(0)
        h.addWidget(self.column, 1)
        h.addStretch(0)
        self.scroll.setWidget(inner)
        outer.addWidget(self.scroll, 1)
        self._stick = True
        bar = self.scroll.verticalScrollBar()
        bar.rangeChanged.connect(lambda _lo, hi: bar.setValue(hi) if self._stick else None)
        bar.valueChanged.connect(lambda v: setattr(self, "_stick", v >= bar.maximum() - 4))

        wrap = QHBoxLayout()
        wrap.setContentsMargins(SIDE_PAD - 8, 0, SIDE_PAD - 8, 0)
        self.composer = Composer(theme)
        self.composer.setMaximumWidth(COLUMN_MAX_W + 16)
        wrap.addWidget(self.composer, 1)
        outer.addLayout(wrap)
        self.input = self.composer.input
        self.composer.submitted.connect(self.submitted.emit)
        self.composer.stop.connect(self.stop.emit)
        self.composer.strip_clicked.connect(self.toggle_tools)

        self.popover = ToolsPopover(theme, self)

        self._thinking: ThinkingGroup | None = None
        self._plain: list[str] = []
        self._tools: list[tuple[str, str]] = []
        self._tool_keys: dict[str, int] = {}
        self._step = 0
        self._started: float | None = None
        self._finished: float | None = None
        self._running = False
        self._tick = QTimer(self)
        self._tick.setInterval(1000)
        self._tick.timeout.connect(self._update_strip)
        # last: the filter can fire as soon as it is installed, so everything it reads must exist
        QApplication.instance().installEventFilter(self)

    # composer / status ---------------------------------------------------------
    def _submit(self) -> None:  # kept for tests and callers that drive the input directly
        self.composer.submit()

    def set_busy(self, busy: bool) -> None:
        """Send <-> Stop. Any engine job counts (opening a session, a lock), not only a task run."""
        self.composer.set_busy(busy)

    def start_run(self) -> None:
        """A task started: reset the strip's tool list, step and clock."""
        self._started = time.monotonic()
        self._finished = None
        self._tools = []
        self._tool_keys = {}
        self._step = 0
        self._running = True
        self._tick.start()
        self._update_strip()

    def finish_run(self) -> None:
        self._finished = time.monotonic()
        self._running = False
        self._tick.stop()
        self._thinking = None
        self._update_strip()

    def set_step(self, step: int) -> None:
        self._step = step
        self._update_strip()

    def add_tool_call(self, label: str, elapsed: str, key: str | None = None) -> None:
        self._tools.append((label, elapsed))
        if key is not None:
            self._tool_keys[key] = len(self._tools) - 1
        self._tools_changed()

    def update_tool_call(self, key: str, elapsed: str) -> None:
        i = self._tool_keys.get(key)
        if i is not None and i < len(self._tools):
            self._tools[i] = (self._tools[i][0], elapsed)
            self._tools_changed()

    def tool_count(self) -> int:
        return len(self._tools)

    def _tools_changed(self) -> None:
        self._update_strip()
        if self.popover.isVisible():
            self._show_popover()

    def _update_strip(self) -> None:
        n = len(self._tools)
        if self._started is None:
            text = "No tool calls yet"
        else:
            end = self._finished if self._finished is not None else time.monotonic()
            el = fmt_elapsed(end - self._started)
            if self._running:
                # tool calls here are finished calls, so the strip says what is running: the controller step
                step = f" · step {self._step}" if self._step else ""
                text = f"Running{step} · {_plural(n, 'tool call')} · {el}"
            else:
                text = f"{_plural(n, 'tool call')} · {el} · done"
        self.composer.status_strip.setText(text)

    def toggle_tools(self) -> None:
        if self.popover.isVisible():
            self.popover.hide()
        else:
            self._show_popover()

    def _show_popover(self) -> None:
        self.popover.set_rows(self._tools)
        strip = self.composer.status_strip
        pos = strip.mapTo(self, strip.rect().topLeft())
        self.popover.adjustSize()
        x = min(pos.x() + 40, self.width() - self.popover.width() - 8)
        self.popover.move(max(8, x), pos.y() - self.popover.height() - 6)
        self.popover.show()
        self.popover.raise_()

    def eventFilter(self, obj: QObject, event: QEvent) -> bool:  # noqa: N802
        if self.popover.isVisible() and event.type() == QEvent.Type.MouseButtonPress:
            me: QMouseEvent = event  # type: ignore[assignment]
            gp = me.globalPosition().toPoint()
            inside = self.popover.rect().contains(self.popover.mapFromGlobal(gp))
            on_strip = self.composer.status_strip.rect().contains(self.composer.status_strip.mapFromGlobal(gp))
            if not inside and not on_strip:
                self.popover.hide()
        if self.popover.isVisible() and event.type() == QEvent.Type.KeyPress and \
                event.key() == Qt.Key.Key_Escape:  # type: ignore[attr-defined]
            self.popover.hide()
        return False

    # thread ------------------------------------------------------------------
    def set_title(self, text: str) -> None:
        self.title.set_full(text)

    def _append(self, widget: QWidget, align: Qt.AlignmentFlag = Qt.AlignmentFlag.AlignLeft) -> None:
        self._stick = True
        self.messages.insertWidget(self.messages.count() - 1, widget, 0, align)
        bar = self.scroll.verticalScrollBar()
        bar.setValue(bar.maximum())

    def add_user(self, text: str, label: str = "") -> None:
        self._append(Bubble(self.theme, text), Qt.AlignmentFlag.AlignRight)
        self._plain.append(f"You: {text}" if not label else f"You {label}: {text}")
        self._thinking = None

    def add_assistant(self, text: str, caption: str = "") -> None:
        self._append(InlineText(self.theme, text))
        self._plain.append(f"{PRODUCT_NAME}: {text}" + (f" ({caption})" if caption else ""))
        if caption:
            self.add_system(caption)
        self._thinking = None

    def add_table(self, columns: list[Column], rows: list[list[str]]) -> TableCard:
        card = TableCard(self.theme, columns, rows)
        self._append(card)
        self._plain.append(card.plain())
        return card

    def add_plot(self, spec: dict, title: str) -> None:
        self._append(PlotCard(self.theme, spec, title))
        self._plain.append(f"[plot] {title}")

    def add_answer(self, math: str, raw: str, *, verified: bool, steps: int, calls: int,
                   reused: bool = False, final_id: str | None = None) -> AnswerCard:
        card = AnswerCard(self.theme, math, verified=verified, steps=steps, calls=calls, reused=reused,
                          final_id=final_id)
        card.view_graph.connect(self.view_graph.emit)
        self._append(card)
        self._plain.append(f"{PRODUCT_NAME}: {raw} [{math}] ({card.chip_text})")
        self._thinking = None
        return card

    def add_thought(self, text: str) -> None:
        if self._thinking is None:
            self._thinking = ThinkingGroup(self.theme)
            self._append(self._thinking)
        self._thinking.add(text)
        self._plain.append(f"thinking: {text}")

    def add_notice(self, text: str) -> None:
        self._append(Notice(self.theme, text))
        self._plain.append(f"notice: {text}")

    def add_system(self, text: str) -> None:
        self._append(SystemLine(text))
        self._plain.append(text)

    add_meta = add_system

    def transcript_text(self) -> str:
        return "\n".join(self._plain)

    def clear(self) -> None:
        clear_layout(self.messages, keep=1)  # keep the trailing stretch
        self._plain.clear()
        self._thinking = None
        self._tools = []
        self._tool_keys = {}
        self._started = self._finished = None
        self._running = False
        self._tick.stop()
        self.popover.hide()
        self._update_strip()


__all__ = ["PRODUCT_NAME", "ChatPane", "Column", "fmt_elapsed"]
