"""Chat under the canvas: message bubbles, collapsible thinking, and the input bar."""
from __future__ import annotations

from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from sciai.ui.layout_util import clear_layout
from sciai.ui.theme.theme import Theme

PRODUCT_NAME = "Science AI"


class Bubble(QFrame):
    MAX_W = 640

    def __init__(self, who: str, text: str, mine: bool, caption: str = "") -> None:
        super().__init__()
        self.setObjectName("bubbleUser" if mine else "bubbleAssistant")
        self.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Preferred)
        self.setMaximumWidth(self.MAX_W)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(12, 8, 12, 8)
        lay.setSpacing(2)
        if not mine:
            name = QLabel(who)
            name.setStyleSheet("font-weight:600;")
            lay.addWidget(name)
        body = QLabel(text)
        body.setWordWrap(True)
        body.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        lay.addWidget(body)
        cap = None
        if caption:
            cap = QLabel(caption)
            cap.setObjectName("muted")
            lay.addWidget(cap)
        # A word-wrapped QLabel reports its narrowest wrap as the size hint, so a
        # Maximum-policy bubble would shrink to one word. Size it from the real
        # single-line width instead, capped so long text still wraps.
        inner_max = self.MAX_W - 24
        body.ensurePolished()
        widest = max((body.fontMetrics().horizontalAdvance(line) for line in text.splitlines() or [""]), default=0)
        if cap is not None:
            cap.ensurePolished()
            widest = max(widest, cap.fontMetrics().horizontalAdvance(caption))
        body.setMinimumWidth(min(widest + 4, inner_max))
        self.text = text


class ThinkingGroup(QWidget):
    """Model thoughts for one task, collapsed behind a toggle."""

    def __init__(self, theme: Theme) -> None:
        super().__init__()
        self.lines: list[str] = []
        lay = QVBoxLayout(self)
        lay.setContentsMargins(4, 0, 0, 0)
        lay.setSpacing(2)
        self.toggle = QToolButton()
        self.toggle.setObjectName("link")
        self.toggle.setCheckable(True)
        self.toggle.toggled.connect(self._toggled)
        self.body = QLabel()
        self.body.setObjectName("muted")
        self.body.setWordWrap(True)
        self.body.setStyleSheet(f"font-size:{theme.px('size_small_px')}px; padding-left:12px;")
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
        self.toggle.setText(f"{arrow} Thinking · {n} step{'s' if n != 1 else ''}")
        self.body.setText("\n".join(f"{i}. {t}" for i, t in enumerate(self.lines, 1)))

    def _toggled(self, on: bool) -> None:
        self.body.setVisible(on)
        self._refresh()


class ChatPane(QWidget):
    submitted = pyqtSignal(str)
    stop = pyqtSignal()

    def __init__(self, theme: Theme) -> None:
        super().__init__()
        self.theme = theme
        self.setObjectName("cardBody")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(14, 10, 14, 12)
        lay.setSpacing(8)

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.scroll.setFixedHeight(190)
        inner = QWidget()
        self.messages = QVBoxLayout(inner)
        self.messages.setContentsMargins(0, 0, 4, 0)
        self.messages.setSpacing(6)
        self.messages.addStretch(1)
        self.scroll.setWidget(inner)
        lay.addWidget(self.scroll)

        row = QHBoxLayout()
        row.setSpacing(8)
        self.input = QLineEdit()
        self.input.setPlaceholderText("Ask or steer the controller")
        self.input.returnPressed.connect(self._submit)
        self.input.setMinimumHeight(34)
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self.stop.emit)
        self.send = QPushButton("Send")
        self.send.clicked.connect(self._submit)
        row.addWidget(self.input, 1)
        row.addWidget(self.stop_btn)
        row.addWidget(self.send)
        lay.addLayout(row)
        self._thinking: ThinkingGroup | None = None
        self._plain: list[str] = []

    # input -----------------------------------------------------------------
    def _submit(self) -> None:
        text = self.input.text().strip()
        if text:
            self.input.clear()
            self.submitted.emit(text)

    def set_busy(self, busy: bool) -> None:
        self.stop_btn.setEnabled(busy)
        self.input.setPlaceholderText("Steer the controller while it works" if busy else "Ask or steer the controller")
        if not busy:
            self._thinking = None

    # transcript ------------------------------------------------------------
    def _append(self, widget: QWidget, align: Qt.AlignmentFlag) -> None:
        self.messages.insertWidget(self.messages.count() - 1, widget, 0, align)
        QTimer.singleShot(0, lambda: self.scroll.verticalScrollBar().setValue(
            self.scroll.verticalScrollBar().maximum()))

    def add_user(self, text: str, label: str = "") -> None:
        shown = f"{label} {text}".strip() if label else text
        self._append(Bubble("You", shown, mine=True), Qt.AlignmentFlag.AlignRight)
        self._plain.append(f"You: {shown}")
        self._thinking = None

    def add_assistant(self, text: str, caption: str = "") -> None:
        self._append(Bubble(PRODUCT_NAME, text, mine=False, caption=caption), Qt.AlignmentFlag.AlignLeft)
        self._plain.append(f"{PRODUCT_NAME}: {text}" + (f" ({caption})" if caption else ""))
        self._thinking = None

    def add_thought(self, text: str) -> None:
        if self._thinking is None:
            self._thinking = ThinkingGroup(self.theme)
            self._append(self._thinking, Qt.AlignmentFlag.AlignLeft)
        self._thinking.add(text)
        self._plain.append(f"thinking: {text}")

    def add_notice(self, text: str) -> None:
        box = QFrame()
        box.setObjectName("notice")
        lay = QHBoxLayout(box)
        lay.setContentsMargins(10, 6, 10, 6)
        lbl = QLabel(f"⚠ {text}")
        lbl.setWordWrap(True)
        lbl.setStyleSheet(f"color:{self.theme.hex('warning')};")
        lay.addWidget(lbl)
        self._append(box, Qt.AlignmentFlag.AlignLeft)
        self._plain.append(f"notice: {text}")

    def add_meta(self, text: str) -> None:
        lbl = QLabel(text)
        lbl.setObjectName("muted")
        lbl.setStyleSheet(f"font-size:{self.theme.px('size_small_px')}px; padding-left:4px;")
        self._append(lbl, Qt.AlignmentFlag.AlignLeft)
        self._plain.append(text)

    def transcript_text(self) -> str:
        return "\n".join(self._plain)

    def clear(self) -> None:
        clear_layout(self.messages, keep=1)  # keep the trailing stretch
        self._plain.clear()
        self._thinking = None
