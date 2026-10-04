"""Pinned note popover: a white rounded card with the numbered blue dot, a text field
and a black "Send" pill (as in the reference's figure view)."""
from __future__ import annotations

from PyQt6.QtCore import QEvent, QObject, QRectF, Qt, pyqtSignal
from PyQt6.QtGui import QKeyEvent, QPainter, QPaintEvent
from PyQt6.QtWidgets import QFrame, QHBoxLayout, QPlainTextEdit, QPushButton, QVBoxLayout, QWidget

PIN_D = 22  # marker diameter on the canvas and in the popover


def add_soft_shadow(w: QWidget) -> None:
    """Popover shadow (blur 18, y 4, 10% black). Fine on small popovers; never put this on a card."""
    from PyQt6.QtGui import QColor
    from PyQt6.QtWidgets import QGraphicsDropShadowEffect

    fx = QGraphicsDropShadowEffect(w)
    fx.setBlurRadius(18)
    fx.setOffset(0, 4)
    fx.setColor(QColor(0, 0, 0, 26))
    w.setGraphicsEffect(fx)


class PinDot(QWidget):
    def __init__(self, number: int, color, text_color, font) -> None:  # noqa: ANN001
        super().__init__()
        self.number, self.color, self.text_color, self.font_ = number, color, text_color, font
        self.setFixedSize(PIN_D, PIN_D)

    def set_number(self, n: int) -> None:
        self.number = n
        self.update()

    def paintEvent(self, event: QPaintEvent) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(self.color)
        p.drawEllipse(QRectF(0, 0, PIN_D, PIN_D))
        p.setPen(self.text_color)
        p.setFont(self.font_)
        p.drawText(QRectF(0, 0, PIN_D, PIN_D), Qt.AlignmentFlag.AlignCenter, str(self.number))
        p.end()


class PinEditor(QFrame):
    sent = pyqtSignal(str, str, int, float, float)  # node_id, text, number, rel_x, rel_y
    cancelled = pyqtSignal()
    closed = pyqtSignal()  # hidden for any reason (sent, cancelled, clicked away)

    W = 280

    def __init__(self, parent: QWidget, theme=None) -> None:  # noqa: ANN001
        super().__init__(parent)
        self.setObjectName("pinEditor")
        self.setFixedWidth(self.W)
        add_soft_shadow(self)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(12, 10, 10, 10)
        lay.setSpacing(6)
        row = QHBoxLayout()
        row.setSpacing(8)
        if theme is not None:
            self.dot = PinDot(1, theme.c("pin"), theme.c("accent_text"), theme.ui_font("size_small_px", bold=True))
        else:  # unthemed fallback (tests that build a bare editor)
            from PyQt6.QtGui import QColor, QFont
            self.dot = PinDot(1, QColor("#2C78D1"), QColor("#FFFFFF"), QFont())
        row.addWidget(self.dot, 0, Qt.AlignmentFlag.AlignTop)
        self.text = QPlainTextEdit()
        self.text.setObjectName("pinText")
        self.text.setPlaceholderText("Add a note for this step")
        self.text.setFixedHeight(48)
        self.text.installEventFilter(self)
        row.addWidget(self.text, 1)
        lay.addLayout(row)
        foot = QHBoxLayout()
        foot.addStretch(1)
        self.send_btn = QPushButton("Send")
        self.send_btn.setObjectName("darkPill")
        self.send_btn.clicked.connect(self._send)
        foot.addWidget(self.send_btn)
        lay.addLayout(foot)
        self._ctx: tuple[str, int, float, float] | None = None
        self.handle = ""
        self.hide()

    def open_for(self, node_id: str, handle: str, number: int, rx: float, ry: float) -> None:
        self._ctx = (node_id, number, rx, ry)
        self.handle = handle
        self.dot.set_number(number)
        self.text.clear()
        self.setToolTip(f"Note {number} on {handle}")
        self.adjustSize()
        self.show()
        self.raise_()
        self.text.setFocus()

    def eventFilter(self, obj: QObject, event: QEvent) -> bool:  # noqa: N802
        if obj is self.text and event.type() == QEvent.Type.KeyPress:
            ke: QKeyEvent = event  # type: ignore[assignment]
            if ke.key() == Qt.Key.Key_Escape:
                self._cancel()
                return True
            if ke.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter) and \
                    not ke.modifiers() & Qt.KeyboardModifier.ShiftModifier:
                self._send()
                return True
        return False

    def hideEvent(self, event) -> None:  # noqa: N802, ANN001
        super().hideEvent(event)
        self.closed.emit()

    def _send(self) -> None:
        text = self.text.toPlainText().strip()
        if not text or self._ctx is None:
            return
        node_id, number, rx, ry = self._ctx
        self.hide()
        self.sent.emit(node_id, text, number, rx, ry)

    def _cancel(self) -> None:
        self.hide()
        self.cancelled.emit()
