"""Pinned comment editor: a numbered note with a text box and a Send button."""
from __future__ import annotations

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QFrame, QHBoxLayout, QLabel, QPlainTextEdit, QPushButton, QVBoxLayout, QWidget


class PinEditor(QFrame):
    sent = pyqtSignal(str, str, int, float, float)  # node_id, text, number, rel_x, rel_y
    cancelled = pyqtSignal()

    def __init__(self, parent: QWidget) -> None:
        super().__init__(parent)
        self.setObjectName("pinEditor")
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setFixedWidth(260)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(8, 8, 8, 8)
        self.label = QLabel()
        self.text = QPlainTextEdit()
        self.text.setPlaceholderText("e.g. recheck this bound")
        self.text.setFixedHeight(64)
        row = QHBoxLayout()
        cancel = QPushButton("Cancel")
        send = QPushButton("Send")
        row.addStretch(1)
        row.addWidget(cancel)
        row.addWidget(send)
        lay.addWidget(self.label)
        lay.addWidget(self.text)
        lay.addLayout(row)
        cancel.clicked.connect(self._cancel)
        send.clicked.connect(self._send)
        self._ctx: tuple[str, int, float, float] | None = None
        self.hide()

    def open_for(self, node_id: str, handle: str, number: int, rx: float, ry: float) -> None:
        self._ctx = (node_id, number, rx, ry)
        self.label.setText(f"Note {number} on {handle}")
        self.text.clear()
        self.show()
        self.raise_()
        self.text.setFocus()

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
