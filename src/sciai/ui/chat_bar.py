"""Chat under the canvas: transcript strip plus "Ask or steer the controller"."""
from __future__ import annotations

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QHBoxLayout, QLineEdit, QPushButton, QTextBrowser, QVBoxLayout, QWidget

from sciai.ui.theme.theme import Theme


class ChatPane(QWidget):
    submitted = pyqtSignal(str)
    stop = pyqtSignal()

    def __init__(self, theme: Theme) -> None:
        super().__init__()
        self.theme = theme
        self.setObjectName("chatPane")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(12, 8, 12, 10)
        lay.setSpacing(6)
        self.transcript = QTextBrowser()
        self.transcript.setOpenLinks(False)
        self.transcript.setFixedHeight(130)
        lay.addWidget(self.transcript)
        row = QHBoxLayout()
        self.input = QLineEdit()
        self.input.setPlaceholderText("Ask or steer the controller")
        self.input.returnPressed.connect(self._submit)
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self.stop.emit)
        self.send = QPushButton("Send")
        self.send.setObjectName("primary")
        self.send.clicked.connect(self._submit)
        row.addWidget(self.input, 1)
        row.addWidget(self.stop_btn)
        row.addWidget(self.send)
        lay.addLayout(row)

    def _submit(self) -> None:
        text = self.input.text().strip()
        if text:
            self.input.clear()
            self.submitted.emit(text)

    def set_busy(self, busy: bool) -> None:
        self.stop_btn.setEnabled(busy)
        self.input.setPlaceholderText("Steer the controller (it is working)" if busy else "Ask or steer the controller")

    def add(self, who: str, text: str, muted: bool = False) -> None:
        from html import escape

        color = self.theme.hex("text_muted" if muted else "text")
        self.transcript.append(f'<p style="margin:2px 0;color:{color}"><b>{escape(who)}</b> {escape(text)}</p>')
        sb = self.transcript.verticalScrollBar()
        sb.setValue(sb.maximum())

    def clear(self) -> None:
        self.transcript.clear()
