"""Read-only code viewer: gray line-number gutter, Python highlighting, no wrap, thin h-scrollbar."""
from __future__ import annotations

import keyword
import re

from PyQt6.QtCore import QRect, QSize, Qt
from PyQt6.QtGui import QColor, QFont, QPainter, QPaintEvent, QResizeEvent, QSyntaxHighlighter, QTextCharFormat
from PyQt6.QtWidgets import QPlainTextEdit, QWidget

from sciai.ui.theme.theme import Theme

GUTTER_PAD = 12


class PythonHighlighter(QSyntaxHighlighter):
    def __init__(self, doc, theme: Theme) -> None:  # noqa: ANN001
        super().__init__(doc)

        def fmt(key: str, italic: bool = False) -> QTextCharFormat:
            f = QTextCharFormat()
            f.setForeground(QColor(theme.hex(key)))
            if italic:
                f.setFontItalic(True)
            return f

        kw = r"\b(" + "|".join(keyword.kwlist) + r")\b"
        self.rules = [
            (re.compile(r"\b[A-Za-z_]\w*(?=\()"), fmt("code_fn")),
            (re.compile(kw), fmt("code_kw")),
            (re.compile(r"\b\d+(\.\d+)?\b"), fmt("code_kw")),
            (re.compile(r"[rbfu]?'[^'\n]*'|[rbfu]?\"[^\"\n]*\""), fmt("code_str")),
            (re.compile(r"#.*$"), fmt("code_comment")),
        ]

    def highlightBlock(self, text: str) -> None:  # noqa: N802
        for rx, f in self.rules:  # later rules win (strings over keywords, comments over all)
            for m in rx.finditer(text):
                self.setFormat(m.start(), m.end() - m.start(), f)


class _Gutter(QWidget):
    def __init__(self, view: "CodeView") -> None:
        super().__init__(view)
        self.view = view

    def sizeHint(self) -> QSize:  # noqa: N802
        return QSize(self.view.gutter_width(), 0)

    def paintEvent(self, event: QPaintEvent) -> None:  # noqa: N802
        self.view.paint_gutter(event)


class CodeView(QPlainTextEdit):
    def __init__(self, theme: Theme) -> None:
        super().__init__()
        self.theme = theme
        self.setObjectName("code")
        self.setReadOnly(True)
        self.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.gutter = _Gutter(self)
        self.highlighter = PythonHighlighter(self.document(), theme)
        self.blockCountChanged.connect(lambda _n: self._update_margins())
        self.updateRequest.connect(self._scroll_gutter)
        self._update_margins()

    def gutter_width(self) -> int:
        digits = max(2, len(str(max(1, self.blockCount()))))
        return GUTTER_PAD + self.fontMetrics().horizontalAdvance("9") * digits + GUTTER_PAD

    def _update_margins(self) -> None:
        self.setViewportMargins(self.gutter_width(), 0, 0, 0)

    def _scroll_gutter(self, rect: QRect, dy: int) -> None:
        if dy:
            self.gutter.scroll(0, dy)
        else:
            self.gutter.update(0, rect.y(), self.gutter.width(), rect.height())

    def resizeEvent(self, event: QResizeEvent) -> None:  # noqa: N802
        super().resizeEvent(event)
        cr = self.contentsRect()
        self.gutter.setGeometry(QRect(cr.left(), cr.top(), self.gutter_width(), cr.height()))

    def paint_gutter(self, event: QPaintEvent) -> None:
        p = QPainter(self.gutter)
        p.fillRect(event.rect(), self.theme.c("code_gutter"))
        f = QFont(self.font())
        p.setFont(f)
        p.setPen(self.theme.c("code_line_no"))
        block = self.firstVisibleBlock()
        top = round(self.blockBoundingGeometry(block).translated(self.contentOffset()).top())
        h = round(self.blockBoundingRect(block).height())
        w = self.gutter.width() - GUTTER_PAD
        while block.isValid() and top <= event.rect().bottom():
            if block.isVisible() and top + h >= event.rect().top():
                p.drawText(0, top, w, h, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                           str(block.blockNumber() + 1))
            block = block.next()
            top += h
            h = round(self.blockBoundingRect(block).height())
        p.end()
