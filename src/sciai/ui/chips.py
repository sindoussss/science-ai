"""One painter for status chips, shared by the canvas, the legend and table cards, so a painted
chip looks exactly like a QSS one (Theme.chip_css): panel fill, gray text, 1px gray outline
(dashed for "invalidated"), 6px radius, 12px medium text, 2px/8px padding. No symbols, no color."""
from __future__ import annotations

from PyQt6.QtCore import QRectF, Qt
from PyQt6.QtGui import QBrush, QFontMetricsF, QPainter, QPen

from sciai.ui.theme.theme import Theme

PAD_X = 8
PAD_Y = 2


def chip_label(status: str, text: str | None = None) -> str:
    return text if text is not None else status


def chip_size(theme: Theme, label: str) -> tuple[float, float]:
    fm = QFontMetricsF(theme.chip_font())
    return fm.horizontalAdvance(label) + 2 * PAD_X, fm.height() + 2 * PAD_Y


def paint_chip(p: QPainter, theme: Theme, x: float, cy: float, status: str, label: str,
               right_aligned: bool = False) -> QRectF:
    """Draw a chip vertically centered on ``cy``; ``x`` is its left edge (or right edge)."""
    fg = theme.c(f"status_{status}")
    w, h = chip_size(theme, label)
    rect = QRectF(x - w if right_aligned else x, cy - h / 2, w, h)
    p.save()
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    p.setBrush(QBrush(theme.c("panel")))
    if status == "invalidated":
        p.setPen(QPen(theme.c("border_strong"), 1, Qt.PenStyle.DashLine))
    else:
        p.setPen(QPen(theme.c("border"), 1))
    r = rect.adjusted(0.5, 0.5, -0.5, -0.5)
    radius = theme.radius("chip")
    p.drawRoundedRect(r, radius, radius)
    p.setFont(theme.chip_font())
    p.setPen(fg)
    p.drawText(rect, Qt.AlignmentFlag.AlignCenter, label)
    p.restore()
    return rect
