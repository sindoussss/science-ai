"""Small line icons drawn with QPainter, so no icon font or image files are needed."""
from __future__ import annotations

from functools import lru_cache

from PyQt6.QtCore import QPointF, QRectF, Qt
from PyQt6.QtGui import QColor, QIcon, QPainter, QPainterPath, QPen, QPixmap

SIZE = 16


def _pen(color: QColor, w: float = 1.4) -> QPen:
    pen = QPen(color, w)
    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
    return pen


def _draw(name: str, p: QPainter, c: QColor) -> None:
    p.setPen(_pen(c))
    p.setBrush(Qt.BrushStyle.NoBrush)
    if name == "plus":
        p.drawLine(QPointF(8, 3), QPointF(8, 13))
        p.drawLine(QPointF(3, 8), QPointF(13, 8))
    elif name == "chat":
        path = QPainterPath()
        path.addRoundedRect(QRectF(2.5, 3, 11, 8), 2.5, 2.5)
        p.drawPath(path)
        p.drawLine(QPointF(5.5, 11), QPointF(4.5, 13.5))
        p.drawLine(QPointF(4.5, 13.5), QPointF(8, 11))
    elif name == "file":
        path = QPainterPath(QPointF(4, 2.5))
        path.lineTo(9.5, 2.5)
        path.lineTo(12.5, 5.5)
        path.lineTo(12.5, 13.5)
        path.lineTo(4, 13.5)
        path.closeSubpath()
        p.drawPath(path)
        p.drawLine(QPointF(9.5, 2.5), QPointF(9.5, 5.5))
        p.drawLine(QPointF(9.5, 5.5), QPointF(12.5, 5.5))
    elif name == "check":
        p.setPen(_pen(c, 1.8))
        p.drawLine(QPointF(3.5, 8.5), QPointF(6.5, 11.5))
        p.drawLine(QPointF(6.5, 11.5), QPointF(12.5, 4.5))
    elif name == "chevron_down":
        p.drawLine(QPointF(4.5, 6.5), QPointF(8, 10))
        p.drawLine(QPointF(8, 10), QPointF(11.5, 6.5))
    elif name == "chevron_right":
        p.drawLine(QPointF(6.5, 4.5), QPointF(10, 8))
        p.drawLine(QPointF(10, 8), QPointF(6.5, 11.5))
    elif name == "empty":
        p.drawEllipse(QRectF(6.5, 6.5, 3, 3))


@lru_cache(maxsize=64)
def icon(name: str, color: str) -> QIcon:
    out = QIcon()
    for scale in (1, 2):
        pm = QPixmap(SIZE * scale, SIZE * scale)
        pm.setDevicePixelRatio(scale)
        pm.fill(Qt.GlobalColor.transparent)
        p = QPainter(pm)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        _draw(name, p, QColor(color))
        p.end()
        out.addPixmap(pm)
    return out


def paint_icon(p: QPainter, name: str, color: QColor, x: float, y: float) -> None:
    """Draw an icon directly into another widget's painter at (x, y)."""
    p.save()
    p.translate(x, y)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    _draw(name, p, color)
    p.restore()
