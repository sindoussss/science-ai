"""Small line icons drawn with QPainter on a 16x16 grid, so no icon font or image files are needed."""
from __future__ import annotations

from functools import lru_cache

from PyQt6.QtCore import QPointF, QRectF, Qt
from PyQt6.QtGui import QBrush, QColor, QIcon, QPainter, QPainterPath, QPen, QPixmap, QPolygonF

SIZE = 16


def _pen(color: QColor, w: float = 1.4) -> QPen:
    pen = QPen(color, w)
    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
    return pen


def _line(p: QPainter, *pts: tuple[float, float]) -> None:
    for a, b in zip(pts, pts[1:]):
        p.drawLine(QPointF(*a), QPointF(*b))


def _draw(name: str, p: QPainter, c: QColor) -> None:  # noqa: C901 - one branch per icon
    p.setPen(_pen(c))
    p.setBrush(Qt.BrushStyle.NoBrush)
    if name == "plus":
        _line(p, (8, 3), (8, 13))
        _line(p, (3, 8), (13, 8))
    elif name == "chat":
        path = QPainterPath()
        path.addRoundedRect(QRectF(2.5, 3, 11, 8), 2.5, 2.5)
        p.drawPath(path)
        _line(p, (5.5, 11), (4.5, 13.5), (8, 11))
    elif name == "file":
        path = QPainterPath(QPointF(4, 2.5))
        for x, y in ((9.5, 2.5), (12.5, 5.5), (12.5, 13.5), (4, 13.5)):
            path.lineTo(x, y)
        path.closeSubpath()
        p.drawPath(path)
        _line(p, (9.5, 2.5), (9.5, 5.5), (12.5, 5.5))
    elif name == "files":
        p.drawRoundedRect(QRectF(5, 2.5, 8, 9.5), 1.5, 1.5)
        _line(p, (3, 5), (3, 13.5), (10, 13.5))
    elif name == "briefcase":
        p.drawRoundedRect(QRectF(2.5, 5, 11, 8), 1.5, 1.5)
        _line(p, (6, 5), (6, 3.5), (10, 3.5), (10, 5))
        _line(p, (2.5, 8.5), (13.5, 8.5))
    elif name == "check":
        p.setPen(_pen(c, 1.8))
        _line(p, (3.5, 8.5), (6.5, 11.5), (12.5, 4.5))
    elif name == "check_circle":
        p.drawEllipse(QRectF(2, 2, 12, 12))
        _line(p, (5.3, 8.2), (7.2, 10), (10.7, 6.2))
    elif name == "x_circle":
        p.drawEllipse(QRectF(2, 2, 12, 12))
        _line(p, (6, 6), (10, 10))
        _line(p, (10, 6), (6, 10))
    elif name == "dash_circle":
        p.drawEllipse(QRectF(2, 2, 12, 12))
        _line(p, (5.5, 8), (10.5, 8))
    elif name == "chevron_down":
        _line(p, (4.5, 6.5), (8, 10), (11.5, 6.5))
    elif name == "chevron_right":
        _line(p, (6.5, 4.5), (10, 8), (6.5, 11.5))
    elif name == "chevron_left":
        _line(p, (9.5, 4.5), (6, 8), (9.5, 11.5))
    elif name == "arrow_left":
        _line(p, (13, 8), (3, 8))
        _line(p, (7, 4), (3, 8), (7, 12))
    elif name == "arrow_up":
        p.setPen(_pen(c, 1.8))
        _line(p, (8, 13), (8, 3.5))
        _line(p, (4, 7.5), (8, 3.5), (12, 7.5))
    elif name == "stop":
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(c))
        p.drawRoundedRect(QRectF(4.5, 4.5, 7, 7), 1.5, 1.5)
    elif name == "bolt":
        poly = QPolygonF([QPointF(9, 1.8), QPointF(3.5, 9), QPointF(7.5, 9), QPointF(6.5, 14.2),
                          QPointF(12.5, 6.8), QPointF(8.5, 6.8)])
        p.drawPolygon(poly)
    elif name == "mic":
        p.drawRoundedRect(QRectF(6, 2, 4, 7.5), 2, 2)
        path = QPainterPath(QPointF(3.8, 7.5))
        path.cubicTo(QPointF(3.8, 13), QPointF(12.2, 13), QPointF(12.2, 7.5))
        p.drawPath(path)
        _line(p, (8, 11.8), (8, 14))
    elif name == "tools":
        p.drawRoundedRect(QRectF(2.5, 2.5, 5, 5), 1.5, 1.5)
        p.drawRoundedRect(QRectF(8.5, 2.5, 5, 5), 1.5, 1.5)
        p.drawRoundedRect(QRectF(2.5, 8.5, 5, 5), 1.5, 1.5)
        p.drawRoundedRect(QRectF(8.5, 8.5, 5, 5), 2.5, 2.5)
    elif name == "gear":
        p.drawEllipse(QRectF(5.6, 5.6, 4.8, 4.8))
        for i in range(8):
            p.save()
            p.translate(8, 8)
            p.rotate(i * 45)
            p.drawLine(QPointF(0, -4.6), QPointF(0, -6.4))
            p.restore()
        p.drawEllipse(QRectF(3.4, 3.4, 9.2, 9.2))
    elif name == "notebook":
        p.drawRoundedRect(QRectF(2.5, 3, 11, 10), 1.5, 1.5)
        _line(p, (5.5, 3), (5.5, 13))
        _line(p, (8, 6.5), (11, 6.5))
    elif name == "download":
        _line(p, (8, 2.5), (8, 10))
        _line(p, (5, 7), (8, 10), (11, 7))
        _line(p, (3, 11), (3, 13.5), (13, 13.5), (13, 11))
    elif name == "close":
        _line(p, (4, 4), (12, 12))
        _line(p, (12, 4), (4, 12))
    elif name == "more":
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(c))
        for x in (3.5, 8, 12.5):
            p.drawEllipse(QPointF(x, 8), 1.2, 1.2)
    elif name == "graph":
        p.drawRoundedRect(QRectF(5.5, 2, 5, 3.5), 1, 1)
        p.drawRoundedRect(QRectF(2, 10.5, 5, 3.5), 1, 1)
        p.drawRoundedRect(QRectF(9, 10.5, 5, 3.5), 1, 1)
        _line(p, (8, 5.5), (8, 8), (4.5, 8), (4.5, 10.5))
        _line(p, (8, 8), (11.5, 8), (11.5, 10.5))
    elif name == "dot":
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(c))
        p.drawEllipse(QPointF(8, 8), 3.2, 3.2)
    elif name == "hollow":
        p.setPen(_pen(c, 1.1))
        p.drawEllipse(QPointF(8, 8), 2.6, 2.6)
    elif name == "empty":
        p.drawEllipse(QRectF(6.5, 6.5, 3, 3))


@lru_cache(maxsize=256)
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


def paint_icon(p: QPainter, name: str, color: QColor, x: float, y: float, size: float = SIZE) -> None:
    """Draw an icon directly into another widget's painter at (x, y)."""
    p.save()
    p.translate(x, y)
    if size != SIZE:
        p.scale(size / SIZE, size / SIZE)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    _draw(name, p, color)
    p.restore()
