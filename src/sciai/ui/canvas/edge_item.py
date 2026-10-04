"""Edges: thin bezier stroke plus a small filled arrowhead (stroke is never filled)."""
from __future__ import annotations

import math

from PyQt6.QtCore import QPointF, QRectF, Qt
from PyQt6.QtGui import QBrush, QPainter, QPainterPath, QPen, QPolygonF
from PyQt6.QtWidgets import QGraphicsItem, QStyleOptionGraphicsItem, QWidget

from sciai.graph.model import EdgeKind
from sciai.ui.theme.theme import Theme

ARROW = 6.0


class EdgeItem(QGraphicsItem):
    """Arrow between two nodes; on the top-to-bottom canvas it runs down from a dependency into its dependent."""

    def __init__(self, kind: EdgeKind, theme: Theme) -> None:
        super().__init__()
        self.kind = kind
        color = {EdgeKind.CHECKS: "edge_check", EdgeKind.CONFLICTS_WITH: "edge_conflict"}.get(kind, "edge")
        style = {EdgeKind.CHECKS: Qt.PenStyle.DotLine, EdgeKind.CONFLICTS_WITH: Qt.PenStyle.DashLine}.get(
            kind, Qt.PenStyle.SolidLine)
        self.color = theme.c(color)
        self.pen = QPen(self.color, 1.5 if kind == EdgeKind.CONFLICTS_WITH else 1.25, style)
        self.pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        self.path = QPainterPath()
        self.head = QPolygonF()
        self.setZValue(0)

    def set_ends(self, a: QPointF, b: QPointF, vertical: bool = False) -> None:
        """Vertical: down from a to b. Otherwise sideways, bulging out of a and into b in the
        direction of travel (an assumption's right edge into the problem's left edge)."""
        self.prepareGeometryChange()
        path = QPainterPath(a)
        if vertical:
            dy = max(12.0, abs(b.y() - a.y()) * 0.5) * (1 if b.y() >= a.y() else -1)
            c1, c2 = QPointF(a.x(), a.y() + dy), QPointF(b.x(), b.y() - dy)
        else:
            dx = max(24.0, abs(b.x() - a.x()) * 0.45) * (1 if b.x() >= a.x() else -1)
            c1, c2 = QPointF(a.x() + dx, a.y()), QPointF(b.x() - dx, b.y())
        # stop the stroke at the arrow's base so the tip stays crisp
        ang = math.atan2(b.y() - c2.y(), b.x() - c2.x())
        base = QPointF(b.x() - math.cos(ang) * ARROW, b.y() - math.sin(ang) * ARROW)
        path.cubicTo(c1, c2, base)
        self.path = path
        left = QPointF(b.x() - math.cos(ang - 0.45) * ARROW, b.y() - math.sin(ang - 0.45) * ARROW)
        right = QPointF(b.x() - math.cos(ang + 0.45) * ARROW, b.y() - math.sin(ang + 0.45) * ARROW)
        self.head = QPolygonF([b, left, right])

    def boundingRect(self) -> QRectF:  # noqa: N802
        return self.path.boundingRect().united(self.head.boundingRect()).adjusted(-4, -4, 4, 4)

    def paint(self, p: QPainter, option: QStyleOptionGraphicsItem, widget: QWidget | None = None) -> None:
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setPen(self.pen)
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawPath(self.path)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(self.color))
        p.drawPolygon(self.head)
