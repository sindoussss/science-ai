from __future__ import annotations

import math

from PyQt6.QtCore import QLineF, QPointF, Qt
from PyQt6.QtGui import QBrush, QPainterPath, QPen, QPolygonF
from PyQt6.QtWidgets import QGraphicsPathItem

from sciai.graph.model import EdgeKind
from sciai.ui.theme.theme import Theme


class EdgeItem(QGraphicsPathItem):
    """Arrow from a node to what it depends on (src -> dst)."""

    def __init__(self, kind: EdgeKind, theme: Theme) -> None:
        super().__init__()
        self.kind = kind
        self.theme = theme
        color = {EdgeKind.CHECKS: "edge_check", EdgeKind.CONFLICTS_WITH: "edge_conflict"}.get(kind, "edge")
        style = {EdgeKind.CHECKS: Qt.PenStyle.DotLine, EdgeKind.CONFLICTS_WITH: Qt.PenStyle.DashLine}.get(
            kind, Qt.PenStyle.SolidLine)
        width = 2.0 if kind == EdgeKind.CONFLICTS_WITH else 1.2
        self.setPen(QPen(theme.c(color), width, style))
        self.setBrush(QBrush(theme.c(color)))
        self.setZValue(0)

    def set_ends(self, a: QPointF, b: QPointF) -> None:
        path = QPainterPath(a)
        dx = (b.x() - a.x()) * 0.5
        path.cubicTo(QPointF(a.x() + dx, a.y()), QPointF(b.x() - dx, b.y()), b)
        line = QLineF(path.pointAtPercent(0.97), b)
        ang = math.atan2(line.dy(), line.dx())
        size = 7
        p1 = b - QPointF(math.cos(ang - 0.4) * size, math.sin(ang - 0.4) * size)
        p2 = b - QPointF(math.cos(ang + 0.4) * size, math.sin(ang + 0.4) * size)
        path.addPolygon(QPolygonF([b, p1, p2, b]))
        self.setPath(path)
