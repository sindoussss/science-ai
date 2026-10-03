"""A graph node on the canvas. Status uses color AND an icon + text label."""
from __future__ import annotations

from typing import Callable

from PyQt6.QtCore import QPointF, QRectF, Qt
from PyQt6.QtGui import QBrush, QFontMetrics, QPainter, QPainterPath, QPen
from PyQt6.QtWidgets import QGraphicsItem, QGraphicsSceneMouseEvent, QStyleOptionGraphicsItem, QWidget

from sciai.graph.model import Node, NodeType, Status
from sciai.ui.theme.theme import STATUS_ICON, Theme

W, H = 210, 78
PLOT_H = 120
CHECK_W, CHECK_H = 196, 30


class NodeItem(QGraphicsItem):
    def __init__(self, node: Node, handle: str, theme: Theme,
                 on_click: Callable[["NodeItem", QPointF], None]) -> None:
        super().__init__()
        self.node, self.handle, self.theme = node, handle, theme
        self.on_click = on_click
        self.selected = False
        self.highlight = False
        self.pins: list[tuple[int, float, float]] = []  # (number, rel_x, rel_y)
        self.setAcceptHoverEvents(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setZValue(1 if node.type == NodeType.CHECK else 2)

    # geometry ------------------------------------------------------------
    def size(self) -> tuple[float, float]:
        if self.node.type == NodeType.CHECK:
            return CHECK_W, CHECK_H
        if self.node.result and self.node.result.get("kind") == "plotspec":
            return W, PLOT_H
        return W, H

    def boundingRect(self) -> QRectF:  # noqa: N802 - Qt API
        w, h = self.size()
        return QRectF(-2, -2, w + 4, h + 4)

    def update_node(self, node: Node, handle: str) -> None:
        self.prepareGeometryChange()
        self.node, self.handle = node, handle
        self.update()

    # painting ------------------------------------------------------------
    def paint(self, p: QPainter, option: QStyleOptionGraphicsItem, widget: QWidget | None = None) -> None:
        t = self.theme
        n = self.node
        w, h = self.size()
        status = n.status.value
        fg, bg = t.status_colors(status)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = QRectF(0, 0, w, h)

        pen = QPen(t.c("border_strong"), 1)
        if n.status == Status.INVALIDATED:
            pen = QPen(t.c("status_invalidated"), 1.2, Qt.PenStyle.DashLine)
        elif n.status in (Status.VERIFIED, Status.FAILED, Status.HYPOTHESIS, Status.PROPOSED):
            pen = QPen(fg, 1.2)
        if self.highlight:
            pen = QPen(t.c("edge_conflict"), 2)
        if self.selected:
            pen = QPen(t.c("selection"), 2)
        p.setPen(pen)
        p.setBrush(QBrush(t.c("panel") if n.status != Status.INVALIDATED else t.c("status_invalidated_bg")))
        p.drawRoundedRect(rect, t.radius("node"), t.radius("node"))

        # status pill: icon + label
        label = f"{STATUS_ICON.get(status, '')} {status}"
        if n.type == NodeType.CHECK and n.result:
            label = f"{STATUS_ICON.get(status, '')} {n.result.get('outcome', '')}"
        p.setFont(t.ui_font("size_small", bold=True))
        fm = QFontMetrics(p.font())
        pill_w = fm.horizontalAdvance(label) + 12
        pill = QRectF(w - pill_w - 6, 6 if n.type != NodeType.CHECK else (h - 17) / 2, pill_w, 17)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(bg))
        p.drawRoundedRect(pill, 8, 8)
        p.setPen(fg)
        p.drawText(pill, Qt.AlignmentFlag.AlignCenter, label)

        # title
        p.setPen(t.c("text"))
        p.setFont(t.ui_font(bold=True))
        title_w = w - pill_w - 18
        label_text = f"{self.handle}  {n.title}"
        if n.type == NodeType.CHECK:
            p.setFont(t.ui_font("size_small"))
            label_text = f"{self.handle}  {n.title.split(': ', 1)[-1].replace('_', ' ')}"
        title = QFontMetrics(p.font()).elidedText(label_text, Qt.TextElideMode.ElideRight, int(title_w))
        top = 6 if n.type != NodeType.CHECK else (h - 18) / 2
        p.drawText(QRectF(8, top, title_w, 18), Qt.AlignmentFlag.AlignVCenter, title)

        if n.type == NodeType.CHECK:
            return
        # badges: lock / warning
        badges = ("🔒 " if n.locked else "") + ("⚠" if n.warning or n.flags else "")
        sub = n.tool_name or n.type.value
        p.setFont(t.ui_font("size_small"))
        p.setPen(t.c("text_muted"))
        p.drawText(QRectF(8, 26, w - 16, 16), Qt.AlignmentFlag.AlignVCenter,
                   QFontMetrics(p.font()).elidedText(f"{badges} {sub}".strip(), Qt.TextElideMode.ElideRight, w - 16))

        if n.result and n.result.get("kind") == "plotspec":
            self._paint_plot(p, QRectF(8, 46, w - 16, h - 54))
        else:
            text = n.content if n.type in (NodeType.PROBLEM, NodeType.FINAL, NodeType.ERROR) else n.display_result()
            p.setFont(t.mono_font())
            p.setPen(t.c("text") if n.status != Status.INVALIDATED else t.c("text_faint"))
            p.drawText(QRectF(8, 46, w - 16, h - 50), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop,
                       QFontMetrics(p.font()).elidedText(text, Qt.TextElideMode.ElideRight, int(w - 16)))

        for num, rx, ry in self.pins:
            c = QPointF(rx * w, ry * h)
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QBrush(t.c("pin")))
            p.drawEllipse(c, 8, 8)
            p.setPen(t.c("accent_text"))
            p.setFont(t.ui_font("size_small", bold=True))
            p.drawText(QRectF(c.x() - 8, c.y() - 8, 16, 16), Qt.AlignmentFlag.AlignCenter, str(num))

    def _paint_plot(self, p: QPainter, area: QRectF) -> None:
        spec = self.node.result["value"]
        pts = [(x, y) for x, y in zip(spec.get("x", []), spec.get("y", [])) if y is not None]
        if len(pts) < 2:
            return
        xs, ys = [a for a, _ in pts], [b for _, b in pts]
        x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
        if x1 == x0:
            return
        if y1 == y0:
            y0, y1 = y0 - 1, y1 + 1
        path = QPainterPath()
        for i, (x, y) in enumerate(pts):
            px = area.left() + (x - x0) / (x1 - x0) * area.width()
            py = area.bottom() - (y - y0) / (y1 - y0) * area.height()
            path.moveTo(px, py) if i == 0 else path.lineTo(px, py)
        p.setPen(QPen(self.theme.c("border"), 1))
        p.drawLine(area.bottomLeft(), area.bottomRight())
        p.setPen(QPen(self.theme.c("accent"), 1.5))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawPath(path)

    # interaction ---------------------------------------------------------
    def mousePressEvent(self, event: QGraphicsSceneMouseEvent) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton:
            self.on_click(self, event.pos())
            event.accept()
        else:
            super().mousePressEvent(event)
