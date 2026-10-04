"""A graph node on the canvas. Status uses a soft chip with an icon AND a text label."""
from __future__ import annotations

from typing import Callable

from PyQt6.QtCore import QPointF, QRectF, Qt
from PyQt6.QtGui import QBrush, QFontMetrics, QPainter, QPainterPath, QPen
from PyQt6.QtWidgets import QGraphicsItem, QGraphicsSceneMouseEvent, QStyleOptionGraphicsItem, QWidget

from sciai.graph.model import Node, NodeType, Status
from sciai.ui.canvas.layout import NODE_H, NODE_W, PILL_H, PILL_INSET, PLOT_H
from sciai.ui.theme.theme import STATUS_ICON, Theme

PAD = 12
CHIP_H = 20


def node_tooltip(node: Node, handle: str) -> str:
    parts = [f"{handle} · {node.title}", node.tool_name or node.type.value, f"status: {node.status.value}"]
    body = node.content if node.type in (NodeType.PROBLEM, NodeType.FINAL, NodeType.ERROR) else node.display_result()
    if body:
        parts.append(body[:400])
    return "\n".join(parts)


class NodeItem(QGraphicsItem):
    def __init__(self, node: Node, handle: str, theme: Theme,
                 on_click: Callable[["NodeItem", QPointF], None]) -> None:
        super().__init__()
        self.node, self.handle, self.theme = node, handle, theme
        self.on_click = on_click
        self.selected = False
        self.highlight = False
        self.hovered = False
        self.pins: list[tuple[int, float, float]] = []  # (number, rel_x, rel_y)
        self.math: str | None = None  # Unicode math for a final answer, set by the view
        self.setAcceptHoverEvents(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setZValue(1 if node.type == NodeType.CHECK else 2)
        self.setToolTip(node_tooltip(node, handle))

    # geometry ------------------------------------------------------------
    def size(self) -> tuple[float, float]:
        if self.node.type == NodeType.CHECK:
            return NODE_W - 2 * PILL_INSET, PILL_H
        if self.node.result and self.node.result.get("kind") == "plotspec":
            return NODE_W, PLOT_H
        return NODE_W, NODE_H

    def boundingRect(self) -> QRectF:  # noqa: N802 - Qt API
        w, h = self.size()
        return QRectF(-3, -3, w + 6, h + 6)

    def update_node(self, node: Node, handle: str) -> None:
        self.prepareGeometryChange()
        self.node, self.handle = node, handle
        self.setToolTip(node_tooltip(node, handle))
        self.update()

    # painting ------------------------------------------------------------
    def _chip(self, p: QPainter, right: float, top: float, status: str, label: str, h: float = CHIP_H) -> float:
        t = self.theme
        fg, bg = t.status_colors(status)
        p.setFont(t.ui_font("size_node_small_px", bold=True))
        fm = QFontMetrics(p.font())
        text = f"{STATUS_ICON.get(status, '')} {label}"
        w = fm.horizontalAdvance(text) + 14
        rect = QRectF(right - w, top, w, h)
        p.setBrush(QBrush(bg))
        p.setPen(QPen(fg, 1, Qt.PenStyle.DashLine) if status == "invalidated" else Qt.PenStyle.NoPen)
        p.drawRoundedRect(rect, h / 2, h / 2)
        p.setPen(fg)
        p.drawText(rect, Qt.AlignmentFlag.AlignCenter, text)
        return w

    def paint(self, p: QPainter, option: QStyleOptionGraphicsItem, widget: QWidget | None = None) -> None:
        t = self.theme
        n = self.node
        w, h = self.size()
        status = n.status.value
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setRenderHint(QPainter.RenderHint.TextAntialiasing)
        rect = QRectF(0.5, 0.5, w - 1, h - 1)
        radius = h / 2 if n.type == NodeType.CHECK else t.radius("node")

        if n.status == Status.INVALIDATED:
            pen = QPen(t.c("status_invalidated"), 1, Qt.PenStyle.DashLine)
            fill = t.c("status_invalidated_bg")
        else:
            pen = QPen(t.c("border_strong") if self.hovered else t.c("border"), 1)
            fill = t.c("panel")
        if self.highlight:
            pen = QPen(t.c("edge_conflict"), 1.75)
        if self.selected:
            pen = QPen(t.c("selection"), 1.75)
        p.setPen(pen)
        p.setBrush(QBrush(fill))
        p.drawRoundedRect(rect, radius, radius)

        if n.type == NodeType.CHECK:
            self._paint_pill(p, w, h)
            return

        chip_w = self._chip(p, w - PAD + 2, PAD - 2, status, status)

        # title row: handle (faint) + title 13px semibold, elided (tooltip has the full text)
        title_font = t.ui_font("size_node_title_px", bold=True)
        p.setFont(title_font)
        fm = QFontMetrics(title_font)
        avail = w - chip_w - 2 * PAD - 4
        handle_font = t.ui_font("size_node_small_px")
        handle_w = QFontMetrics(handle_font).horizontalAdvance(self.handle) + 6
        p.setFont(handle_font)
        p.setPen(t.c("text_faint"))
        p.drawText(QRectF(PAD, PAD - 2, handle_w, CHIP_H), Qt.AlignmentFlag.AlignVCenter, self.handle)
        p.setFont(title_font)
        p.setPen(t.c("text") if n.status != Status.INVALIDATED else t.c("text_muted"))
        p.drawText(QRectF(PAD + handle_w, PAD - 2, avail - handle_w, CHIP_H), Qt.AlignmentFlag.AlignVCenter,
                   fm.elidedText(n.title, Qt.TextElideMode.ElideRight, int(avail - handle_w)))

        # second row: tool / type (secondary), plus lock and warning marks
        marks = ("🔒 " if n.locked else "") + ("⚠ " if (n.warning or n.flags) else "")
        p.setFont(t.ui_font("size_node_body_px"))
        p.setPen(t.c("text_secondary") if not marks else t.c("warning"))
        sub = f"{marks}{n.tool_name or n.type.value}"
        inner = w - 2 * PAD
        p.drawText(QRectF(PAD, 31, inner, 16), Qt.AlignmentFlag.AlignVCenter,
                   QFontMetrics(p.font()).elidedText(sub, Qt.TextElideMode.ElideRight, int(inner)))

        if n.result and n.result.get("kind") == "plotspec":
            self._paint_plot(p, QRectF(PAD, 52, inner, h - 52 - PAD))
        else:
            if n.type == NodeType.FINAL and self.math:
                text, font = self.math, t.ui_font("size_node_title_px", bold=True)
            else:
                text = n.content if n.type in (NodeType.PROBLEM, NodeType.FINAL, NodeType.ERROR) \
                    else n.display_result()
                font = t.mono_font("size_node_body_px")
            p.setFont(font)
            p.setPen(t.c("text") if n.status != Status.INVALIDATED else t.c("text_faint"))
            p.drawText(QRectF(PAD, 49, inner, 17), Qt.AlignmentFlag.AlignVCenter,
                       QFontMetrics(font).elidedText(text, Qt.TextElideMode.ElideRight, int(inner)))

        for num, rx, ry in self.pins:
            c = QPointF(rx * w, ry * h)
            p.setPen(QPen(t.c("panel"), 1.5))
            p.setBrush(QBrush(t.c("pin")))
            p.drawEllipse(c, 8, 8)
            p.setPen(t.c("accent_text"))
            p.setFont(t.ui_font("size_small_px", bold=True))
            p.drawText(QRectF(c.x() - 8, c.y() - 8, 16, 16), Qt.AlignmentFlag.AlignCenter, str(num))

    def _paint_pill(self, p: QPainter, w: float, h: float) -> None:
        """Check node: a 24px pill under its target with the outcome on the right."""
        t = self.theme
        n = self.node
        outcome = (n.result or {}).get("outcome", "inconclusive")
        chip_status = {"pass": "verified", "fail": "failed"}.get(outcome, "inconclusive")
        cw = self._chip(p, w - 4, 4, chip_status, outcome, h=h - 8)
        font = t.ui_font("size_node_small_px")
        p.setFont(font)
        p.setPen(t.c("text_secondary"))
        label = f"{self.handle}  check · {n.title.split(': ', 1)[-1].replace('_', ' ')}"
        room = int(w - cw - PAD - 8)
        p.drawText(QRectF(PAD - 2, 0, room, h), Qt.AlignmentFlag.AlignVCenter,
                   QFontMetrics(font).elidedText(label, Qt.TextElideMode.ElideRight, room))

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
            if i == 0:
                path.moveTo(px, py)
            else:
                path.lineTo(px, py)
        p.setPen(QPen(self.theme.c("border"), 1))
        p.drawLine(area.bottomLeft(), area.bottomRight())
        p.setPen(QPen(self.theme.c("accent"), 1.5))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawPath(path)

    # interaction ---------------------------------------------------------
    def hoverEnterEvent(self, event) -> None:  # noqa: N802, ANN001
        self.hovered = True
        self.update()

    def hoverLeaveEvent(self, event) -> None:  # noqa: N802, ANN001
        self.hovered = False
        self.update()

    def mousePressEvent(self, event: QGraphicsSceneMouseEvent) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton:
            self.on_click(self, event.pos())
            event.accept()
        else:
            super().mousePressEvent(event)
