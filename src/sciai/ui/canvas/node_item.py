"""A graph node on the canvas. Status uses a soft chip with an icon AND a text label."""
from __future__ import annotations

from typing import Callable

from PyQt6.QtCore import QPointF, QRectF, Qt
from PyQt6.QtGui import QBrush, QFontMetricsF, QPainter, QPainterPath, QPen
from PyQt6.QtWidgets import QGraphicsItem, QGraphicsSceneMouseEvent, QStyleOptionGraphicsItem, QWidget

from sciai.graph.model import Node, NodeType, Status
from sciai.ui.canvas.layout import NODE_H, NODE_W, PILL_H, PILL_INSET, PLOT_H
from sciai.ui.canvas.pins import PIN_D
from sciai.ui.chips import chip_label, paint_chip
from sciai.ui.theme.theme import Theme

PAD = 12
CHIP_H = 20  # status chip height (12px text + 2px/8px padding); it straddles the top border
TITLE_Y, TITLE_H = 12, 18  # 12px top padding
BODY_Y, BODY_H = 34, 18  # ends at 52 = 64 - 12 bottom padding


# Nodes whose body row shows their text rather than a tool result.
TEXT_TYPES = (NodeType.PROBLEM, NodeType.FINAL, NodeType.ERROR, NodeType.ASSUMPTION, NodeType.ENTITY)


def assumption_chip(node: Node) -> tuple[str, str]:
    """(chip status style, label) for an assumption: it is never verified, only assumed or rejected."""
    if node.status == Status.FAILED:
        return "failed", "rejected"
    if node.status == Status.INVALIDATED:
        return "invalidated", "invalidated"
    return "inconclusive", "assumed"


def node_tooltip(node: Node, handle: str) -> str:
    parts = [f"{handle} · {node.title}", node.tool_name or node.type.value, f"status: {node.status.value}"]
    body = node.content if node.type in TEXT_TYPES else node.display_result()
    if body:
        parts.append(body[:400])
    return "\n".join(parts)



def body_prefix(n: Node, handle: str) -> str:
    """Faint words before a node's value: its handle, then "locked" and "flagged" when they apply.
    Assumptions are always locked, so they don't repeat it."""
    words = [handle]
    if n.locked and n.type != NodeType.ASSUMPTION:
        words.append("locked")
    if n.warning or n.flags:
        words.append("flagged")
    return " · ".join(words)

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
        self.pending_pin: tuple[int, float, float] | None = None  # being written in the pin popover
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
        # the status chip straddles the top; pins (22px) may sit on any edge
        m = PIN_D / 2 + 2
        return QRectF(-m, -max(CHIP_H / 2 + 2, m), w + 2 * m, h + max(CHIP_H / 2 + 2, m) + m)

    def title_layout(self) -> tuple[str, bool]:
        """(text drawn in the title row, whether the full title fits). Used by paint and the metrics test."""
        font = self.theme.ui_font("size_node_title_px", bold=True)
        fm = QFontMetricsF(font)
        room = self.size()[0] - 2 * PAD
        if fm.horizontalAdvance(self.node.title) <= room:
            return self.node.title, True
        return fm.elidedText(self.node.title, Qt.TextElideMode.ElideRight, room), False

    def update_node(self, node: Node, handle: str) -> None:
        self.prepareGeometryChange()
        self.node, self.handle = node, handle
        self.setToolTip(node_tooltip(node, handle))
        self.update()

    # painting ------------------------------------------------------------
    def _chip(self, p: QPainter, right: float, cy: float, status: str, label: str) -> float:
        """Status chip with its right edge at ``right``, centered on ``cy``; returns its width."""
        return paint_chip(p, self.theme, right, cy, status, chip_label(status, label), right_aligned=True).width()

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
        elif n.type == NodeType.ASSUMPTION:
            # outlined: canvas fill, so an assumption never reads as a computed result
            pen = QPen(t.c("border_strong"), 1.25)
            fill = t.c("canvas")
        else:
            pen = QPen(t.c("border_strong") if self.hovered else t.c("border"), 1)
            fill = t.c("panel")
        if self.highlight:  # a conflict: dark dashed outline, never a colored one
            pen = QPen(t.c("selection"), 1.75, Qt.PenStyle.DashLine)
        if self.selected:
            pen = QPen(t.c("selection"), 1.75)
        p.setPen(pen)
        p.setBrush(QBrush(fill))
        p.drawRoundedRect(rect, radius, radius)

        if n.type == NodeType.CHECK:
            self._paint_pill(p, w, h)
            return

        # Status chip straddles the top border (right), so the title row keeps the full inner width.
        if n.type == NodeType.ASSUMPTION:
            self._chip(p, w - PAD, 0, *assumption_chip(n))
        else:
            self._chip(p, w - PAD, 0, status, status)

        # Title row: 13px semibold, full inner width; elided only past it (tooltip has the full text).
        shown, _fits = self.title_layout()
        p.setFont(t.ui_font("size_node_title_px", bold=True))
        p.setPen(t.c("text") if n.status != Status.INVALIDATED else t.c("text_muted"))
        p.drawText(QRectF(PAD, TITLE_Y, w - 2 * PAD, TITLE_H), Qt.AlignmentFlag.AlignVCenter, shown)

        # Body row: handle and state words (faint) then the value (mono 12px) or the answer as math text.
        x = float(PAD)
        inner_right = w - PAD
        small = t.ui_font("size_node_small_px")
        prefix = body_prefix(n, self.handle)
        p.setFont(small)
        p.setPen(t.c("text_faint"))
        tw = QFontMetricsF(small).horizontalAdvance(prefix)
        p.drawText(QRectF(x, BODY_Y, tw + 1, BODY_H), Qt.AlignmentFlag.AlignVCenter, prefix)
        x += tw + 6

        if n.result and n.result.get("kind") == "plotspec":
            self._paint_plot(p, QRectF(PAD, BODY_Y + BODY_H + 4, w - 2 * PAD, h - BODY_Y - BODY_H - 4 - PAD))
        else:
            if n.type == NodeType.FINAL and self.math:
                text, font = self.math, t.ui_font("size_node_title_px", bold=True)
            elif n.type == NodeType.ASSUMPTION:  # the title carries the text
                source = (n.tool_inputs or {}).get("source")
                text = "default" if source == "default" else "suggested"
                font = t.ui_font("size_node_small_px")
            else:
                text = n.content if n.type in TEXT_TYPES else n.display_result()
                font = t.mono_font("size_node_body_px")
            p.setFont(font)
            p.setPen(t.c("text") if n.status != Status.INVALIDATED else t.c("text_faint"))
            room = inner_right - x
            p.drawText(QRectF(x, BODY_Y, room, BODY_H), Qt.AlignmentFlag.AlignVCenter,
                       QFontMetricsF(font).elidedText(text, Qt.TextElideMode.ElideRight, room))

        for num, rx, ry in [*self.pins, *([self.pending_pin] if self.pending_pin else [])]:
            c = QPointF(rx * w, ry * h)
            r = PIN_D / 2
            p.setPen(QPen(t.c("panel"), 1.5))
            p.setBrush(QBrush(t.c("pin")))
            p.drawEllipse(c, r, r)
            p.setPen(t.c("accent_text"))
            p.setFont(t.ui_font("size_small_px", bold=True))
            p.drawText(QRectF(c.x() - r, c.y() - r, PIN_D, PIN_D), Qt.AlignmentFlag.AlignCenter, str(num))

    def _paint_pill(self, p: QPainter, w: float, h: float) -> None:
        """Check node: a 22px pill under its target with the outcome on the right."""
        t = self.theme
        n = self.node
        outcome = (n.result or {}).get("outcome", "inconclusive")
        chip_status = {"pass": "verified", "fail": "failed"}.get(outcome, "inconclusive")
        cw = self._chip(p, w - 3, h / 2, chip_status, outcome)
        font = t.ui_font("size_node_small_px")
        p.setFont(font)
        p.setPen(t.c("text_secondary"))
        label = f"{self.handle}  check · {n.title.split(': ', 1)[-1].replace('_', ' ')}"
        room = w - cw - 10 - 6
        p.drawText(QRectF(10, 0, room, h), Qt.AlignmentFlag.AlignVCenter,
                   QFontMetricsF(font).elidedText(label, Qt.TextElideMode.ElideRight, room))

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
