"""Live reasoning graph canvas."""
from __future__ import annotations

import math

from PyQt6.QtCore import QPointF, QRectF, Qt, pyqtSignal
from PyQt6.QtGui import QMouseEvent, QPainter, QPen, QResizeEvent, QWheelEvent
from PyQt6.QtWidgets import QFrame, QGraphicsScene, QGraphicsView

from sciai.graph.model import EdgeKind, Node, NodeType
from sciai.ui.canvas.edge_item import EdgeItem
from sciai.ui.canvas.layout import NODE_H, layered_positions
from sciai.ui.canvas.node_item import NodeItem
from sciai.ui.canvas.pins import PinEditor
from sciai.ui.mathtext import answer_text
from sciai.ui.theme.theme import Theme


class GraphView(QGraphicsView):
    node_selected = pyqtSignal(str)
    pin_sent = pyqtSignal(str, str, int, float, float)

    def __init__(self, theme: Theme) -> None:
        super().__init__()
        self.theme = theme
        self.scene_ = QGraphicsScene(self)
        self.setScene(self.scene_)
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setRenderHint(QPainter.RenderHint.TextAntialiasing)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setViewportUpdateMode(QGraphicsView.ViewportUpdateMode.FullViewportUpdate)
        self._auto_fit = True  # until the user pans or zooms
        self.nodes: dict[str, Node] = {}
        self.handles: dict[str, str] = {}
        self.edges: list[tuple[str, str, EdgeKind]] = []
        self.items_: dict[str, NodeItem] = {}
        self.edge_items: list[tuple[EdgeItem, str, str]] = []
        self.selected: str | None = None
        self.pins: dict[str, list[tuple[int, float, float]]] = {}
        self.pin_editor = PinEditor(self.viewport())
        self.pin_editor.sent.connect(self._pin_sent)

    # model -----------------------------------------------------------------
    def load(self, nodes: list[tuple[Node, str]], edges: list[tuple[str, str, EdgeKind]],
             pins: dict[str, list[tuple[int, float, float]]] | None = None) -> None:
        self.nodes = {n.id: n for n, _ in nodes}
        self.handles = {n.id: h for n, h in nodes}
        self.edges = list(edges)
        self.pins = pins or {}
        self.selected = None
        self._auto_fit = True
        self._rebuild()

    def add_node(self, node: Node, handle: str, depends_on: list[str]) -> None:
        self.nodes[node.id] = node
        self.handles[node.id] = handle
        for d in depends_on:
            self.edges.append((node.id, d, EdgeKind.DEPENDS_ON))
        self._rebuild()
        if not self._auto_fit:
            self.ensureVisible(self.items_[node.id], 60, 60)

    def update_node(self, node: Node) -> None:
        if node.id not in self.nodes:
            return
        self.nodes[node.id] = node
        item = self.items_.get(node.id)
        if item is not None:
            item.update_node(node, self.handles[node.id])
            if node.type == NodeType.FINAL:
                item.math = self._math(node)

    def remove_node(self, node_id: str) -> None:
        self.nodes.pop(node_id, None)
        self.edges = [e for e in self.edges if node_id not in (e[0], e[1])]
        self._rebuild()

    def add_edge(self, src: str, dst: str, kind: EdgeKind) -> None:
        if (src, dst, kind) not in self.edges:
            self.edges.append((src, dst, kind))
            self._rebuild()

    def highlight(self, ids: list[str]) -> None:
        for nid, item in self.items_.items():
            item.highlight = nid in ids
            item.update()

    def select(self, node_id: str | None) -> None:
        self.selected = node_id
        for nid, item in self.items_.items():
            item.selected = nid == node_id
            item.update()
        if node_id in self.items_ and not self._auto_fit:
            self.ensureVisible(self.items_[node_id], 60, 60)

    # drawing ---------------------------------------------------------------
    def _rebuild(self) -> None:
        self.scene_.clear()
        self.items_.clear()
        self.edge_items.clear()
        pos = layered_positions(self.nodes, self.edges)
        for nid, node in self.nodes.items():
            item = NodeItem(node, self.handles.get(nid, "?"), self.theme, self._clicked)
            item.pins = self.pins.get(nid, [])
            item.selected = nid == self.selected
            if node.type == NodeType.FINAL:
                item.math = self._math(node)
            x, y = pos.get(nid, (0.0, 0.0))
            item.setPos(x, y)
            self.scene_.addItem(item)
            self.items_[nid] = item
        for src, dst, kind in self.edges:
            a, b = self.items_.get(src), self.items_.get(dst)
            if a is None or b is None or kind == EdgeKind.CHECKS:
                continue  # a check pill sits directly under its target; no connector needed
            e = EdgeItem(kind, self.theme)
            ah = a.size()[1]
            bw = b.size()[0]
            # attach at mid-height of the node body (a plot node is taller)
            e.set_ends(a.pos() + QPointF(0, min(ah, NODE_H) / 2), b.pos() + QPointF(bw, min(b.size()[1], NODE_H) / 2))
            self.scene_.addItem(e)
            self.edge_items.append((e, src, dst))
        self.scene_.setSceneRect(self.scene_.itemsBoundingRect().adjusted(-2000, -2000, 2000, 2000))
        if self._auto_fit:
            self.fit()

    def _math(self, node: Node) -> str:
        by_handle = {h: nid for nid, h in self.handles.items()}
        return answer_text(node, lambda h: self.nodes.get(by_handle.get(h, "")))

    MAX_FIT_SCALE = 1.3
    FIT_PADDING = 48
    DOT_STEP = 20

    def fit(self) -> None:
        """fitInView(itemsBoundingRect + padding), capped so a small graph isn't blown up."""
        if not self.items_:
            return
        pad = self.FIT_PADDING
        rect = self.scene_.itemsBoundingRect().adjusted(-pad, -pad, pad, pad)
        self.resetTransform()
        vw, vh = self.viewport().width(), self.viewport().height()
        if vw <= 0 or vh <= 0:
            return
        scale = min(vw / rect.width(), vh / rect.height(), self.MAX_FIT_SCALE)
        self.scale(scale, scale)
        self.centerOn(rect.center())

    def drawBackground(self, p: QPainter, rect: QRectF) -> None:  # noqa: N802
        """Canvas fill plus a faint 1px dot grid every 20 scene px (skipped when zoomed far out)."""
        p.fillRect(rect, self.theme.c("canvas"))
        step = self.DOT_STEP
        if self.transform().m11() * step < 8:
            return
        pen = QPen(self.theme.c("canvas_dot"), 1.0)
        pen.setCosmetic(True)  # stays 1 device px at any zoom
        p.setPen(pen)
        x0 = math.floor(rect.left() / step) * step
        y0 = math.floor(rect.top() / step) * step
        pts = [QPointF(x, y)
               for x in range(int(x0), int(rect.right()) + 1, step)
               for y in range(int(y0), int(rect.bottom()) + 1, step)]
        p.drawPoints(pts)

    # interaction -----------------------------------------------------------
    def _clicked(self, item: NodeItem, local: QPointF) -> None:
        nid = item.node.id
        if self.selected == nid and item.node.type != NodeType.CHECK:
            w, h = item.size()
            number = max([p[0] for p in self.pins.get(nid, [])] + [0]) + 1
            rx, ry = max(0.0, min(1.0, local.x() / w)), max(0.0, min(1.0, local.y() / h))
            view_pt = self.mapFromScene(item.mapToScene(local))
            self.pin_editor.move(min(view_pt.x() + 12, self.viewport().width() - 270),
                                 min(view_pt.y() + 12, self.viewport().height() - 150))
            self.pin_editor.open_for(nid, self.handles.get(nid, ""), number, rx, ry)
            return
        self.select(nid)
        self.node_selected.emit(nid)

    def _pin_sent(self, node_id: str, text: str, number: int, rx: float, ry: float) -> None:
        self.pins.setdefault(node_id, []).append((number, rx, ry))
        if node_id in self.items_:
            self.items_[node_id].pins = self.pins[node_id]
            self.items_[node_id].update()
        self.pin_sent.emit(node_id, text, number, rx, ry)

    def wheelEvent(self, event: QWheelEvent) -> None:  # noqa: N802
        self._auto_fit = False
        factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
        self.scale(factor, factor)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if event.buttons() & Qt.MouseButton.LeftButton:
            self._auto_fit = False  # the user is panning
        super().mouseMoveEvent(event)

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if self.itemAt(event.pos()) is None:  # double-click empty canvas: fit again
            self._auto_fit = True
            self.fit()
        else:
            super().mouseDoubleClickEvent(event)

    def resizeEvent(self, event: QResizeEvent) -> None:  # noqa: N802
        super().resizeEvent(event)
        if self._auto_fit:
            self.fit()
