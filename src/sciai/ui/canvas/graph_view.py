"""Live reasoning graph canvas."""
from __future__ import annotations

from PyQt6.QtCore import QPointF, Qt, pyqtSignal
from PyQt6.QtGui import QMouseEvent, QPainter, QResizeEvent, QWheelEvent
from PyQt6.QtWidgets import QFrame, QGraphicsScene, QGraphicsView

from sciai.graph.model import EdgeKind, Node, NodeType
from sciai.ui.canvas.edge_item import EdgeItem
from sciai.ui.canvas.layout import layered_positions
from sciai.ui.canvas.node_item import NodeItem
from sciai.ui.canvas.pins import PinEditor
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
        self.fit()

    def add_node(self, node: Node, handle: str, depends_on: list[str]) -> None:
        self.nodes[node.id] = node
        self.handles[node.id] = handle
        for d in depends_on:
            self.edges.append((node.id, d, EdgeKind.DEPENDS_ON))
        self._rebuild()
        if self._auto_fit:
            self.fit()
        else:
            self.ensureVisible(self.items_[node.id], 60, 60)

    def update_node(self, node: Node) -> None:
        if node.id not in self.nodes:
            return
        self.nodes[node.id] = node
        item = self.items_.get(node.id)
        if item is not None:
            item.update_node(node, self.handles[node.id])

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
        if node_id in self.items_:
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
            x, y = pos.get(nid, (0.0, 0.0))
            item.setPos(x, y)
            self.scene_.addItem(item)
            self.items_[nid] = item
        for src, dst, kind in self.edges:
            a, b = self.items_.get(src), self.items_.get(dst)
            if a is None or b is None:
                continue
            e = EdgeItem(kind, self.theme)
            aw, ah = a.size()
            bw, bh = b.size()
            if kind == EdgeKind.CHECKS:
                e.set_ends(a.pos() + QPointF(aw / 2, 0), b.pos() + QPointF(bw / 2, bh), vertical=True)
            else:
                e.set_ends(a.pos() + QPointF(0, ah / 2), b.pos() + QPointF(bw, bh / 2))
            self.scene_.addItem(e)
            self.edge_items.append((e, src, dst))
        self.scene_.setSceneRect(self.scene_.itemsBoundingRect().adjusted(-2000, -2000, 2000, 2000))

    MAX_FIT_SCALE = 1.25

    def fit(self) -> None:
        """Fit the whole graph in the canvas (capped so a tiny graph isn't blown up)."""
        if not self.items_:
            return
        rect = self.scene_.itemsBoundingRect().adjusted(-28, -28, 28, 28)
        self.resetTransform()
        vw, vh = self.viewport().width(), self.viewport().height()
        if vw <= 0 or vh <= 0:
            return
        scale = min(vw / rect.width(), vh / rect.height(), self.MAX_FIT_SCALE)
        self.scale(scale, scale)
        self.centerOn(rect.center())

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
