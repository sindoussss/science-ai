"""Live reasoning graph canvas."""
from __future__ import annotations

import math

from PyQt6.QtCore import QPointF, QRectF, Qt, pyqtSignal
from PyQt6.QtGui import QKeySequence, QMouseEvent, QPainter, QPen, QResizeEvent, QShortcut, QTransform, QWheelEvent
from PyQt6.QtWidgets import QFrame, QGraphicsScene, QGraphicsView, QToolButton

from sciai.graph.model import EdgeKind, Node, NodeType
from sciai.ui.canvas.edge_item import EdgeItem
from sciai.ui.canvas.layout import NODE_H, layered_positions
from sciai.ui.canvas.legend import Legend
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
        # View mode: "auto" (place after every layout), "manual" (user panned/zoomed),
        # "fit" (Fit button / Ctrl+0: may go below the floor until the next layout).
        self._mode = "auto"
        self._recent: str | None = None  # most recently added or updated node
        self.nodes: dict[str, Node] = {}
        self.handles: dict[str, str] = {}
        self.edges: list[tuple[str, str, EdgeKind]] = []
        self.items_: dict[str, NodeItem] = {}
        self.edge_items: list[tuple[EdgeItem, str, str]] = []
        self.selected: str | None = None
        self.pins: dict[str, list[tuple[int, float, float]]] = {}
        self.pin_editor = PinEditor(self.viewport())
        self.pin_editor.sent.connect(self._pin_sent)
        self.legend = Legend(theme, self)
        self.fit_btn = QToolButton(self)
        self.fit_btn.setObjectName("overlayButton")
        self.fit_btn.setText("Fit")
        self.fit_btn.setToolTip("Show the whole graph (Ctrl+0)")
        self.fit_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.fit_btn.clicked.connect(self.fit_all)
        QShortcut(QKeySequence("Ctrl+0"), self, activated=self.fit_all,
                  context=Qt.ShortcutContext.WindowShortcut)

    # model -----------------------------------------------------------------
    def load(self, nodes: list[tuple[Node, str]], edges: list[tuple[str, str, EdgeKind]],
             pins: dict[str, list[tuple[int, float, float]]] | None = None) -> None:
        self.nodes = {n.id: n for n, _ in nodes}
        self.handles = {n.id: h for n, h in nodes}
        self.edges = list(edges)
        self.pins = pins or {}
        self.selected = None
        self._recent = None
        self._mode = "auto"
        self._rebuild()

    def add_node(self, node: Node, handle: str, depends_on: list[str]) -> None:
        self.nodes[node.id] = node
        self.handles[node.id] = handle
        for d in depends_on:
            self.edges.append((node.id, d, EdgeKind.DEPENDS_ON))
        self._recent = node.id
        self._rebuild()
        if self._mode == "manual":
            self.ensureVisible(self.items_[node.id], 60, 60)

    def update_node(self, node: Node) -> None:
        if node.id not in self.nodes:
            return
        self.nodes[node.id] = node
        self._recent = node.id
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
        if self._mode == "auto":
            self._place()  # recenters on the new focus when the graph doesn't fit
        elif node_id in self.items_:
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
        if self._mode == "fit":
            self._mode = "auto"  # Fit lasts until the next layout
        if self._mode == "auto":
            self._place()

    def _math(self, node: Node) -> str:
        by_handle = {h: nid for nid, h in self.handles.items()}
        return answer_text(node, lambda h: self.nodes.get(by_handle.get(h, "")))

    MIN_SCALE = 0.92  # readable floor: node titles stay >= 12px on screen
    MAX_SCALE = 1.3
    DEFAULT_SCALE = 1.0
    FIT_PADDING = 48
    DOT_STEP = 20

    def scale_factor(self) -> float:
        return self.transform().m11()

    def _fit_scale(self) -> tuple[float, QRectF]:
        pad = self.FIT_PADDING
        rect = self.scene_.itemsBoundingRect().adjusted(-pad, -pad, pad, pad)
        vw, vh = self.viewport().width(), self.viewport().height()
        if vw <= 0 or vh <= 0 or rect.isEmpty():
            return self.DEFAULT_SCALE, rect
        return min(vw / rect.width(), vh / rect.height()), rect

    def _set_scale(self, s: float) -> None:
        self.setTransform(QTransform.fromScale(s, s))

    def focus_node(self) -> str | None:
        for nid in (self.selected, self._recent):
            if nid in self.items_:
                return nid
        return None

    def _place(self) -> None:
        """Auto/fit placement. Never shrinks below MIN_SCALE in auto mode: if the graph
        doesn't fit at the floor, show it at 1.0 centered on the focus node and let the user pan."""
        if not self.items_:
            return
        fit, rect = self._fit_scale()
        if self._mode == "fit":
            self._set_scale(min(fit, self.MAX_SCALE))
            self.centerOn(rect.center())
            return
        if fit >= self.MIN_SCALE:
            self._set_scale(min(fit, self.MAX_SCALE))
            self.centerOn(rect.center())
            return
        self._set_scale(self.DEFAULT_SCALE)
        focus = self.focus_node()
        target = self.items_[focus].sceneBoundingRect().center() if focus else rect.center()
        self.centerOn(self._clamp_center(target, rect))

    def _clamp_center(self, target: QPointF, bounds: QRectF) -> QPointF:
        """Keep the focus as central as possible without showing empty canvas past the graph's edges."""
        s = self.scale_factor()
        half_w, half_h = self.viewport().width() / s / 2, self.viewport().height() / s / 2

        def axis(t: float, lo: float, hi: float, half: float) -> float:
            if hi - lo <= 2 * half:
                return (lo + hi) / 2  # this axis fits: center it
            return min(max(t, lo + half), hi - half)

        return QPointF(axis(target.x(), bounds.left(), bounds.right(), half_w),
                       axis(target.y(), bounds.top(), bounds.bottom(), half_h))

    def fit(self) -> None:
        """Return to automatic placement (used after loading and on double-click)."""
        self._mode = "auto"
        self._place()

    def fit_all(self) -> None:
        """Fit button / Ctrl+0: show everything, even below the floor, until the next layout."""
        self._mode = "fit"
        self._place()

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
        self._mode = "manual"
        cur = self.scale_factor()
        factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
        # Never zoom out past the floor (a Fit below it may still be zoomed back in).
        target = max(min(self.MIN_SCALE, cur), min(self.MAX_SCALE, cur * factor))
        self.scale(target / cur, target / cur)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if event.buttons() & Qt.MouseButton.LeftButton:
            self._mode = "manual"  # the user is panning
        super().mouseMoveEvent(event)

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if self.itemAt(event.pos()) is None:  # double-click empty canvas: back to auto placement
            self.fit()
        else:
            super().mouseDoubleClickEvent(event)

    def resizeEvent(self, event: QResizeEvent) -> None:  # noqa: N802
        super().resizeEvent(event)
        m = 12
        hint = self.legend.sizeHint()
        self.legend.setGeometry(m, self.height() - hint.height() - m, self.width() - 2 * m, hint.height())
        fb = self.fit_btn.sizeHint()
        self.fit_btn.setGeometry(self.width() - fb.width() - m, m, fb.width(), fb.height())
        if self._mode in ("auto", "fit"):
            self._place()
