"""Thin overlay scrollbars: 8px, transparent track, no arrows, shown only on hover or while scrolling.

Qt has no overlay scrollbars outside macOS, so each scroll area's own bar is turned off (it keeps
working as the scroll model: wheel, keys and programmatic scrolling still drive it) and a mirror
bar is floated over the viewport's edge, kept in sync both ways. Installing is automatic: an
application event filter picks up every QAbstractScrollArea the first time it is shown.
"""
from __future__ import annotations

from PyQt6.QtCore import QEvent, QObject, Qt, QTimer
from PyQt6.QtWidgets import QAbstractScrollArea, QApplication, QGraphicsView, QScrollBar, QWidget

WIDTH = 8
EDGE = 2  # gap between the bar and the area's edge
LINGER_MS = 900  # stays visible this long after the last scroll


class OverlayBar(QScrollBar):
    def __init__(self, area: QAbstractScrollArea, orientation: Qt.Orientation) -> None:
        super().__init__(orientation, area)
        self.setObjectName("overlayBar")
        self.area = area
        self.src = area.verticalScrollBar() if orientation == Qt.Orientation.Vertical else area.horizontalScrollBar()
        self.setCursor(Qt.CursorShape.ArrowCursor)
        self._hover = False
        self._linger = QTimer(self)
        self._linger.setSingleShot(True)
        self._linger.setInterval(LINGER_MS)
        self._linger.timeout.connect(self._refresh)
        self.src.rangeChanged.connect(self._sync_range)
        self.src.valueChanged.connect(self._src_moved)
        self.valueChanged.connect(self._moved)
        self._sync_range()
        self.hide()

    def _sync_range(self, *_a: int) -> None:
        self.blockSignals(True)
        self.setRange(self.src.minimum(), self.src.maximum())
        self.setPageStep(self.src.pageStep())
        self.setSingleStep(self.src.singleStep())
        self.setValue(self.src.value())
        self.blockSignals(False)
        self._refresh()

    def _src_moved(self, v: int) -> None:
        # programmatic scrolls (stick-to-bottom) don't flash the bar; user scrolling does (see watcher)
        self.blockSignals(True)
        self.setValue(v)
        self.blockSignals(False)

    def _moved(self, v: int) -> None:
        self.src.setValue(v)

    def flash(self) -> None:
        """Show while scrolling, then fade out after a short linger."""
        self._linger.start()
        self._refresh()

    def set_hover(self, on: bool) -> None:
        self._hover = on
        self._refresh()

    def scrollable(self) -> bool:
        return self.maximum() > self.minimum()

    def _refresh(self) -> None:
        want = self.scrollable() and (self._hover or self._linger.isActive() or self.isSliderDown())
        if want:
            self.place()
            self.show()
            self.raise_()
        else:
            self.hide()

    def place(self) -> None:
        r = self.area.rect()
        if self.orientation() == Qt.Orientation.Vertical:
            self.setGeometry(r.width() - WIDTH - EDGE, EDGE, WIDTH, r.height() - 2 * EDGE)
        else:
            self.setGeometry(EDGE, r.height() - WIDTH - EDGE, r.width() - 2 * EDGE, WIDTH)


class _AreaWatcher(QObject):
    """Per-area: hover shows the bars, resizes re-place them."""

    def __init__(self, area: QAbstractScrollArea, bars: list[OverlayBar]) -> None:
        super().__init__(area)
        self.bars = bars
        area.installEventFilter(self)
        area.viewport().installEventFilter(self)

    def eventFilter(self, obj: QObject, event: QEvent) -> bool:  # noqa: N802
        t = event.type()
        if t in (QEvent.Type.Enter, QEvent.Type.HoverEnter):
            for b in self.bars:
                b.set_hover(True)
        elif t in (QEvent.Type.Leave, QEvent.Type.HoverLeave):
            area = self.parent()
            under = isinstance(area, QWidget) and area.underMouse()
            for b in self.bars:
                b.set_hover(under or b.underMouse())
        elif t in (QEvent.Type.Wheel, QEvent.Type.KeyPress):
            for b in self.bars:
                b.flash()
        elif t == QEvent.Type.Resize:
            for b in self.bars:
                if b.isVisible():
                    b.place()
        return False


def install(area: QAbstractScrollArea) -> list[OverlayBar]:
    """Replace the area's native bars with overlay bars (idempotent)."""
    if area.property("overlayBars") or isinstance(area, QGraphicsView):
        return []  # the canvas pans by dragging; it has no bars at all
    area.setProperty("overlayBars", True)
    bars = []
    for orient, policy, setter in (
            (Qt.Orientation.Vertical, area.verticalScrollBarPolicy(), area.setVerticalScrollBarPolicy),
            (Qt.Orientation.Horizontal, area.horizontalScrollBarPolicy(), area.setHorizontalScrollBarPolicy)):
        if policy == Qt.ScrollBarPolicy.ScrollBarAlwaysOff:
            continue
        setter(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        bars.append(OverlayBar(area, orient))
    if bars:
        _AreaWatcher(area, bars)
    return bars


class _Installer(QObject):
    def eventFilter(self, obj: QObject, event: QEvent) -> bool:  # noqa: N802
        if event.type() == QEvent.Type.Show and isinstance(obj, QAbstractScrollArea):
            install(obj)
        return False


_installer: _Installer | None = None


def install_everywhere(root: QWidget | None = None) -> None:
    """Overlay bars for every scroll area under ``root`` now, and for every one shown later."""
    global _installer
    app = QApplication.instance()
    if _installer is None and app is not None:
        _installer = _Installer(app)
        app.installEventFilter(_installer)
    if root is not None:
        for area in root.findChildren(QAbstractScrollArea):
            install(area)
