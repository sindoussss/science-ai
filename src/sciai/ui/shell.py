"""Window shell: app background, floating cards with an optional soft shadow, hairline splitters."""
from __future__ import annotations

from PyQt6.QtCore import QEvent, QObject, QPoint, QRectF, Qt
from PyQt6.QtGui import QBrush, QColor, QPainter, QPaintEvent
from PyQt6.QtWidgets import QFrame, QLabel, QSizePolicy, QSplitter, QSplitterHandle, QVBoxLayout, QWidget

from sciai.ui.theme.theme import Theme


class Card(QFrame):
    """White card: 12px radius and a 1px border (QSS #card)."""

    def __init__(self, content: QWidget) -> None:
        super().__init__()
        self.setObjectName("card")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(1, 1, 1, 1)
        lay.setSpacing(0)
        lay.addWidget(content)


class AppRoot(QWidget):
    """Paints the app background and, when enabled, a very soft shadow under each card
    (0 1px 2px rgba(0,0,0,0.04)). Painting it here is cheap; a QGraphicsEffect on a
    card would re-render the whole card, graph canvas included, on every update."""

    def __init__(self, theme: Theme) -> None:
        super().__init__()
        self.setObjectName("appRoot")
        self.theme = theme
        self.cards: list[Card] = []

    def track(self, card: Card) -> Card:
        self.cards.append(card)
        card.installEventFilter(self)
        return card

    def eventFilter(self, obj: QObject, event: QEvent) -> bool:  # noqa: N802
        if event.type() in (QEvent.Type.Move, QEvent.Type.Resize, QEvent.Type.Show, QEvent.Type.Hide):
            self.update()
        return False

    def paintEvent(self, event: QPaintEvent) -> None:  # noqa: N802
        p = QPainter(self)
        p.fillRect(self.rect(), self.theme.c("app_bg"))
        if self.theme.effect("card_shadow"):
            p.setRenderHint(QPainter.RenderHint.Antialiasing)
            p.setPen(Qt.PenStyle.NoPen)
            base = self.theme.c("shadow")
            r = self.theme.radius("card")
            for card in self.cards:
                if not card.isVisible():
                    continue
                top_left = card.mapTo(self, QPoint(0, 0))
                rect = QRectF(top_left.x(), top_left.y(), card.width(), card.height())
                # y-offset 1px, blur ~2px: two layers, the outer one fainter
                for grow, alpha in ((1.5, 0.5), (0.5, 1.0)):
                    c = QColor(base)
                    c.setAlphaF(base.alphaF() * alpha)
                    p.setBrush(QBrush(c))
                    p.drawRoundedRect(rect.translated(0, 1).adjusted(-grow, -grow + 0.5, grow, grow),
                                      r + grow, r + grow)
        p.end()


class _HairlineHandle(QSplitterHandle):
    def __init__(self, orientation: Qt.Orientation, parent: QSplitter, color: QColor) -> None:
        super().__init__(orientation, parent)
        self.color = color

    def paintEvent(self, event: QPaintEvent) -> None:  # noqa: N802
        p = QPainter(self)
        if self.orientation() == Qt.Orientation.Vertical:
            p.fillRect(0, self.height() // 2, self.width(), 1, self.color)
        else:
            p.fillRect(self.width() // 2, 0, 1, self.height(), self.color)
        p.end()


class HairlineSplitter(QSplitter):
    """A splitter whose handle looks like a 1px rule but has a 7px grab area.

    With ``last_fraction`` set (two panes), the last pane keeps that share of the
    splitter's height across resizes, and dragging can never push it above
    ``max_last_fraction``."""

    def __init__(self, orientation: Qt.Orientation, theme: Theme, last_fraction: float | None = None,
                 max_last_fraction: float = 1.0) -> None:
        super().__init__(orientation)
        self.color = theme.c("border")
        self.setHandleWidth(7)
        self.setChildrenCollapsible(False)
        self.max_last_fraction = max_last_fraction
        self.last_fraction = last_fraction
        self.splitterMoved.connect(self._moved)

    def createHandle(self) -> QSplitterHandle:  # noqa: N802
        h = _HairlineHandle(self.orientation(), self, self.color)
        h.setCursor(Qt.CursorShape.SplitVCursor if self.orientation() == Qt.Orientation.Vertical
                    else Qt.CursorShape.SplitHCursor)
        return h

    def _length(self) -> int:
        return self.height() if self.orientation() == Qt.Orientation.Vertical else self.width()

    def _apply_fraction(self) -> None:
        if self.last_fraction is None or self.count() != 2:
            return
        total = self._length()
        avail = total - self.handleWidth()
        if avail <= 0:
            return
        frac = min(self.last_fraction, self.max_last_fraction)
        last = int(frac * total)  # a share of the whole column, handle included
        self.blockSignals(True)
        self.setSizes([avail - last, last])
        self.blockSignals(False)

    def _moved(self, _pos: int, _index: int) -> None:
        if self.last_fraction is None or self.count() != 2:
            return
        total = self._length()
        if total > 0:
            self.last_fraction = min(self.sizes()[1] / total, self.max_last_fraction)
            self._apply_fraction()

    def resizeEvent(self, event) -> None:  # noqa: N802, ANN001
        super().resizeEvent(event)
        self._apply_fraction()

    def showEvent(self, event) -> None:  # noqa: N802, ANN001
        super().showEvent(event)
        self._apply_fraction()


class ElidedLabel(QLabel):
    """A one-line label that shrinks with an ellipsis instead of forcing its width on the layout."""

    def __init__(self, text: str) -> None:
        super().__init__()
        self.full = text
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.setToolTip(text)

    def resizeEvent(self, event) -> None:  # noqa: N802, ANN001
        super().resizeEvent(event)
        self.setText(self.fontMetrics().elidedText(self.full, Qt.TextElideMode.ElideRight, self.width()))
