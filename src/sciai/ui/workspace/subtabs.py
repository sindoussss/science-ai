"""Sub-tab row of the node view (Code | Execution Log | Messages | Environment | Review).

A row of text buttons with exact spacing instead of a QTabBar: QTabBar adds per-tab style
margins that can't be set from QSS, so the 18px gaps and 16px side padding couldn't be held.
The active tab is blue with a 2px underline under its label. The API mirrors the parts of
QTabWidget the panel and tests use (count, tabText, currentIndex, setCurrentIndex, currentChanged,
setTabIcon, widget).

When the row is too narrow (a workspace dragged toward its minimum), the gaps shrink to 12px,
then the side padding to 12px, then the longest label elides; labels never overlap or clip.
"""
from __future__ import annotations

import math

from PyQt6.QtCore import QRect, QSize, Qt, pyqtSignal
from PyQt6.QtGui import QFontMetricsF, QIcon, QPainter, QPaintEvent
from PyQt6.QtWidgets import QButtonGroup, QSizePolicy, QStackedWidget, QToolButton, QVBoxLayout, QWidget

from sciai.ui.theme.theme import Theme, keep_fractional_width

GAP = 18
MIN_GAP = 12
SIDE = 16
MIN_SIDE = 12  # only at the narrowest workspace drag, after the gaps are down to 12
ROW_H = 40
ICON = 14
ICON_SPACING = 4  # the style's space between a tool button's icon and its text


class _TabButton(QToolButton):
    """A tab label whose size is exactly its drawn content (icon + text), with no style padding,
    so the gaps the row sets are the real gaps between labels."""

    def content_width(self, text: str | None = None) -> int:
        self.ensurePolished()
        w = QFontMetricsF(self.font()).horizontalAdvance(self.text() if text is None else text)
        if not self.icon().isNull():
            w += ICON + ICON_SPACING
        return math.ceil(w) + 1

    def sizeHint(self) -> QSize:  # noqa: N802
        return QSize(self.content_width(), ROW_H - 2)

    def minimumSizeHint(self) -> QSize:  # noqa: N802
        return self.sizeHint()


class _TabRow(QWidget):
    def __init__(self, owner: "SubTabs") -> None:
        super().__init__()
        self.owner = owner
        self.setFixedHeight(ROW_H)
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)

    def resizeEvent(self, event) -> None:  # noqa: N802, ANN001
        super().resizeEvent(event)
        self.owner.relayout()

    def paintEvent(self, event: QPaintEvent) -> None:  # noqa: N802
        t = self.owner.theme
        p = QPainter(self)
        p.fillRect(0, self.height() - 1, self.width(), 1, t.c("border"))
        b = self.owner.buttons[self.owner.currentIndex()] if self.owner.buttons else None
        if b is not None:
            r = self.owner.label_rect(self.owner.currentIndex())
            p.fillRect(r.left(), self.height() - 2, r.width(), 2, t.c("text"))
        p.end()


class SubTabs(QWidget):
    currentChanged = pyqtSignal(int)  # noqa: N815 - mirrors QTabWidget

    def __init__(self, theme: Theme) -> None:
        super().__init__()
        self.theme = theme
        self.buttons: list[_TabButton] = []
        self.labels: list[str] = []
        self.group = QButtonGroup(self)
        self.group.setExclusive(True)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        self.row = _TabRow(self)
        self.stack = QStackedWidget()
        lay.addWidget(self.row)
        lay.addWidget(self.stack, 1)
        self.gap = GAP
        self.side = SIDE

    # QTabWidget-like API ------------------------------------------------------------
    def addTab(self, widget: QWidget, label: str) -> int:  # noqa: N802
        i = len(self.buttons)
        b = _TabButton(self.row)
        b.setObjectName("subTab")
        b.setCheckable(True)
        b.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        b.setIconSize(QSize(ICON, ICON))
        b.setCursor(Qt.CursorShape.PointingHandCursor)
        b.setText(label)
        keep_fractional_width(b, self.theme.px("size_tab_px"))
        b.clicked.connect(lambda _=False, k=i: self.setCurrentIndex(k))
        self.group.addButton(b)
        self.buttons.append(b)
        self.labels.append(label)
        self.stack.addWidget(widget)
        if i == 0:
            b.setChecked(True)
        self.relayout()
        return i

    def count(self) -> int:
        return len(self.buttons)

    def tabText(self, i: int) -> str:  # noqa: N802
        return self.labels[i]

    def widget(self, i: int) -> QWidget:
        return self.stack.widget(i)

    def currentIndex(self) -> int:  # noqa: N802
        return self.stack.currentIndex()

    def setCurrentIndex(self, i: int) -> None:  # noqa: N802
        if not 0 <= i < len(self.buttons):
            return
        self.buttons[i].setChecked(True)
        changed = i != self.stack.currentIndex()
        self.stack.setCurrentIndex(i)
        self.row.update()
        if changed:
            self.currentChanged.emit(i)

    def setTabIcon(self, i: int, ic: QIcon) -> None:  # noqa: N802
        self.buttons[i].setIcon(ic)
        self.relayout()

    # layout ------------------------------------------------------------------------
    @staticmethod
    def _natural(b: _TabButton, text: str) -> int:
        """Width of the full label as drawn (icon included)."""
        return b.content_width(text)

    def needed_width(self, gap: int = GAP) -> int:
        return 2 * SIDE + sum(self._natural(b, t) for b, t in zip(self.buttons, self.labels)) + \
            gap * max(0, len(self.buttons) - 1)

    def relayout(self) -> None:
        if not self.buttons:
            return
        avail = self.row.width()
        natural = [self._natural(b, t) for b, t in zip(self.buttons, self.labels)]
        n_gaps = max(1, len(self.buttons) - 1)
        room = avail - 2 * SIDE - sum(natural)
        self.gap = max(MIN_GAP, min(GAP, room // n_gaps)) if room > 0 else MIN_GAP
        # still too wide at 12px gaps: give up side padding (down to 12), then elide the widest label
        n_real = len(self.buttons) - 1
        over = 2 * SIDE + sum(natural) + self.gap * n_real - avail
        self.side = SIDE - min(SIDE - MIN_SIDE, max(0, (over + 1) // 2))
        widths = list(natural)
        over = 2 * self.side + sum(widths) + self.gap * n_real - avail
        if over > 0:  # elide the widest label just enough
            k = max(range(len(widths)), key=lambda i: widths[i])
            widths[k] = max(24, widths[k] - over)
        x = self.side
        for b, text, w, nat in zip(self.buttons, self.labels, widths, natural):
            shown = text
            if w < nat:
                icon_w = 0 if b.icon().isNull() else ICON + ICON_SPACING
                shown = QFontMetricsF(b.font()).elidedText(text, Qt.TextElideMode.ElideRight, w - icon_w - 1)
            b.setText(shown)
            b.setToolTip(text if shown != text else "")
            b.setGeometry(QRect(x, 0, w, ROW_H - 2))
            x += w + self.gap
        self.row.update()

    def label_rect(self, i: int) -> QRect:
        """The drawn label (text, plus the icon when there is one) in row coordinates: buttons are
        sized to exactly their content, so this is the button's geometry."""
        return self.buttons[i].geometry()

    def label_gaps(self) -> list[int]:
        rs = [self.label_rect(i) for i in range(self.count())]
        return [b.left() - (a.right() + 1) for a, b in zip(rs, rs[1:])]
