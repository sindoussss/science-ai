"""Sub-tab row of the node view (Code | Data | Plot | Execution Log | Messages | Environment | Review;
Data and Plot only for nodes that have them).

A row of text buttons with exact spacing instead of a QTabBar: QTabBar adds per-tab style
margins that can't be set from QSS, so the 18px gaps and 16px side padding couldn't be held.
The active tab is blue with a 2px underline under its label. The API mirrors the parts of
QTabWidget the panel and tests use (count, tabText, currentIndex, setCurrentIndex, currentChanged,
setTabIcon, widget).

When the row is too narrow (a workspace dragged toward its minimum), the gaps shrink to 12px,
then the side padding to 12px, then tabs that have a short label use it ("Log" for "Execution
Log", needed when Data or Plot is shown at the narrowest workspace), then the longest label
elides; labels never overlap or clip.
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
        self.hidden: set[int] = set()
        self.short: dict[int, str] = {}

    # QTabWidget-like API ------------------------------------------------------------
    def addTab(self, widget: QWidget, label: str, short: str | None = None) -> int:  # noqa: N802
        """``short`` is drawn instead of ``label`` when the row is too narrow for every full label."""
        i = len(self.buttons)
        if short:
            self.short[i] = short
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
        if not 0 <= i < len(self.buttons) or i in self.hidden:
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

    def indexOf(self, label: str) -> int:  # noqa: N802
        return self.labels.index(label) if label in self.labels else -1

    def isTabVisible(self, i: int) -> bool:  # noqa: N802
        return i not in self.hidden

    def setTabVisible(self, i: int, visible: bool) -> None:  # noqa: N802
        """Show or hide a tab (Data and Plot appear only for nodes that have them). Hiding the
        current tab moves to the first visible one."""
        if visible == (i not in self.hidden):
            return
        if visible:
            self.hidden.discard(i)
            self.buttons[i].show()
        else:
            self.hidden.add(i)
            self.buttons[i].hide()
            if self.currentIndex() == i:
                self.setCurrentIndex(next(k for k in range(len(self.buttons)) if k not in self.hidden))
        self.relayout()

    def visible_indexes(self) -> list[int]:
        return [i for i in range(len(self.buttons)) if i not in self.hidden]

    # layout ------------------------------------------------------------------------
    @staticmethod
    def _natural(b: _TabButton, text: str) -> int:
        """Width of the full label as drawn (icon included)."""
        return b.content_width(text)

    def needed_width(self, gap: int = GAP) -> int:
        shown = self.visible_indexes()
        return 2 * SIDE + sum(self._natural(self.buttons[i], self.labels[i]) for i in shown) + \
            gap * max(0, len(shown) - 1)

    def relayout(self) -> None:
        shown = self.visible_indexes()
        if not shown:
            return
        buttons = [self.buttons[i] for i in shown]
        labels = [self.labels[i] for i in shown]
        avail = self.row.width()
        natural = [self._natural(b, t) for b, t in zip(buttons, labels)]
        n_gaps = max(1, len(buttons) - 1)
        room = avail - 2 * SIDE - sum(natural)
        self.gap = max(MIN_GAP, min(GAP, room // n_gaps)) if room > 0 else MIN_GAP
        # still too wide at 12px gaps: give up side padding (down to 12), then elide the widest label
        n_real = len(buttons) - 1
        over = 2 * SIDE + sum(natural) + self.gap * n_real - avail
        self.side = SIDE - min(SIDE - MIN_SIDE, max(0, (over + 1) // 2))
        widths = list(natural)
        over = 2 * self.side + sum(widths) + self.gap * n_real - avail
        # then use a tab's short label ("Log" for "Execution Log"), the biggest saving first
        for k in sorted(range(len(labels)), key=lambda k: natural[k] - self._short_width(shown[k]), reverse=True):
            if over <= 0:
                break
            short = self.short.get(shown[k])
            if short:
                labels[k] = short
                saved = natural[k] - self._natural(buttons[k], short)
                natural[k] = widths[k] = natural[k] - saved
                over -= saved
        if labels != [self.labels[i] for i in shown]:  # short labels freed room: widen the gaps again
            room = avail - 2 * SIDE - sum(natural)
            self.gap = max(MIN_GAP, min(GAP, room // n_gaps)) if room > 0 else MIN_GAP
            over = 2 * SIDE + sum(natural) + self.gap * n_real - avail
            self.side = SIDE - min(SIDE - MIN_SIDE, max(0, (over + 1) // 2))
            over = 2 * self.side + sum(widths) + self.gap * n_real - avail
        if over > 0:  # elide the widest label just enough
            k = max(range(len(widths)), key=lambda i: widths[i])
            widths[k] = max(24, widths[k] - over)
        x = self.side
        for i, b, text, w, nat in zip(shown, buttons, labels, widths, natural):
            drawn = text
            if w < nat:
                icon_w = 0 if b.icon().isNull() else ICON + ICON_SPACING
                drawn = QFontMetricsF(b.font()).elidedText(text, Qt.TextElideMode.ElideRight, w - icon_w - 1)
            b.setText(drawn)
            b.setToolTip(self.labels[i] if drawn != self.labels[i] else "")
            b.setGeometry(QRect(x, 0, w, ROW_H - 2))
            x += w + self.gap
        self.row.update()

    def _short_width(self, i: int) -> int:
        short = self.short.get(i)
        return self._natural(self.buttons[i], short) if short else self._natural(self.buttons[i], self.labels[i])

    def label_rect(self, i: int) -> QRect:
        """The drawn label (text, plus the icon when there is one) in row coordinates: buttons are
        sized to exactly their content, so this is the button's geometry."""
        return self.buttons[i].geometry()

    def label_gaps(self) -> list[int]:
        rs = [self.label_rect(i) for i in self.visible_indexes()]
        return [b.left() - (a.right() + 1) for a, b in zip(rs, rs[1:])]
