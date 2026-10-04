"""Slim legend row under the graph: soft status chips plus the interaction hint.

Painted so that a narrow workspace drops the hint first and then trailing chips, instead
of squeezing or clipping text (the full legend is always in the tooltip)."""
from __future__ import annotations

from PyQt6.QtCore import QRectF, QSize, Qt
from PyQt6.QtGui import QFontMetricsF, QPainter, QPaintEvent
from PyQt6.QtWidgets import QSizePolicy, QWidget

from sciai.ui.chips import chip_label, chip_size, paint_chip
from sciai.ui.theme.theme import Theme

HINT = "Click to inspect · click again to pin"
STATUSES = ("verified", "proposed", "failed", "invalidated", "hypothesis")
H = 36
PAD_X = 16
GAP = 6


class Legend(QWidget):
    def __init__(self, theme: Theme, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("legendRow")
        self.theme = theme
        self.setFixedHeight(H)
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        self.setToolTip("Status: " + ", ".join(STATUSES) + "\n" + HINT)
        self.hint_font = theme.ui_font("size_legend_px")

    def sizeHint(self) -> QSize:  # noqa: N802
        return QSize(self._full_width(), H)

    def _chip_w(self, status: str) -> float:
        return chip_size(self.theme, chip_label(status))[0]

    def _full_width(self) -> int:
        chips = sum(self._chip_w(s) for s in STATUSES) + GAP * (len(STATUSES) - 1)
        return int(2 * PAD_X + chips + 16 + QFontMetricsF(self.hint_font).horizontalAdvance(HINT))

    def shown(self) -> tuple[list[str], bool]:
        """(chips that fit, whether the hint fits) at the current width."""
        room = self.width() - 2 * PAD_X
        chips: list[str] = []
        used = 0.0
        for s in STATUSES:
            w = self._chip_w(s) + (GAP if chips else 0)
            if used + w > room:
                break
            chips.append(s)
            used += w
        hint_w = 16 + QFontMetricsF(self.hint_font).horizontalAdvance(HINT)
        return chips, len(chips) == len(STATUSES) and used + hint_w <= room

    def paintEvent(self, event: QPaintEvent) -> None:  # noqa: N802
        t = self.theme
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), t.c("panel"))
        p.fillRect(0, 0, self.width(), 1, t.c("border"))
        chips, hint = self.shown()
        x = float(PAD_X)
        for s in chips:
            x += paint_chip(p, t, x, H / 2, s, chip_label(s)).width() + GAP
        if hint:
            p.setFont(self.hint_font)
            p.setPen(t.c("text_secondary"))
            p.drawText(QRectF(x, 0, self.width() - PAD_X - x, H),
                       Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter, HINT)
        p.end()
