"""Status legend floating over the bottom of the canvas (so the canvas keeps its full height)."""
from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QFrame, QHBoxLayout, QLabel, QWidget

from sciai.ui.shell import ElidedLabel
from sciai.ui.theme.theme import STATUS_ICON, Theme

HINT = "Click to inspect · click again to pin"


class Legend(QFrame):
    def __init__(self, theme: Theme, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("canvasLegend")
        lay = QHBoxLayout(self)
        lay.setContentsMargins(10, 6, 10, 6)
        lay.setSpacing(6)
        self.chips: list[QLabel] = []
        for status in ("verified", "proposed", "failed", "invalidated", "hypothesis"):
            chip = QLabel(f"{STATUS_ICON[status]} {status}")
            chip.setStyleSheet(theme.chip_css(status, "size_legend_px"))
            lay.addWidget(chip)
            self.chips.append(chip)
        lay.addSpacing(6)
        self.hint = ElidedLabel(HINT)
        self.hint.setObjectName("muted")
        self.hint.setStyleSheet(f"font-size:{theme.css('size_legend_px')};")
        self.hint.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        lay.addWidget(self.hint, 1)
