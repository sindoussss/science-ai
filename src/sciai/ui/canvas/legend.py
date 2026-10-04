from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QHBoxLayout, QLabel, QWidget

from sciai.ui.shell import ElidedLabel
from sciai.ui.theme.theme import STATUS_ICON, Theme


class Legend(QWidget):
    def __init__(self, theme: Theme) -> None:
        super().__init__()
        lay = QHBoxLayout(self)
        lay.setContentsMargins(14, 8, 14, 8)
        lay.setSpacing(8)
        for status in ("verified", "proposed", "failed", "invalidated", "hypothesis"):
            chip = QLabel(f"{STATUS_ICON[status]} {status}")
            chip.setStyleSheet(theme.chip_css(status))
            lay.addWidget(chip)
        lay.addSpacing(8)
        hint = ElidedLabel("Click to inspect · click again to pin a note · scroll to zoom · double-click to refit")
        hint.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        hint.setObjectName("muted")
        hint.setStyleSheet(f"font-size:{theme.css('size_small_px')};")
        lay.addWidget(hint, 1)
