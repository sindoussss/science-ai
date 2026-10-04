from __future__ import annotations

from PyQt6.QtWidgets import QHBoxLayout, QLabel, QWidget

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
        lay.addStretch(1)
        hint = QLabel("Click to inspect · click again to pin · scroll to zoom")
        hint.setObjectName("muted")
        hint.setStyleSheet(f"font-size:{theme.px('size_small_px')}px;")
        lay.addWidget(hint)
