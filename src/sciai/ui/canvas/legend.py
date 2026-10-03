from __future__ import annotations

from PyQt6.QtWidgets import QHBoxLayout, QLabel, QWidget

from sciai.ui.theme.theme import STATUS_ICON, Theme


class Legend(QWidget):
    def __init__(self, theme: Theme) -> None:
        super().__init__()
        lay = QHBoxLayout(self)
        lay.setContentsMargins(12, 4, 12, 4)
        lay.setSpacing(14)
        for status in ("verified", "proposed", "failed", "invalidated", "hypothesis"):
            fg, bg = theme.hex(f"status_{status}"), theme.hex(f"status_{status}_bg")
            border = "dashed" if status == "invalidated" else "solid"
            chip = QLabel(f"{STATUS_ICON[status]} {status}")
            chip.setStyleSheet(f"color:{fg}; background:{bg}; border:1px {border} {fg}; border-radius:9px;"
                               "padding:1px 8px; font-weight:600;")
            lay.addWidget(chip)
        hint = QLabel("Click a node to inspect it; click it again to pin a note.")
        hint.setObjectName("muted")
        lay.addStretch(1)
        lay.addWidget(hint)
