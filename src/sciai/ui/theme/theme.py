"""Theme tokens -> QSS. All colors, radii and fonts live in tokens_*.json.
Dark mode is a second token file: swap, not rewrite."""
from __future__ import annotations

import json
from dataclasses import dataclass
from importlib import resources
from typing import Any

from PyQt6.QtGui import QColor, QFont, QFontDatabase

STATUS_ICON = {
    "verified": "✓",
    "failed": "✕",
    "invalidated": "⊘",
    "proposed": "?",
    "hypothesis": "◇",
}


@dataclass
class Theme:
    tokens: dict[str, Any]

    @classmethod
    def load(cls, name: str = "light") -> "Theme":
        text = resources.files("sciai.ui.theme").joinpath(f"tokens_{name}.json").read_text(encoding="utf-8")
        return cls(json.loads(text))

    def c(self, key: str) -> QColor:
        return QColor(self.tokens["color"][key])

    def hex(self, key: str) -> str:
        return self.tokens["color"][key]

    def radius(self, key: str) -> int:
        return int(self.tokens["radius"][key])

    def _family(self, key: str) -> str:
        available = set(QFontDatabase.families())
        for fam in self.tokens["font"][key]:
            if fam in available:
                return fam
        return self.tokens["font"][key][-1]

    def ui_font(self, size_key: str = "size_ui", bold: bool = False) -> QFont:
        f = QFont(self._family("ui"), int(self.tokens["font"][size_key]))
        f.setBold(bold)
        return f

    def mono_font(self) -> QFont:
        f = QFont(self._family("mono"), int(self.tokens["font"]["size_mono"]))
        f.setStyleHint(QFont.StyleHint.Monospace)
        return f

    def status_colors(self, status: str) -> tuple[QColor, QColor]:
        return self.c(f"status_{status}"), self.c(f"status_{status}_bg")

    def qss(self) -> str:
        c = self.tokens["color"]
        r = self.tokens["radius"]
        ui = self._family("ui")
        mono = self._family("mono")
        return f"""
* {{ font-family: "{ui}"; color: {c['text']}; }}
QMainWindow, QWidget#canvasPane {{ background: {c['canvas']}; }}
QWidget#sidebar, QWidget#inspector, QWidget#chatPane {{ background: {c['panel']}; }}
QFrame#hairline {{ background: {c['border']}; max-height: 1px; min-height: 1px; border: none; }}
QSplitter::handle {{ background: {c['border']}; width: 1px; }}
QLabel#muted, QLabel#sectionTitle {{ color: {c['text_muted']}; }}
QLabel#sectionTitle {{ font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; }}
QLabel#nodeTitle {{ font-size: 13pt; font-weight: 600; }}
QPushButton {{ background: {c['panel']}; border: 1px solid {c['border_strong']}; border-radius: {r['button']}px;
              padding: 4px 10px; }}
QPushButton:hover {{ border-color: {c['accent']}; }}
QPushButton:disabled {{ color: {c['text_faint']}; border-color: {c['border']}; }}
QPushButton#primary {{ background: {c['accent']}; color: {c['accent_text']}; border-color: {c['accent']}; }}
QPushButton#chip {{ border-radius: {r['chip']}px; padding: 2px 8px; font-family: "{mono}"; }}
QLineEdit, QPlainTextEdit, QTextEdit, QComboBox {{ background: {c['panel']}; border: 1px solid {c['border']};
              border-radius: {r['button']}px; padding: 4px 6px; selection-background-color: {c['accent_soft']};
              selection-color: {c['text']}; }}
QLineEdit:focus, QPlainTextEdit:focus, QTextEdit:focus {{ border-color: {c['accent']}; }}
QPlainTextEdit#mono, QTextEdit#mono {{ font-family: "{mono}"; }}
QListWidget {{ background: {c['panel']}; border: none; outline: none; }}
QListWidget::item {{ padding: 5px 6px; border-radius: {r['button']}px; }}
QListWidget::item:selected {{ background: {c['accent_soft']}; color: {c['text']}; }}
QTabWidget::pane {{ border: none; border-top: 1px solid {c['border']}; }}
QTabBar::tab {{ background: transparent; padding: 6px 10px; color: {c['text_muted']}; border: none; }}
QTabBar::tab:selected {{ color: {c['text']}; border-bottom: 2px solid {c['accent']}; }}
QGraphicsView {{ background: {c['canvas']}; border: none; }}
QScrollBar:vertical {{ background: transparent; width: 8px; }}
QScrollBar::handle:vertical {{ background: {c['border_strong']}; border-radius: 4px; min-height: 24px; }}
QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; width: 0; }}
QToolTip {{ background: {c['panel']}; color: {c['text']}; border: 1px solid {c['border']}; }}
"""
