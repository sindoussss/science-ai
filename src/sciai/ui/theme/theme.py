"""Theme tokens -> fonts + one QSS. All colors, radii and fonts live in tokens_*.json.
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
    "proposed": "●",
    "hypothesis": "◇",
    "inconclusive": "–",
}

_fonts_loaded = False


def load_bundled_fonts() -> None:
    """Register every bundled font file (Inter, Source Serif 4). Needs a QGuiApplication."""
    global _fonts_loaded
    if _fonts_loaded:
        return
    _fonts_loaded = True
    from PyQt6.QtCore import QByteArray

    folder = resources.files("sciai.ui.theme").joinpath("fonts")
    for entry in sorted(folder.iterdir(), key=lambda e: e.name):
        if entry.name.endswith((".ttf", ".otf")):
            QFontDatabase.addApplicationFontFromData(QByteArray(entry.read_bytes()))


def _dpi() -> float:
    from PyQt6.QtGui import QGuiApplication

    screen = QGuiApplication.primaryScreen()
    return screen.logicalDotsPerInchY() if screen is not None else 96.0


def set_px(font: QFont, px: float) -> QFont:
    """Pixel sizes may be fractional (12.5); QFont.setPixelSize only takes ints."""
    if float(px).is_integer():
        font.setPixelSize(int(px))
    else:
        font.setPointSizeF(px * 72.0 / _dpi())
    return font


def css_size(px: float) -> str:
    """QSS truncates fractional px, so fractional sizes are written in points."""
    return f"{int(px)}px" if float(px).is_integer() else f"{px * 72.0 / _dpi():.3f}pt"


@dataclass
class Theme:
    tokens: dict[str, Any]

    @classmethod
    def load(cls, name: str = "light") -> "Theme":
        text = resources.files("sciai.ui.theme").joinpath(f"tokens_{name}.json").read_text(encoding="utf-8")
        return cls(json.loads(text))

    # tokens ---------------------------------------------------------------
    def c(self, key: str) -> QColor:
        return QColor(self.tokens["color"][key])

    def hex(self, key: str) -> str:
        return self.tokens["color"][key]

    def radius(self, key: str) -> int:
        return int(self.tokens["radius"][key])

    def space(self, key: str) -> int:
        return int(self.tokens["space"][key])

    def px(self, key: str) -> float:
        return float(self.tokens["font"][key])

    def css(self, key: str) -> str:
        return css_size(self.px(key))

    def effect(self, key: str) -> bool:
        return bool(self.tokens.get("effects", {}).get(key, False))

    def family(self, key: str) -> str:
        available = set(QFontDatabase.families())
        for fam in self.tokens["font"][key]:
            if fam in available:
                return fam
        return self.tokens["font"][key][-1]

    def ui_font(self, size_key: str = "size_ui_px", bold: bool = False) -> QFont:
        f = set_px(QFont(self.family("ui")), self.px(size_key))
        f.setWeight(QFont.Weight.DemiBold if bold else QFont.Weight.Normal)
        return f

    def mono_font(self, size_key: str = "size_mono_px") -> QFont:
        f = set_px(QFont(self.family("mono")), self.px(size_key))
        f.setStyleHint(QFont.StyleHint.Monospace)
        return f

    def serif_font(self, size_key: str = "size_brand_px", bold: bool = False) -> QFont:
        f = set_px(QFont(self.family("serif")), self.px(size_key))
        f.setWeight(QFont.Weight.DemiBold if bold else QFont.Weight.Normal)
        return f

    def status_colors(self, status: str) -> tuple[QColor, QColor]:
        return self.c(f"status_{status}"), self.c(f"status_{status}_bg")

    def chip_css(self, status: str) -> str:
        fg, bg = self.hex(f"status_{status}"), self.hex(f"status_{status}_bg")
        border = f"1px dashed {fg}" if status == "invalidated" else f"1px solid {bg}"
        return (f"color:{fg}; background:{bg}; border:{border}; border-radius:{self.radius('chip')}px;"
                f"padding:1px 8px; font-weight:600; font-size:{self.css('size_node_small_px')};")

    # stylesheet ----------------------------------------------------------
    def qss(self) -> str:
        c = self.tokens["color"]
        r = self.tokens["radius"]
        ui, mono = self.family("ui"), self.family("mono")
        fs, small, mono_px = self.css("size_ui_px"), self.css("size_small_px"), self.css("size_mono_px")
        cap, title = self.css("size_caption_px"), self.css("size_title_px")
        return f"""
* {{ font-family: "{ui}"; font-size: {fs}; color: {c['text']}; outline: none; }}
QMainWindow, QWidget#appRoot {{ background: {c['app_bg']}; }}
QFrame#card {{ background: {c['panel']}; border: 1px solid {c['border']}; border-radius: {r['card']}px; }}
QFrame#card > QWidget, QWidget#cardBody {{ background: transparent; }}
QSplitter, QSplitter::handle {{ background: transparent; border: none; }}
QFrame#hairline {{ background: {c['border']}; max-height: 1px; min-height: 1px; border: none; }}
QLabel {{ background: transparent; }}
QLabel#brand {{ font-family: "{self.family('serif')}"; font-size: {self.css('size_brand_px')}; font-weight: 400; }}
QLabel#muted, QLabel#caption {{ color: {c['text_muted']}; }}
QLabel#caption {{ font-size: {cap}; font-weight: 600; padding: 0; }}
QLabel#nodeTitle, QLabel#paneTitle {{ font-size: {title}; font-weight: 600; }}
QLabel#secondary {{ color: {c['text_secondary']}; font-size: {small}; }}
QLabel#mono {{ font-family: "{mono}"; font-size: {mono_px}; }}

QPushButton, QToolButton {{ background: {c['panel']}; border: 1px solid {c['border']}; border-radius: {r['button']}px;
    padding: 5px 12px; color: {c['text']}; }}
QPushButton:hover, QToolButton:hover {{ background: {c['hover']}; }}
QPushButton:disabled, QToolButton:disabled {{ color: {c['text_faint']}; background: {c['panel']}; }}
QPushButton#primary {{ background: {c['accent']}; border-color: {c['accent']}; color: {c['accent_text']};
    font-weight: 600; }}
QPushButton#primary:hover {{ background: {c['accent_hover']}; border-color: {c['accent_hover']}; }}
QPushButton#primary:disabled {{ background: {c['border']}; border-color: {c['border']}; color: {c['text_faint']}; }}
QPushButton#outline {{ padding: 5px 10px; }}
QPushButton#nav {{ border: none; background: transparent; text-align: left; padding: 6px 8px; }}
QPushButton#nav:hover {{ background: {c['hover']}; }}
QPushButton#chip {{ border-radius: {r['chip']}px; padding: 2px 10px; background: {c['subtle']};
    font-family: "{mono}"; font-size: {mono_px}; }}
QPushButton#link, QToolButton#link {{ border: none; background: transparent; color: {c['text_muted']};
    padding: 2px 4px; text-align: left; font-size: {small}; }}
QPushButton#link:hover, QToolButton#link:hover {{ color: {c['text']}; }}
QToolButton#iconButton {{ border: none; background: transparent; padding: 2px 6px; color: {c['text_muted']}; }}
QToolButton#iconButton:hover {{ background: {c['hover']}; color: {c['text']}; }}
QToolButton::menu-indicator {{ image: none; width: 0; }}
QMenu {{ background: {c['panel']}; border: 1px solid {c['border']}; border-radius: {r['button']}px; padding: 4px; }}
QMenu::item {{ padding: 6px 18px; border-radius: 6px; }}
QMenu::item:selected {{ background: {c['hover']}; }}

QLineEdit, QPlainTextEdit, QTextEdit {{ background: {c['panel']}; border: 1px solid {c['border']};
    border-radius: {r['button']}px; padding: 6px 8px; selection-background-color: {c['accent_soft']};
    selection-color: {c['text']}; }}
QLineEdit:focus, QPlainTextEdit:focus, QTextEdit:focus {{ border-color: {c['accent']}; }}
QPlainTextEdit#code {{ font-family: "{mono}"; font-size: {mono_px}; background: {c['subtle']};
    border: none; border-radius: 0; padding: 8px 10px; }}

QListWidget {{ background: transparent; border: none; padding: 0; }}
QListWidget::item {{ padding: 6px 6px; border: none; border-radius: {r['button']}px; color: {c['text']}; }}
QListWidget::item:hover {{ background: {c['hover']}; }}
QListWidget::item:selected {{ background: {c['accent_soft']}; color: {c['text']}; }}

QTabWidget::pane {{ border: none; border-top: 1px solid {c['border']}; top: -1px; }}
QTabBar {{ background: transparent; }}
QTabBar::tab {{ background: transparent; border: none; padding: 7px 12px; margin-right: 2px;
    color: {c['text_muted']}; border-bottom: 2px solid transparent; }}
QTabBar::tab:hover {{ color: {c['text']}; }}
QTabBar::tab:selected {{ color: {c['text']}; border-bottom: 2px solid {c['accent']}; }}

QScrollArea {{ background: transparent; border: none; }}
QScrollArea > QWidget > QWidget {{ background: transparent; }}
QGraphicsView {{ background: {c['canvas']}; border: none; }}
QScrollBar:vertical {{ background: transparent; width: 8px; margin: 2px; }}
QScrollBar:horizontal {{ background: transparent; height: 8px; margin: 2px; }}
QScrollBar::handle {{ background: {c['border_strong']}; border-radius: 3px; min-height: 24px; min-width: 24px; }}
QScrollBar::add-line, QScrollBar::sub-line, QScrollBar::add-page, QScrollBar::sub-page {{
    width: 0; height: 0; background: transparent; }}

QFrame#evidenceCard {{ background: {c['panel']}; border: 1px solid {c['border']}; border-radius: {r['button']}px; }}
QFrame#bubbleUser {{ background: {c['bubble_user']}; border: none; border-radius: {r['bubble']}px; }}
QFrame#bubbleAssistant {{ background: {c['bubble_assistant']}; border: 1px solid {c['border']};
    border-radius: {r['bubble']}px; }}
QFrame#answerCard {{ background: {c['panel']}; border: 1px solid {c['border']}; border-radius: {r['card']}px; }}
QLabel#answerMath {{ font-size: {self.css('size_answer_px')}; font-weight: 600; }}
QFrame#notice {{ background: {c['status_proposed_bg']}; border: none; border-radius: {r['button']}px; }}
QFrame#pinEditor {{ background: {c['panel']}; border: 1px solid {c['border']}; border-radius: {r['card']}px; }}
QToolTip {{ background: {c['panel']}; color: {c['text']}; border: 1px solid {c['border']}; padding: 4px; }}
QDialog {{ background: {c['panel']}; }}
"""
