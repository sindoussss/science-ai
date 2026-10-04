"""Theme tokens -> fonts + one QSS. All colors, radii and fonts live in tokens_*.json.
Dark mode is a second token file: swap, not rewrite."""
from __future__ import annotations

import json
from dataclasses import dataclass
from importlib import resources
from typing import Any

from PyQt6.QtGui import QColor, QFont, QFontDatabase

# Status is shown as a plain word in a neutral outlined chip: no symbols, no colored fills or borders.
STATUSES = ("verified", "failed", "invalidated", "proposed", "hypothesis", "inconclusive")

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
    """Pixel sizes may be fractional (12.5); QFont.setPixelSize only takes ints.

    Where the font engine renders a fractional size at the next whole pixel (hinted FreeType does),
    the font is also narrowed by the same ratio so text keeps the width of the requested size."""
    if float(px).is_integer():
        font.setPixelSize(int(px))
    else:
        font.setPointSizeF(px * 72.0 / _dpi())
        font.setStretch(fractional_stretch(font, px))
    return font


def fractional_stretch(font: QFont, px: float) -> int:
    """QFont stretch (100 = normal) that makes ``font``, sized to ``px``, as wide as ``px`` text."""
    if float(px).is_integer():
        return 100
    from PyQt6.QtGui import QFontInfo

    probe = QFont(font)
    probe.setStretch(100)
    shown = QFontInfo(probe).pointSizeF() * _dpi() / 72.0
    return 100 if shown <= 0 or abs(shown - px) < 0.05 else round(100 * px / shown)


def keep_fractional_width(widget, px: float) -> None:  # noqa: ANN001 - any QWidget
    """For a widget whose fractional font size comes from the QSS: QSS has no font-stretch, but a
    stretch set on the widget's own font survives the stylesheet (which only sets the size)."""
    f = widget.font()
    f.setStretch(fractional_stretch(set_px(QFont(f), px), px))
    widget.setFont(f)


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

    def chip_font(self) -> QFont:
        f = set_px(QFont(self.family("ui")), self.px("size_chip_px"))
        f.setWeight(QFont.Weight.Medium)
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

    def chip_css(self, status: str, size_key: str = "size_chip_px") -> str:
        """Neutral status chip: panel fill, gray text, 1px gray outline (dashed for invalidated),
        6px radius, 2px/8px padding."""
        fg = self.hex(f"status_{status}")
        border = (f"1px dashed {self.hex('border_strong')}" if status == "invalidated"
                  else f"1px solid {self.hex('border')}")
        return (f"color:{fg}; background:{self.hex('panel')}; border:{border}; border-radius:{self.radius('chip')}px;"
                f"padding:2px 8px; font-weight:500; font-size:{self.css(size_key)};")

    # stylesheet ----------------------------------------------------------
    def qss(self) -> str:
        c = self.tokens["color"]
        r = self.tokens["radius"]
        ui, mono, serif = self.family("ui"), self.family("mono"), self.family("serif")
        fs, small, mono_px = self.css("size_ui_px"), self.css("size_small_px"), self.css("size_mono_px")
        title, chat = self.css("size_title_px"), self.css("size_chat_px")
        return f"""
* {{ font-family: "{ui}"; font-size: {fs}; color: {c['text']}; outline: none; }}
QMainWindow, QWidget#appRoot {{ background: {c['app_bg']}; }}
QFrame#card {{ background: {c['panel']}; border: 1px solid {c['border']}; border-radius: {r['card']}px; }}
QFrame#card > QWidget, QWidget#cardBody, QWidget#flat {{ background: transparent; }}
QSplitter, QSplitter::handle {{ background: transparent; border: none; }}
QFrame#hairline {{ background: {c['border']}; max-height: 1px; min-height: 1px; border: none; }}
QLabel {{ background: transparent; }}
QLabel#brand {{ font-family: "{serif}"; font-size: {self.css('size_brand_px')}; font-weight: 400; }}
QLabel#muted, QLabel#secondary {{ color: {c['text_secondary']}; font-size: {small}; }}
QLabel#caption {{ color: {c['text_secondary']}; font-size: {small}; padding: 0; }}
QLabel#sessionTitle {{ color: {c['text_title']}; font-size: {self.css('size_session_title_px')}; font-weight: 500; }}
QLabel#paneTitle {{ color: {c['text_title']}; font-size: {fs}; }}
QLabel#nodeTitle {{ font-size: {title}; font-weight: 600; }}
QLabel#mono {{ font-family: "{mono}"; font-size: {mono_px}; }}
QLabel#cellTag {{ font-family: "{mono}"; font-size: {mono_px}; color: {c['text_title']}; }}
QLabel#toolChip {{ font-family: "{mono}"; font-size: {self.css('size_node_small_px')}; color: {c['text_title']};
    background: {c['panel']}; border: 1px solid {c['border']}; border-radius: 9px; padding: 1px 8px; }}
QLabel#chatText {{ font-size: {chat}; }}

QPushButton, QToolButton {{ background: {c['panel']}; border: 1px solid {c['border']}; border-radius: {r['button']}px;
    padding: 5px 12px; color: {c['text']}; }}
QPushButton:hover, QToolButton:hover {{ background: {c['hover']}; }}
QPushButton:disabled, QToolButton:disabled {{ color: {c['text_faint']}; }}
QPushButton#primary {{ background: {c['accent']}; border-color: {c['accent']}; color: {c['accent_text']};
    font-weight: 500; padding: 5px 12px; }}
QPushButton#primary:hover {{ background: {c['accent_hover']}; border-color: {c['accent_hover']}; }}
QPushButton#primary:disabled {{ background: {c['border']}; border-color: {c['border']}; color: {c['text_faint']}; }}
QPushButton#outline {{ padding: 4px 10px; }}
QPushButton#nav, QToolButton#nav {{ border: none; background: transparent; text-align: left; padding: 8px 8px;
    border-radius: {r['button']}px; }}
QPushButton#nav:hover, QToolButton#nav:hover {{ background: {c['hover']}; }}
QPushButton#chip {{ border-radius: 12px; padding: 2px 10px; background: {c['panel']}; border: 1px solid {c['border']};
    font-family: "{mono}"; font-size: {mono_px}; }}
QPushButton#link, QToolButton#link {{ border: none; background: transparent; color: {c['text_secondary']};
    padding: 2px 4px; text-align: left; font-size: {small}; }}
QPushButton#link:hover, QToolButton#link:hover {{ color: {c['text']}; }}
QToolButton#iconButton {{ border: none; background: transparent; padding: 4px; border-radius: 6px;
    color: {c['text_secondary']}; }}
QToolButton#iconButton:hover {{ background: {c['hover']}; color: {c['text']}; }}
QToolButton::menu-indicator {{ image: none; width: 0; }}
QToolButton#pillTab {{ border: none; background: transparent; border-radius: {r['pill']}px; padding: 5px 12px;
    color: {c['text_title']}; }}
QToolButton#pillTab:checked {{ background: {c['subtle']}; color: {c['text']}; }}
QToolButton#pillTab:hover {{ background: {c['hover']}; }}
QToolButton#liveButton {{ border: 1px solid {c['border']}; background: {c['panel']}; border-radius: 8px; padding: 3px 8px;
    color: {c['text']}; }}
QLabel#badge {{ background: {c['badge_bg']}; border: none; border-radius: 4px;
    padding: 0px 5px; color: {c['text_title']}; font-size: {small}; }}
QLabel#projectName {{ font-size: {title}; font-weight: 500; }}
QToolButton#sectionToggle {{ border: none; background: transparent; padding: 2px 0px; color: {c['text_secondary']};
    font-size: {small}; }}
QToolButton#sectionToggle:hover {{ color: {c['text']}; }}
QLabel#versionTag {{ background: {c['subtle']}; border-radius: 6px; padding: 1px 6px; font-size: {small};
    color: {c['text_title']}; }}
QLabel#outputText {{ font-family: "{mono}"; font-size: {mono_px}; color: {c['text']}; }}
QLabel#monoSecondary {{ font-family: "{mono}"; font-size: {mono_px}; color: {c['text_secondary']}; }}
QLabel#codeChip {{ font-family: "{mono}"; font-size: {mono_px}; color: {c['chip_code_text']};
    background: {c['chip_code_bg']}; border-radius: 4px; padding: 1px 4px; }}
QLabel#ruleChip {{ background: {c['panel']}; color: {c['text_title']}; border: 1px solid {c['border']};
    border-radius: {r['chip']}px;
    padding: 2px 8px; font-weight: 500; font-size: {self.css('size_chip_px')}; }}
QPlainTextEdit#pinText {{ border: none; background: transparent; padding: 0px; }}
QToolButton#liveButton {{ padding-right: 20px; }}
QToolButton#overlayButton {{ background: {c['panel']}; border: 1px solid {c['border']}; border-radius: {r['button']}px;
    padding: 3px 10px; font-size: {small}; }}
QPushButton#send {{ background: {c['send']}; border: none; border-radius: 10px; padding: 0; }}
QPushButton#send:hover {{ background: {c['send_hover']}; }}
QPushButton#send:disabled {{ background: {c['border_strong']}; }}
QPushButton#darkPill {{ background: {c['pill_dark']}; color: {c['pill_dark_text']}; border: none; border-radius: 8px;
    padding: 4px 12px; font-weight: 600; }}
QToolButton#statusStrip {{ border: none; background: transparent; text-align: left; padding: 0px 16px;
    color: {c['status_strip_text']}; font-size: {small}; }}
QToolButton#composerIcon {{ border: none; background: transparent; padding: 6px; border-radius: 8px; }}
QToolButton#composerIcon:hover {{ background: {c['hover']}; }}
QToolButton#subTab {{ border: none; background: transparent; padding: 0px; color: {c['text']};
    font-size: {self.css('size_tab_px')}; }}
QToolButton#subTab:checked {{ color: {c['text']}; }}
QToolButton#subTab:hover {{ color: {c['text']}; }}
QToolButton#statusStrip:hover {{ color: {c['text']}; }}
QMenu {{ background: {c['panel']}; border: 1px solid {c['border']}; border-radius: {r['button']}px; padding: 4px; }}
QMenu::item {{ padding: 6px 18px; border-radius: 6px; }}
QMenu::item:selected {{ background: {c['hover']}; }}

QLineEdit, QPlainTextEdit, QTextEdit {{ background: {c['panel']}; border: 1px solid {c['border']};
    border-radius: {r['button']}px; padding: 6px 8px; selection-background-color: {c['accent_soft']};
    selection-color: {c['text']}; }}
QLineEdit:focus, QPlainTextEdit:focus {{ border-color: {c['border_strong']}; }}
QPlainTextEdit#composerInput {{ border: none; background: transparent; padding: 0px; font-size: {chat}; }}
QPlainTextEdit#code, QPlainTextEdit#log {{ font-family: "{mono}"; font-size: {mono_px}; background: {c['code_bg']};
    border: none; border-radius: 0; padding: 6px 8px; color: {c['code_plain']}; }}

QListWidget {{ background: transparent; border: none; padding: 0; }}
QListWidget::item {{ padding: 6px 6px; border: none; border-radius: {r['button']}px; color: {c['text']}; }}
QListWidget::item:hover {{ background: {c['hover']}; }}
QListWidget::item:selected {{ background: {c['bubble_user']}; color: {c['text']}; }}

QTabWidget::pane {{ border: none; border-top: 1px solid {c['border']}; top: -1px; }}
QTabBar {{ background: transparent; }}
QTabBar::tab {{ background: transparent; border: none; padding: 7px 5px; margin-right: 0px;
    color: {c['text']}; border-bottom: 2px solid transparent; }}
QTabBar::tab:hover {{ color: {c['text']}; }}
QTabBar::tab:selected {{ color: {c['text']}; border-bottom: 2px solid {c['text']}; }}

QScrollArea {{ background: transparent; border: none; }}
QScrollArea > QWidget > QWidget {{ background: transparent; }}
QGraphicsView {{ background: {c['canvas']}; border: none; }}
QScrollBar {{ background: transparent; border: none; margin: 0; }}
QScrollBar:vertical {{ width: 8px; }}
QScrollBar:horizontal {{ height: 8px; }}
QScrollBar::handle {{ background: {c['scroll_handle']}; border-radius: 4px; min-height: 24px; min-width: 24px; }}
QScrollBar::add-line, QScrollBar::sub-line {{ width: 0; height: 0; border: none; background: transparent; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}

QFrame#evidenceCard {{ background: {c['panel']}; border: 1px solid {c['border']}; border-radius: {r['card']}px; }}
QFrame#bubbleUser {{ background: {c['bubble_user']}; border: none; border-radius: {r['bubble']}px; }}
QFrame#answerCard {{ background: {c['panel']}; border: 1px solid {c['border']}; border-radius: {r['card']}px; }}
QLabel#answerMath {{ font-size: {self.css('size_answer_px')}; font-weight: 600; }}
QFrame#notice {{ background: {c['subtle']}; border: none; border-radius: {r['button']}px; }}
QWidget#composerCard {{ background: transparent; }}
QFrame#popover, QFrame#pinEditor {{ background: {c['panel']}; border: 1px solid {c['border']}; border-radius: {r['card']}px; }}
QFrame#legendRow {{ background: {c['panel']}; border: none; border-top: 1px solid {c['border']}; }}
QFrame#outputBox {{ background: {c['panel']}; border: 1px solid {c['border']}; border-radius: 8px; }}
QToolTip {{ background: {c['panel']}; color: {c['text']}; border: 1px solid {c['border']}; padding: 4px; }}
QLabel#dataName {{ font-size: {title}; font-weight: 600; }}
QTableWidget#dataPreview {{ background: {c['panel']}; border: 1px solid {c['border']}; border-radius: 8px;
    gridline-color: {c['border']}; font-family: "{mono}"; font-size: {mono_px}; color: {c['table_text']};
    selection-background-color: {c['accent_soft']}; selection-color: {c['text']}; }}
QTableWidget#dataPreview QHeaderView::section {{ background: {c['table_header_bg']}; color: {c['text_title']};
    border: none; border-right: 1px solid {c['border']}; border-bottom: 1px solid {c['border']};
    padding: 4px 8px; font-family: "{ui}"; font-size: {small}; }}
QTableWidget#dataPreview QTableCornerButton::section {{ background: {c['table_header_bg']}; border: none;
    border-bottom: 1px solid {c['border']}; border-right: 1px solid {c['border']}; }}
QDialog {{ background: {c['panel']}; }}
"""
