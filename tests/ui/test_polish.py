"""Measured checks for the strict polish pass (one test per item of Yeri's list), on real widgets
after the injected-fault run at 1200x720. Geometry is read from the laid-out widgets and colors
from grabbed pixels where a token alone wouldn't prove what is drawn."""
from __future__ import annotations

import colorsys
import json
from importlib import resources

import pytest

pytest.importorskip("PyQt6")

from PyQt6.QtCore import QEvent, QPoint, QPointF, Qt  # noqa: E402
from PyQt6.QtGui import QColor, QFont, QFontInfo, QFontMetricsF, QWheelEvent  # noqa: E402
from PyQt6.QtWidgets import QAbstractScrollArea, QApplication, QGraphicsView, QStyle, QStyleOptionSlider  # noqa: E402

from tests.ui.metrics import NODE_TABS, pump, setup_app  # noqa: E402

LONG_TITLE = ("Second derivative test for f(x) = x^2 sin(x) on the interval [0, pi], with every step "
              "checked independently and finally.")  # 120 chars
assert len(LONG_TITLE) == 120


def _x(w, ref) -> int:  # noqa: ANN001
    return w.mapTo(ref, QPoint(0, 0)).x()


def _y(w, ref) -> int:  # noqa: ANN001
    return w.mapTo(ref, QPoint(0, 0)).y()


def _close(a: QColor, hex_: str, tol: int = 6) -> bool:
    b = QColor(hex_)
    return max(abs(a.red() - b.red()), abs(a.green() - b.green()), abs(a.blue() - b.blue())) <= tol


def _tokens() -> dict:
    return json.loads(resources.files("sciai.ui.theme").joinpath("tokens_light.json").read_text())


# 1. Session title ---------------------------------------------------------------------------------
def test_long_title_is_one_elided_line_and_the_thread_starts_16px_below(fault_window):
    win = fault_window((1200, 720))
    app = setup_app()
    chat = win.chat
    chat.set_title(LONG_TITLE)
    bar = chat.scroll.verticalScrollBar()
    bar.setValue(0)
    pump(app, lambda: True)
    lbl = chat.title
    fm = QFontMetricsF(lbl.font())
    assert QFontInfo(lbl.font()).pixelSize() == 15
    assert lbl.font().weight() == QFont.Weight.Medium
    assert not lbl.wordWrap()
    assert lbl.text().endswith("…") and lbl.text() != LONG_TITLE
    assert fm.horizontalAdvance(lbl.text()) <= lbl.contentsRect().width()
    assert lbl.height() < 2 * fm.lineSpacing()  # one line, never two
    title_bottom = _y(lbl, chat) + lbl.height()
    assert _y(chat.scroll, chat) >= title_bottom + 16  # the thread's viewport starts below the gap
    first = next(chat.messages.itemAt(i).widget() for i in range(chat.messages.count())
                 if chat.messages.itemAt(i).widget() is not None)
    assert _y(first, chat) >= title_bottom + 16  # and so does the first message


# 2. Overlay scrollbars ----------------------------------------------------------------------------
def test_every_scroll_area_uses_thin_overlay_bars_shown_on_hover_or_scroll(fault_window):
    from sciai.ui.scrollbars import LINGER_MS, OverlayBar

    win = fault_window((1200, 720))
    app = setup_app()
    for name in ("code", "review"):  # visit the node tabs so their scroll areas are shown too
        win.workspace.show_node_tab()
        win.node_panel.tabs.setCurrentIndex(0 if name == "code" else 4)
        pump(app, lambda: True)
    areas = [a for a in win.findChildren(QAbstractScrollArea) if not isinstance(a, QGraphicsView)]
    assert areas
    for a in areas:
        assert a.property("overlayBars"), type(a).__name__
        assert a.verticalScrollBarPolicy() == Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        assert a.horizontalScrollBarPolicy() == Qt.ScrollBarPolicy.ScrollBarAlwaysOff
    win.workspace.show_graph()
    area = win.chat.scroll
    bar = next(b for b in area.findChildren(OverlayBar) if b.orientation() == Qt.Orientation.Vertical)
    pump(app, lambda: True)
    assert bar.scrollable()
    assert not bar.isVisible()  # hidden at rest
    QApplication.sendEvent(area, QEvent(QEvent.Type.Enter))
    pump(app, lambda: True)
    assert bar.isVisible() and bar.width() == 8
    assert bar.geometry().right() <= area.width() - 1  # floats over the area, takes no layout room
    # handle #D8D6CF, no arrow buttons
    opt = QStyleOptionSlider()
    bar.initStyleOption(opt)
    st = bar.style()
    handle = st.subControlRect(QStyle.ComplexControl.CC_ScrollBar, opt, QStyle.SubControl.SC_ScrollBarSlider, bar)
    for sc in (QStyle.SubControl.SC_ScrollBarAddLine, QStyle.SubControl.SC_ScrollBarSubLine):
        r = st.subControlRect(QStyle.ComplexControl.CC_ScrollBar, opt, sc, bar)
        assert r.height() <= 0 or r.width() <= 0
    img = bar.grab().toImage()
    assert _close(img.pixelColor(handle.center()), "#D8D6CF", tol=8)
    QApplication.sendEvent(area, QEvent(QEvent.Type.Leave))
    pump(app, lambda: True)
    assert not bar.isVisible()
    wheel = QWheelEvent(QPointF(20, 20), QPointF(20, 20), QPoint(0, 0), QPoint(0, -120), Qt.MouseButton.NoButton,
                        Qt.KeyboardModifier.NoModifier, Qt.ScrollPhase.NoScrollPhase, False)
    QApplication.sendEvent(area.viewport(), wheel)
    pump(app, lambda: True)
    assert bar.isVisible()  # while scrolling
    pump(app, lambda: not bar.isVisible(), timeout=LINGER_MS / 1000 + 2)  # then fades out


# 3. Proportions -----------------------------------------------------------------------------------
@pytest.mark.parametrize("width", [1200, 1280, 1440, 1680, 1920])
def test_proportions_and_tab_row_at_any_width(fault_window, width):
    win = fault_window((1200, 720))
    app = setup_app()
    win.resize(width, 900)
    win.workspace.show_node_tab()
    pump(app, lambda: win.width() == width)
    side, chat, ws = win.region_widths()
    assert side + chat + ws == width
    assert side == max(200, min(240, round(0.17 * width)))
    assert abs(ws - max(440, min(640, round(0.40 * width)))) <= 1
    assert chat >= 440 and win.chat.width() >= 440
    tabs = win.node_panel.tabs
    assert [b.text() for b in tabs.buttons] == NODE_TABS  # all five in full
    assert tabs.label_gaps() == [18, 18, 18, 18]
    assert tabs.side == 16
    assert tabs.label_rect(0).left() == 16 and tabs.label_rect(4).right() < tabs.row.width() - 16
    assert QFontInfo(tabs.buttons[0].font()).pixelSize() in (13, 14)  # 13.5px (engine rounds the raster)
    for b in tabs.buttons:
        assert b.width() >= b.sizeHint().width()  # never clipped


# 4. One content column ----------------------------------------------------------------------------
def test_text_table_answer_and_composer_share_one_column(fault_window):
    from sciai.ui.chat.inline import InlineText, PlotCard, TableCard
    from sciai.ui.chat_bar import AnswerCard

    win = fault_window((1200, 720))
    chat = win.chat
    pad = round(0.045 * chat.width())
    assert chat.pad() == pad
    blocks = [w for cls in (TableCard, AnswerCard, PlotCard, InlineText) for w in chat.findChildren(cls)
              if w.isVisible()]
    assert any(isinstance(w, TableCard) for w in blocks) and any(isinstance(w, AnswerCard) for w in blocks)
    edges = {(type(w).__name__, _x(w, chat), w.width()) for w in blocks}
    left, width = pad, chat.width() - 2 * pad
    for name, x, w in edges:
        assert x == left, name
        if name != "InlineText":  # cards fill the column; text wraps inside it
            assert w == width, name
        else:
            assert w <= width
    c = chat.composer
    assert (_x(c, chat), c.width()) == (left, width)
    assert _x(chat.title, chat) + chat.title.contentsMargins().left() == left


def test_column_holds_at_the_narrowest_chat(fault_window):
    """Workspace dragged to its widest at 1200: the chat is at its 440px floor and every card still
    fits the column (columns elide, the answer card's button wraps) instead of running past it."""
    from sciai.ui.chat.inline import TableCard
    from sciai.ui.chat_bar import AnswerCard

    win = fault_window((1200, 720))
    app = setup_app()
    sp = win.main_split
    sp.moveSplitter(1, 1)
    pump(app, lambda: True)
    assert win.chat.width() == 440  # the chat pane's floor (its region, gutters included, is wider)
    chat = win.chat
    left, width = chat.pad(), chat.width() - 2 * chat.pad()
    for card in [*chat.findChildren(TableCard), *chat.findChildren(AnswerCard), chat.composer]:
        assert (_x(card, chat), card.width()) == (left, width), type(card).__name__
    table = chat.findChildren(TableCard)[0]
    assert table.elided() == []


# 5. Chips -----------------------------------------------------------------------------------------
# Yeri, 2026-10-04: no emoji and no color in borders. Status is a plain word in a neutral outlined chip.
def _sat(hex_: str) -> float:
    """Chroma (max - min channel, 0..1): HSV saturation calls a warm near-black "colorful"."""
    c = QColor(hex_)
    return (max(c.red(), c.green(), c.blue()) - min(c.red(), c.green(), c.blue())) / 255


def test_chips_are_neutral_outlines_with_plain_words():
    from PyQt6.QtGui import QImage, QPainter

    from sciai.ui.chips import chip_label, chip_size, paint_chip
    from sciai.ui.theme.theme import STATUSES, Theme

    setup_app()
    for name in ("light", "dark"):
        t = Theme.load(name)
        for status in STATUSES:
            assert _sat(t.hex(f"status_{status}")) < 0.1, (name, status)
            assert _sat(t.hex(f"status_{status}_bg")) < 0.1, (name, status)
    t = Theme.load("light")
    assert t.radius("chip") == 6
    assert QFontInfo(t.chip_font()).pixelSize() == 12
    for status in STATUSES:
        assert chip_label(status) == status  # the word only, no symbol
        css = t.chip_css(status)
        assert "border-radius:6px" in css and "padding:2px 8px" in css and "font-size:12px" in css
        assert f"background:{t.hex('panel')}" in css
        if status == "invalidated":
            assert f"1px dashed {t.hex('border_strong')}" in css
        else:
            assert f"1px solid {t.hex('border')}" in css
    w, h = chip_size(t, "verified")
    fm = QFontMetricsF(t.chip_font())
    assert (w, h) == (fm.horizontalAdvance("verified") + 16, fm.height() + 4)
    # what is painted: panel fill inside a gray outline
    img = QImage(200, 40, QImage.Format.Format_ARGB32)
    img.fill(QColor("#FFFFFF"))
    p = QPainter(img)
    r = paint_chip(p, t, 10, 20, "failed", "failed")
    p.end()
    assert _close(img.pixelColor(int(r.left()) + 3, int(r.center().y())), t.hex("panel"), tol=2)
    edge = img.pixelColor(int(r.center().x()), int(r.top()))
    assert _sat(edge.name()) < 0.1 and edge.name().upper() != "#FFFFFF"
    stakes = t.qss().split("#ruleChip", 1)[1].split("}", 1)[0]
    assert t.hex("panel").lower() in stakes.lower() and t.hex("border").lower() in stakes.lower()


def test_no_colored_borders_in_the_stylesheet():
    """Every border, outline and focus ring is gray. The one filled button (Download script) is
    excepted: its border is the same color as its fill, so it draws no colored outline."""
    import re

    from sciai.ui.theme.theme import Theme

    setup_app()
    for name in ("light", "dark"):
        t = Theme.load(name)
        rules = [r for r in t.qss().split("}") if "#primary" not in r]
        for rule in rules:
            for decl in re.findall(r"border[\w-]*\s*:[^;]*", rule):
                for color in re.findall(r"#[0-9A-Fa-f]{6}\b", decl):
                    assert _sat(color) < 0.1, (name, decl.strip())
        for key in ("border", "border_strong", "selection", "edge_conflict", "pin", "warning"):
            assert _sat(t.hex(key)) < 0.1, (name, key)


def test_no_emoji_or_symbol_glyphs_in_the_ui():
    """Status, warnings and disclosure arrows are words or drawn icons, never glyphs."""
    import pathlib

    import sciai.ui

    banned = [(0x2190, 0x21FF), (0x2300, 0x23FF), (0x2460, 0x24FF), (0x25A0, 0x25FF), (0x2600, 0x27BF),
              (0x2900, 0x297F), (0x2B00, 0x2BFF), (0xFE00, 0xFE0F), (0x1F000, 0x1FAFF), (0x2295, 0x22A1)]
    found = []
    for path in pathlib.Path(sciai.ui.__file__).parent.rglob("*.py"):
        for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            for ch in line:
                if any(a <= ord(ch) <= b for a, b in banned):
                    found.append(f"{path.name}:{i} {ch!r}")
    assert not found, found


# 6. Composer --------------------------------------------------------------------------------------
def test_composer_card_strip_icons_and_send(fault_window):
    from sciai.ui.chat_bar import Composer

    win = fault_window((1200, 720))
    t = win.theme
    c = win.chat.composer
    assert (Composer.RADIUS, Composer.PAD) == (20, 16)
    assert t.hex("border").upper() == "#E4E2DB" and t.hex("status_strip_bg").upper() == "#F4F3EF"
    fx = c.graphicsEffect()
    assert (fx.blurRadius(), fx.offset().x(), fx.offset().y(), fx.color().alpha()) == (8, 0, 2, 13)
    img = c.grab().toImage()  # the widget itself, effect excluded
    assert _close(img.pixelColor(c.width() // 2, Composer.STRIP_H // 2 + 6), "#F4F3EF", tol=3)  # strip
    assert _close(img.pixelColor(c.width() // 2, Composer.STRIP_H + 4), t.hex("panel"), tol=3)  # card
    assert _close(img.pixelColor(c.width() // 2, 0), "#E4E2DB", tol=12)  # 1px border on top of the strip
    strip_font = c.status_strip.font()
    assert QFontMetricsF(strip_font).horizontalAdvance("Running") == pytest.approx(
        QFontMetricsF(t.ui_font("size_small_px")).horizontalAdvance("Running"), abs=0.6)  # 12.5px wide
    for b in (c.attach_btn, c.tools_btn, c.mic_btn):
        assert (b.iconSize().width(), b.width(), b.height()) == (20, 32, 32)  # 20px icon, 12px hit padding
    s = c.send_btn
    assert (s.width(), s.height()) == (36, 36)
    assert _close(s.grab().toImage().pixelColor(18, 6), t.hex("send"), tol=6)
    # 16px padding: icon glyph, input text and send button sit 16px inside the card
    glyph_left = _x(c.attach_btn, c) + (32 - 20) // 2
    assert glyph_left == 16
    assert _x(c.input.viewport(), c) == 16
    assert _x(s, c) + s.width() == c.width() - 16
    assert _y(c.input, c) == Composer.STRIP_H + 16


# 7. Vertical rhythm -------------------------------------------------------------------------------
def test_blocks_are_24px_apart_and_table_rows_on_the_8px_grid(fault_window):
    from sciai.ui.chat.inline import InlineText, TableCard
    from sciai.ui.chat_bar import AnswerCard

    win = fault_window((1200, 720))
    chat = win.chat
    lay = chat.messages
    seq = [lay.itemAt(i).widget() for i in range(lay.count())]
    seq = [w for w in seq if w is not None and w.isVisible()]
    table = next(w for w in seq if isinstance(w, TableCard))
    answer = next(w for w in seq if isinstance(w, AnswerCard))
    i = seq.index(table)
    assert isinstance(seq[i - 1], InlineText)  # the summary paragraph right above the table
    trio = [seq[i - 1], table, *seq[i + 1: seq.index(answer) + 1]]
    for a, b in zip(trio, trio[1:]):
        assert b.y() - (a.y() + a.height()) == 24, (type(a).__name__, type(b).__name__)
    assert (TableCard.HEAD_H, TableCard.ROW_H, TableCard.RADIUS) == (36, 40, 12)
    assert table.height() == 36 + 40 * len(table.rows) + 2
    t = win.theme
    assert t.radius("card") == 12 and t.hex("border").upper() == "#E4E2DB"
    img = table.grab().toImage()
    assert _close(img.pixelColor(table.width() - 6, 36 + 4), t.hex("table_body"), tol=4)  # warm white rows


# 8. Colors ----------------------------------------------------------------------------------------
WHITE_TEXT_ON_ACCENT = {"accent_text", "send_text", "pill_dark_text"}


def test_colors_are_warm_and_sampled():
    colors = _tokens()["color"]
    assert colors["panel"].upper() != "#FFFFFF" and colors["app_bg"].upper() != "#FFFFFF"
    for key, value in colors.items():
        if not isinstance(value, str) or not value.startswith("#"):
            continue
        h = value.lstrip("#")
        if len(h) == 8:  # #AARRGGBB: shadows are pure black at low alpha by design
            continue
        r, g, b = (int(h[k:k + 2], 16) for k in (0, 2, 4))
        sat = colorsys.rgb_to_hsv(r / 255, g / 255, b / 255)[1]
        if key in WHITE_TEXT_ON_ACCENT or key.startswith(("accent", "status_")) or sat > 0.2:
            continue  # accent tints, the chip colors Yeri specified, syntax colors
        assert not r == g == b, f"{key} {value} is pure gray"
        assert r >= b, f"{key} {value} is a cool neutral"
