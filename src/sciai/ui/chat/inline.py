"""Painted chat blocks: flowing text with inline code chips, and table cards.

QLabel rich text can give a span a background but not a radius or padding, so
inline chips (tool names, expressions, node ids) are laid out and painted here.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from PyQt6.QtCore import QRectF, QSize, Qt
from PyQt6.QtGui import QBrush, QFontMetricsF, QPainter, QPaintEvent, QPen
from PyQt6.QtWidgets import QSizePolicy, QWidget

from sciai.ui.theme.theme import STATUS_ICON, Theme

# `backticks` always become chips; node handles and dotted tool names are chipped automatically.
_CHIP_RE = re.compile(r"`([^`]+)`|\b(n\d+)\b|\b([a-z]+\.[a-z_]+)\b")
CHIP_PAD_X = 4.0
CHIP_RADIUS = 4.0


def split_chips(text: str) -> list[tuple[str, bool]]:
    out: list[tuple[str, bool]] = []
    pos = 0
    for m in _CHIP_RE.finditer(text):
        if m.start() > pos:
            out.append((text[pos:m.start()], False))
        out.append((next(g for g in m.groups() if g), True))
        pos = m.end()
    if pos < len(text):
        out.append((text[pos:], False))
    return out


@dataclass
class _Box:
    x: float
    line: int
    w: float
    text: str
    chip: bool


class InlineText(QWidget):
    """Wrapped chat text (15px, line-height 1.6) with pale-blue mono chips."""

    def __init__(self, theme: Theme, text: str, size_key: str = "size_chat_px", color_key: str = "text") -> None:
        super().__init__()
        self.theme = theme
        self.text = text
        self.color_key = color_key
        self.font_ = theme.ui_font(size_key)
        self.mono = theme.mono_font("size_mono_px")
        self.line_h = round(theme.px(size_key) * float(theme.tokens["font"].get("chat_line_height", 1.6)))
        self.parts = split_chips(text)
        sp = QSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
        sp.setHeightForWidth(True)
        self.setSizePolicy(sp)

    # layout ------------------------------------------------------------------
    def _tokens(self) -> list[tuple[str, bool]]:
        toks: list[tuple[str, bool]] = []
        for text, chip in self.parts:
            if chip:
                toks.append((text, True))
            else:
                toks.extend((w, False) for w in re.findall(r"\S+\s*|\s+", text))
        return toks

    def _layout(self, width: float) -> tuple[list[_Box], int]:
        fm, fmm = QFontMetricsF(self.font_), QFontMetricsF(self.mono)
        boxes: list[_Box] = []
        x, line = 0.0, 0
        for tok, chip in self._tokens():
            if chip:
                w = fmm.horizontalAdvance(tok) + 2 * CHIP_PAD_X
                trail = fm.horizontalAdvance(" ") * 0.25
            else:
                w = fm.horizontalAdvance(tok.rstrip())
                trail = fm.horizontalAdvance(tok) - w
            if x > 0 and x + w > width:
                line += 1
                x = 0.0
                if not chip and not tok.strip():
                    continue
            shown = tok
            if w > width:  # a single token wider than the line: elide it
                f = fmm if chip else fm
                shown = f.elidedText(tok, Qt.TextElideMode.ElideRight, width - (2 * CHIP_PAD_X if chip else 0))
                w = width
            boxes.append(_Box(x, line, w, shown, chip))
            x += w + trail
        return boxes, line + 1 if boxes else 1

    def natural_width(self) -> float:
        boxes, _ = self._layout(10_000)
        return max((b.x + b.w for b in boxes), default=0) + 1

    def hasHeightForWidth(self) -> bool:  # noqa: N802
        return True

    def heightForWidth(self, w: int) -> int:  # noqa: N802
        return self._layout(max(1, w))[1] * self.line_h

    def sizeHint(self) -> QSize:  # noqa: N802
        w = int(min(self.natural_width(), 720))
        return QSize(w, self.heightForWidth(w))

    def minimumSizeHint(self) -> QSize:  # noqa: N802
        return QSize(40, self.line_h)

    def plain(self) -> str:
        return self.text

    def paintEvent(self, event: QPaintEvent) -> None:  # noqa: N802
        t = self.theme
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        boxes, _ = self._layout(self.width())
        chip_h = QFontMetricsF(self.mono).height() + 4
        for b in boxes:
            top = b.line * self.line_h
            if b.chip:
                rect = QRectF(b.x, top + (self.line_h - chip_h) / 2, b.w, chip_h)
                p.setPen(Qt.PenStyle.NoPen)
                p.setBrush(QBrush(t.c("chip_code_bg")))
                p.drawRoundedRect(rect, CHIP_RADIUS, CHIP_RADIUS)
                p.setFont(self.mono)
                p.setPen(t.c("chip_code_text"))
                p.drawText(rect.adjusted(CHIP_PAD_X, 0, -CHIP_PAD_X, 0), Qt.AlignmentFlag.AlignVCenter, b.text)
            else:
                p.setFont(self.font_)
                p.setPen(t.c(self.color_key))
                p.drawText(QRectF(b.x, top, b.w + 2, self.line_h), Qt.AlignmentFlag.AlignVCenter, b.text)
        p.end()


@dataclass
class Column:
    title: str
    kind: str = "text"  # text | code | status
    min_w: float = 60
    flex: bool = False


class TableCard(QWidget):
    """Rounded table: gray header row, thin dividers, expression cells as chips, status as soft chips."""

    ROW_H = 36
    PAD = 12

    def __init__(self, theme: Theme, columns: list[Column], rows: list[list[str]], caption: str = "") -> None:
        super().__init__()
        self.theme, self.columns, self.rows, self.caption = theme, columns, rows, caption
        self.font_ = theme.ui_font("size_ui_px")
        self.head_font = theme.ui_font("size_ui_px")
        self.mono = theme.mono_font("size_mono_px")
        self.chip_font = theme.ui_font("size_node_small_px", bold=True)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setFixedHeight(self.ROW_H * (len(rows) + 1) + 2)

    def minimumSizeHint(self) -> QSize:  # noqa: N802
        nat = self.natural_widths()
        return QSize(int(sum(c.min_w if c.flex else w for c, w in zip(self.columns, nat))) + 2, self.height())

    def sizeHint(self) -> QSize:  # noqa: N802
        return QSize(640, self.height())

    def _cell_w(self, col: Column, cell: str) -> float:
        """Width a cell needs, padding included (status/code chips at their natural size)."""
        if not cell:
            return 2 * self.PAD
        if col.kind == "code":
            return QFontMetricsF(self.mono).horizontalAdvance(cell) + 2 * CHIP_PAD_X + 2 + 2 * self.PAD
        if col.kind == "status":
            return QFontMetricsF(self.chip_font).horizontalAdvance(f"{STATUS_ICON.get(cell, '')} {cell}") + 14 \
                + 2 * self.PAD
        return QFontMetricsF(self.font_).horizontalAdvance(cell) + 2 * self.PAD

    def natural_widths(self) -> list[float]:
        head = QFontMetricsF(self.head_font)
        return [max(c.min_w, head.horizontalAdvance(c.title) + 2 * self.PAD,
                    *(self._cell_w(c, r[i]) for r in self.rows if i < len(r)))
                for i, c in enumerate(self.columns)]

    def _widths(self) -> list[float]:
        """Fixed columns get their natural width; flex columns share what is left (and elide)."""
        nat = self.natural_widths()
        fixed = sum(w for c, w in zip(self.columns, nat) if not c.flex)
        flex = [c for c in self.columns if c.flex]
        rest = max(0.0, self.width() - 2 - fixed)
        return [w if not c.flex else max(c.min_w, rest / len(flex)) for c, w in zip(self.columns, nat)]

    def elided(self) -> list[str]:
        """Cells that do not fit at the current width. Flex columns (long expressions) may elide by
        design; anything listed here from a fixed column is real clipping (used by the metrics test)."""
        out = []
        widths = self._widths()
        x = 1.0
        for c, w in zip(self.columns, widths):
            x += w
            if c.flex:
                continue
            if x > self.width() + 0.5:
                out.append(f"table column {c.title!r} past the card edge")
        nat = self.natural_widths()
        for c, w, n in zip(self.columns, widths, nat):
            if not c.flex and w + 0.5 < n:
                out.append(f"table column {c.title!r} narrower than its cells")
        return out

    def plain(self) -> str:
        head = " | ".join(c.title for c in self.columns)
        return "\n".join([head] + [" | ".join(r) for r in self.rows])

    def paintEvent(self, event: QPaintEvent) -> None:  # noqa: N802
        t = self.theme
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        outer = QRectF(0.5, 0.5, self.width() - 1, self.height() - 1)
        r = 10.0
        p.setPen(QPen(t.c("border_strong"), 1))
        p.setBrush(QBrush(t.c("app_bg")))
        p.drawRoundedRect(outer, r, r)
        # header band (rounded top only)
        p.save()
        p.setClipRect(QRectF(0, 0, self.width(), self.ROW_H))
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(t.c("table_header_bg")))
        p.drawRoundedRect(outer.adjusted(0.5, 0.5, -0.5, 0), r, r)
        p.restore()
        widths = self._widths()
        p.setFont(self.head_font)
        p.setPen(t.c("text"))
        x = 1.0
        for col, w in zip(self.columns, widths):
            align = Qt.AlignmentFlag.AlignVCenter | (Qt.AlignmentFlag.AlignRight if col.kind == "num"
                                                    else Qt.AlignmentFlag.AlignLeft)
            p.drawText(QRectF(x + self.PAD, 0, w - 2 * self.PAD, self.ROW_H), align, col.title)
            x += w
        for i, row in enumerate(self.rows):
            top = self.ROW_H * (i + 1)
            p.setPen(QPen(t.c("border"), 1))
            p.drawLine(QRectF(1, top, self.width() - 2, 0).topLeft(), QRectF(1, top, self.width() - 2, 0).topRight())
            x = 1.0
            for col, w, cell in zip(self.columns, widths, row):
                self._cell(p, QRectF(x + self.PAD, top, w - 2 * self.PAD, self.ROW_H), col, cell)
                x += w
        p.end()

    def _cell(self, p: QPainter, rect: QRectF, col: Column, cell: str) -> None:
        t = self.theme
        if not cell:
            return
        if col.kind == "code":
            fm = QFontMetricsF(self.mono)
            text = fm.elidedText(cell, Qt.TextElideMode.ElideRight, rect.width() - 2 * CHIP_PAD_X - 2)
            w = fm.horizontalAdvance(text) + 2 * CHIP_PAD_X + 2
            h = fm.height() + 6
            chip = QRectF(rect.left(), rect.center().y() - h / 2, w, h)
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QBrush(t.c("chip_code_bg")))
            p.drawRoundedRect(chip, CHIP_RADIUS, CHIP_RADIUS)
            p.setFont(self.mono)
            p.setPen(t.c("chip_code_text"))
            p.drawText(chip.adjusted(CHIP_PAD_X, 0, -CHIP_PAD_X, 0), Qt.AlignmentFlag.AlignVCenter, text)
        elif col.kind == "status":
            fg, bg = t.status_colors(cell)
            p.setFont(self.chip_font)
            fm = QFontMetricsF(self.chip_font)
            label = f"{STATUS_ICON.get(cell, '')} {cell}"
            w = min(rect.width(), fm.horizontalAdvance(label) + 14)
            chip = QRectF(rect.left(), rect.center().y() - 10, w, 20)
            p.setBrush(QBrush(bg))
            p.setPen(QPen(fg, 1, Qt.PenStyle.DashLine) if cell == "invalidated" else Qt.PenStyle.NoPen)
            p.drawRoundedRect(chip, 10, 10)
            p.setPen(fg)
            p.drawText(chip, Qt.AlignmentFlag.AlignCenter, label)
        else:
            p.setFont(self.font_)
            p.setPen(t.c("table_text"))
            align = Qt.AlignmentFlag.AlignVCenter | (Qt.AlignmentFlag.AlignRight if col.kind == "num"
                                                    else Qt.AlignmentFlag.AlignLeft)
            p.drawText(rect, align, QFontMetricsF(self.font_).elidedText(cell, Qt.TextElideMode.ElideRight,
                                                                         rect.width()))


class PlotCard(QWidget):
    """A plot result drawn inline in the thread (from a plotspec: x, y, labels)."""

    def __init__(self, theme: Theme, spec: dict, title: str) -> None:
        super().__init__()
        self.theme, self.spec, self.title = theme, spec, title
        self.setFixedHeight(240)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def sizeHint(self) -> QSize:  # noqa: N802
        return QSize(560, 240)

    def paintEvent(self, event: QPaintEvent) -> None:  # noqa: N802
        t = self.theme
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setPen(QPen(t.c("border"), 1))
        p.setBrush(QBrush(t.c("panel")))
        p.drawRoundedRect(QRectF(0.5, 0.5, self.width() - 1, self.height() - 1), 10, 10)
        p.setFont(t.ui_font("size_small_px"))
        p.setPen(t.c("text_title"))
        p.drawText(QRectF(0, 10, self.width(), 18), Qt.AlignmentFlag.AlignHCenter, self.title)
        area = QRectF(44, 36, self.width() - 64, self.height() - 64)
        pts = [(x, y) for x, y in zip(self.spec.get("x", []), self.spec.get("y", [])) if y is not None]
        p.setPen(QPen(t.c("border_strong"), 1))
        p.drawLine(area.bottomLeft(), area.bottomRight())
        p.drawLine(area.bottomLeft(), area.topLeft())
        if len(pts) < 2:
            p.end()
            return
        xs, ys = [a for a, _ in pts], [b for _, b in pts]
        x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
        if x1 == x0:
            p.end()
            return
        if y1 == y0:
            y0, y1 = y0 - 1, y1 + 1
        p.setPen(t.c("text_secondary"))
        p.drawText(QRectF(0, area.top() - 8, 40, 16), Qt.AlignmentFlag.AlignRight, f"{y1:.3g}")
        p.drawText(QRectF(0, area.bottom() - 8, 40, 16), Qt.AlignmentFlag.AlignRight, f"{y0:.3g}")
        p.drawText(QRectF(area.left(), area.bottom() + 4, 80, 16), Qt.AlignmentFlag.AlignLeft, f"{x0:.3g}")
        p.drawText(QRectF(area.right() - 80, area.bottom() + 4, 80, 16), Qt.AlignmentFlag.AlignRight, f"{x1:.3g}")
        from PyQt6.QtGui import QPainterPath

        path = QPainterPath()
        for i, (x, y) in enumerate(pts):
            px = area.left() + (x - x0) / (x1 - x0) * area.width()
            py = area.bottom() - (y - y0) / (y1 - y0) * area.height()
            if i == 0:
                path.moveTo(px, py)
            else:
                path.lineTo(px, py)
        p.setPen(QPen(t.c("accent"), 1.6))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawPath(path)
        p.end()
