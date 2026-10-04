"""Plot specs drawn by matplotlib (graph.plotspec) into QImages, in the theme's colors and font,
for the chat's plot cards, the graph's plot nodes and the node view's Plot tab.

Rendering a figure costs tens of milliseconds, so images are cached by spec, size, pixel ratio
and theme. ``PlotView`` keeps showing the last image (scaled) while it is being resized and
re-renders once the size has settled, so dragging a splitter never renders on every pixel.
"""
from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from importlib import resources
from typing import Any

from PyQt6.QtCore import QRectF, QSize, QTimer
from PyQt6.QtGui import QImage, QPainter, QPaintEvent
from PyQt6.QtWidgets import QSizePolicy, QWidget

from sciai.graph import plotspec
from sciai.ui.theme.theme import Theme

CACHE_SIZE = 48
SETTLE_MS = 120  # a resize re-renders once the size has been still this long
_cache: OrderedDict[tuple, QImage] = OrderedDict()
_font_family: str | None = None
_font_tried = False


def theme_colors(theme: Theme) -> dict[str, str]:
    """The plot palette from the theme: ink series on the panel, gray axes, the accent last."""
    return {"background": theme.hex("panel"), "ink": theme.hex("text"), "axis": theme.hex("border_strong"),
            "grid": theme.hex("subtle"), "muted": theme.hex("text_secondary"), "accent": theme.hex("accent"),
            "text": theme.hex("text_title")}


def plot_font() -> str | None:
    """Register the bundled Inter with matplotlib once; None (matplotlib's default) if it can't."""
    global _font_family, _font_tried
    if _font_tried:
        return _font_family
    _font_tried = True
    try:
        from matplotlib import font_manager

        folder = resources.files("sciai.ui.theme").joinpath("fonts")
        for name in ("Inter-Regular.ttf", "Inter-SemiBold.ttf"):
            with resources.as_file(folder.joinpath(name)) as path:
                font_manager.fontManager.addfont(str(path))
        _font_family = "Inter"
    except Exception:  # noqa: BLE001 - a missing font only changes the look
        _font_family = None
    return _font_family


def spec_key(value: dict[str, Any]) -> str:
    return hashlib.sha1(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def plot_image(theme: Theme, value: dict[str, Any], width: int, height: int, *, dpr: float = 1.0,
               compact: bool = False, key: str | None = None) -> QImage:
    """The spec drawn at ``width`` x ``height`` logical pixels (``dpr`` device pixels per pixel)."""
    width, height, dpr = max(40, int(width)), max(30, int(height)), max(1.0, round(float(dpr), 2))
    colors = theme_colors(theme)
    k = (key or spec_key(value), width, height, dpr, compact, tuple(sorted(colors.items())))
    img = _cache.get(k)
    if img is not None:
        _cache.move_to_end(k)
        return img
    png = plotspec.render_png(value, round(width * dpr), round(height * dpr), dpi=round(100 * dpr),
                              colors=colors, compact=compact, font_family=plot_font())
    img = QImage.fromData(png, "PNG")
    img.setDevicePixelRatio(dpr)
    _cache[k] = img
    while len(_cache) > CACHE_SIZE:
        _cache.popitem(last=False)
    return img


def without_title(value: dict[str, Any]) -> dict[str, Any]:
    """The spec with its title removed, for a card that prints its own caption."""
    return {**plotspec.normalize(value), "title": ""}


class PlotView(QWidget):
    """A plot spec filling the widget. Re-renders when the size settles after a resize."""

    def __init__(self, theme: Theme, value: dict[str, Any] | None = None, *, compact: bool = False) -> None:
        super().__init__()
        self.theme, self.compact = theme, compact
        self.value: dict[str, Any] | None = None
        self._key: str | None = None
        self._image: QImage | None = None
        self._settle = QTimer(self)
        self._settle.setSingleShot(True)
        self._settle.setInterval(SETTLE_MS)
        self._settle.timeout.connect(self._render)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.set_spec(value)

    def sizeHint(self) -> QSize:  # noqa: N802
        return QSize(480, 300)

    def set_spec(self, value: dict[str, Any] | None) -> None:
        self.value = value
        self._key = spec_key(value) if value is not None else None
        self._image = None
        self._settle.stop()
        self.update()

    def image(self) -> QImage | None:
        """The rendered image (rendering now if there is none yet)."""
        if self.value is not None and self._image is None:
            self._render()
        return self._image

    def _render(self) -> None:
        if self.value is None or self.width() < 40 or self.height() < 30:
            return
        self._image = plot_image(self.theme, self.value, self.width(), self.height(),
                                 dpr=self.devicePixelRatioF(), compact=self.compact, key=self._key)
        self.update()

    def resizeEvent(self, event) -> None:  # noqa: N802, ANN001
        super().resizeEvent(event)
        if self._image is None:
            return  # the first paint renders at the final size
        self._settle.start()

    def paintEvent(self, event: QPaintEvent) -> None:  # noqa: N802
        if self.value is None:
            return
        if self._image is None:
            self._render()
            if self._image is None:
                return
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        p.drawImage(QRectF(0, 0, self.width(), self.height()), self._image)
        p.end()


__all__ = ["PlotView", "plot_font", "plot_image", "spec_key", "theme_colors", "without_title"]
