"""Structure depictions drawn by RDKit into QImages, in the theme's colours.

The same shape as ``plot_image``, for the same reason: drawing costs tens of milliseconds, so
images are cached by structure, size, pixel ratio and theme, and ``MolView`` keeps showing the
last one scaled while a splitter is being dragged and redraws once the size has settled.

Monochrome on purpose. The app's plain style has no colour in borders and no element palette
here either: the structure is drawn in the theme's ink on the panel, the way the plots are, so a
depiction sits in the window without turning a corner of it into a different design. ``export``
writes the same drawing to a file on white for sharing.

Coordinates are generated once per structure and cached with it, so the same molecule is drawn
the same way every time rather than flipping between layouts between redraws.
"""
from __future__ import annotations

import hashlib
from collections import OrderedDict
from typing import Any

from PyQt6.QtCore import QRectF, QSize, QTimer
from PyQt6.QtGui import QImage, QPainter, QPaintEvent
from PyQt6.QtSvg import QSvgRenderer
from PyQt6.QtWidgets import QSizePolicy, QWidget

from sciai.ui.theme.theme import Theme

CACHE_SIZE = 32
SETTLE_MS = 120
MIN_W, MIN_H = 60, 48
_cache: OrderedDict[tuple, QImage] = OrderedDict()
_coords: OrderedDict[str, Any] = OrderedDict()
COORD_CACHE = 64


def structure_key(smiles: str) -> str:
    return hashlib.sha1((smiles or "").encode()).hexdigest()


def _hex_to_rgb(value: str) -> tuple[float, float, float]:
    v = value.lstrip("#")
    return tuple(int(v[i:i + 2], 16) / 255.0 for i in (0, 2, 4))  # type: ignore[return-value]


def prepared(smiles: str) -> Any:
    """The molecule with 2D coordinates, generated once and kept so a redraw never moves it."""
    key = structure_key(smiles)
    cached = _coords.get(key)
    if cached is not None:
        _coords.move_to_end(key)
        return cached
    from rdkit import Chem
    from rdkit.Chem import rdDepictor

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"not a structure that can be drawn: {smiles[:60]!r}")
    rdDepictor.SetPreferCoordGen(True)
    rdDepictor.Compute2DCoords(mol)
    _coords[key] = mol
    while len(_coords) > COORD_CACHE:
        _coords.popitem(last=False)
    return mol


def svg(smiles: str, width: int, height: int, *, theme: Theme | None = None,
        compact: bool = False, white: bool = False) -> str:
    """The structure as an SVG string, in the theme's ink or black on white for export."""
    from rdkit.Chem.Draw import rdMolDraw2D

    mol = prepared(smiles)
    drawer = rdMolDraw2D.MolDraw2DSVG(max(MIN_W, int(width)), max(MIN_H, int(height)))
    options = drawer.drawOptions()
    options.useBWAtomPalette()          # plain style: no element colours
    options.clearBackground = True
    if white or theme is None:
        options.setBackgroundColour((1.0, 1.0, 1.0))
    else:
        options.setBackgroundColour(_hex_to_rgb(theme.hex("panel")))
        options.setSymbolColour(_hex_to_rgb(theme.hex("text")))
        options.setLegendColour(_hex_to_rgb(theme.hex("text_secondary")))
    if compact:
        options.bondLineWidth = 1
        options.minFontSize = 7
        options.padding = 0.02
    rdMolDraw2D.PrepareAndDrawMolecule(drawer, mol)
    drawer.FinishDrawing()
    return drawer.GetDrawingText()


def mol_image(theme: Theme, smiles: str, width: int, height: int, *, dpr: float = 1.0,
              compact: bool = False, key: str | None = None) -> QImage:
    """The structure drawn at ``width`` x ``height`` logical pixels."""
    width, height = max(MIN_W, int(width)), max(MIN_H, int(height))
    dpr = max(1.0, round(float(dpr), 2))
    k = (key or structure_key(smiles), width, height, dpr, compact,
         theme.hex("panel"), theme.hex("text"))
    cached = _cache.get(k)
    if cached is not None:
        _cache.move_to_end(k)
        return cached
    markup = svg(smiles, round(width * dpr), round(height * dpr), theme=theme, compact=compact)
    renderer = QSvgRenderer(markup.encode())
    image = QImage(round(width * dpr), round(height * dpr), QImage.Format.Format_ARGB32_Premultiplied)
    image.fill(0)
    painter = QPainter(image)
    renderer.render(painter)
    painter.end()
    image.setDevicePixelRatio(dpr)
    _cache[k] = image
    while len(_cache) > CACHE_SIZE:
        _cache.popitem(last=False)
    return image


def export(smiles: str, path: str, *, width: int = 900, height: int = 700) -> None:
    """Write the depiction to a .svg or .png file, black on white."""
    lower = str(path).lower()
    if lower.endswith(".svg"):
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(svg(smiles, width, height, white=True))
        return
    from rdkit.Chem.Draw import rdMolDraw2D

    mol = prepared(smiles)
    drawer = rdMolDraw2D.MolDraw2DCairo(width, height)
    options = drawer.drawOptions()
    options.useBWAtomPalette()
    options.setBackgroundColour((1.0, 1.0, 1.0))
    rdMolDraw2D.PrepareAndDrawMolecule(drawer, mol)
    drawer.FinishDrawing()
    with open(path, "wb") as handle:
        handle.write(drawer.GetDrawingText())


class MolView(QWidget):
    """A structure filling the widget, redrawn when the size settles after a resize."""

    def __init__(self, theme: Theme, smiles: str | None = None, *, compact: bool = False) -> None:
        super().__init__()
        self.theme, self.compact = theme, compact
        self.smiles: str | None = None
        self._key: str | None = None
        self._image: QImage | None = None
        self._settle = QTimer(self)
        self._settle.setSingleShot(True)
        self._settle.setInterval(SETTLE_MS)
        self._settle.timeout.connect(self._render)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.set_structure(smiles)

    def sizeHint(self) -> QSize:  # noqa: N802
        return QSize(420, 300)

    def set_structure(self, smiles: str | None) -> None:
        self.smiles = smiles or None
        self._key = structure_key(smiles) if smiles else None
        self._image = None
        self._settle.stop()
        self.update()

    def image(self) -> QImage | None:
        if self.smiles is not None and self._image is None:
            self._render()
        return self._image

    def _render(self) -> None:
        if self.smiles is None or self.width() < MIN_W or self.height() < MIN_H:
            return
        try:
            self._image = mol_image(self.theme, self.smiles, self.width(), self.height(),
                                    dpr=self.devicePixelRatioF(), compact=self.compact,
                                    key=self._key)
        except ValueError:
            self._image = None
        self.update()

    def resizeEvent(self, event) -> None:  # noqa: N802, ANN001
        super().resizeEvent(event)
        if self._image is None:
            return  # the first paint draws at the final size
        self._settle.start()

    def paintEvent(self, event: QPaintEvent) -> None:  # noqa: N802
        if self.smiles is None:
            return
        if self._image is None:
            self._render()
            if self._image is None:
                return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        painter.drawImage(QRectF(0, 0, self.width(), self.height()), self._image)
        painter.end()


__all__ = ["MolView", "export", "mol_image", "prepared", "structure_key", "svg"]
