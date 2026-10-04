"""Small layout helpers shared by the panes."""
from __future__ import annotations

from PyQt6.QtWidgets import QLayout


def clear_layout(layout: QLayout, keep: int = 0) -> None:
    """Remove and delete every item except the last ``keep`` ones.

    ``deleteLater`` alone leaves the widget parented and visible until the event
    loop runs, so old and new content paint on top of each other for a frame (or
    in a grab). Each widget is hidden and detached first, then deleted.
    """
    while layout.count() > keep:
        item = layout.takeAt(0)
        w = item.widget()
        if w is not None:
            w.hide()
            w.setParent(None)
            w.deleteLater()
        elif item.layout() is not None:
            clear_layout(item.layout())
