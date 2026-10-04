"""Right card, structured like the reference's Notebook panel.

Header: icon + "Workspace". Below: pill tabs ("Graph", then a pill for the selected node,
e.g. "n2 · f'(x)") on the left and a "Live"/"Idle" status dropdown on the right.
Pages: the graph (with a slim legend row) and the selected node's panel.
"""
from __future__ import annotations

from typing import Any, Callable

from PyQt6.QtCore import QSize, Qt, pyqtSignal
from PyQt6.QtGui import QFontMetrics
from PyQt6.QtWidgets import (
    QButtonGroup,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMenu,
    QStackedWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from sciai.graph.model import Node
from sciai.store.repository import Repository
from sciai.ui.canvas.graph_view import GraphView
from sciai.ui.canvas.legend import Legend
from sciai.ui.icons import icon
from sciai.ui.mathtext import prose_text
from sciai.ui.theme.theme import Theme
from sciai.ui.workspace.node_panel import NodePanel
from sciai.ui.icons import paint_icon

NODE_PILL_MAX_W = 190  # px of label text; longer titles are elided (full title in the tooltip)


def node_pill_text(handle: str, title: str) -> str:
    return f"{handle} · {prose_text(title)}"


class LiveButton(QToolButton):
    """Status dropdown: green dot + "Live" / gray dot + "Idle", with a chevron drawn on the right."""

    CHEVRON = 18

    def __init__(self, theme: Theme) -> None:
        super().__init__()
        self.theme = theme

    def paintEvent(self, event) -> None:  # noqa: N802, ANN001
        super().paintEvent(event)
        from PyQt6.QtGui import QPainter

        p = QPainter(self)
        paint_icon(p, "chevron_down", self.theme.c("text_secondary"), self.width() - self.CHEVRON - 2,
                   (self.height() - 12) / 2, 12)
        p.end()


class Workspace(QWidget):
    stop = pyqtSignal()

    def __init__(self, repo: Repository, theme: Theme, env_provider: Callable[[], dict[str, Any]]) -> None:
        super().__init__()
        self.setObjectName("cardBody")
        self.theme = theme
        t = theme
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)

        head = QHBoxLayout()
        head.setContentsMargins(14, 12, 14, 6)
        head.setSpacing(6)
        mark = QLabel()
        mark.setPixmap(icon("notebook", t.hex("text_title")).pixmap(QSize(16, 16)))
        self.title = QLabel("Workspace")
        self.title.setObjectName("paneTitle")
        head.addWidget(mark)
        head.addWidget(self.title)
        head.addStretch(1)
        lay.addLayout(head)

        tabs = QHBoxLayout()
        tabs.setContentsMargins(10, 0, 12, 8)
        tabs.setSpacing(4)
        self.graph_tab = QToolButton()
        self.graph_tab.setObjectName("pillTab")
        self.graph_tab.setText("Graph")
        self.graph_tab.setCheckable(True)
        self.node_tab = QToolButton()
        self.node_tab.setObjectName("pillTab")
        self.node_tab.setCheckable(True)
        self.node_tab.hide()
        self._group = group = QButtonGroup(self)
        group.setExclusive(True)
        group.addButton(self.graph_tab)
        group.addButton(self.node_tab)
        self.graph_tab.clicked.connect(self.show_graph)
        self.node_tab.clicked.connect(self.show_node_tab)
        tabs.addWidget(self.graph_tab)
        tabs.addWidget(self.node_tab)
        tabs.addStretch(1)
        self.live_btn = LiveButton(t)
        self.live_btn.setObjectName("liveButton")
        self.live_btn.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.live_btn.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self.live_btn.setIconSize(QSize(10, 10))
        menu = QMenu(self.live_btn)
        self.stop_action = menu.addAction("Stop the run", self.stop.emit)
        menu.addAction("Fit graph\tCtrl+0", lambda: (self.show_graph(), self.graph.fit_all()))
        self.live_btn.setMenu(menu)
        tabs.addWidget(self.live_btn)
        lay.addLayout(tabs)
        rule = QFrame()
        rule.setObjectName("hairline")
        lay.addWidget(rule)

        self.stack = QStackedWidget()
        graph_page = QWidget()
        gl = QVBoxLayout(graph_page)
        gl.setContentsMargins(0, 0, 0, 0)
        gl.setSpacing(0)
        self.graph = GraphView(t)
        self.legend = Legend(t)
        gl.addWidget(self.graph, 1)
        gl.addWidget(self.legend)
        self.node_panel = NodePanel(repo, t, env_provider)
        self.node_panel.closed.connect(self.close_node)
        self.stack.addWidget(graph_page)
        self.stack.addWidget(self.node_panel)
        lay.addWidget(self.stack, 1)
        self.set_live(False)
        self.show_graph()

    # tabs -----------------------------------------------------------------------
    def show_graph(self) -> None:
        self.graph_tab.setChecked(True)
        self.stack.setCurrentIndex(0)

    def show_node_tab(self) -> None:
        if self.node_panel.node is None:
            return
        self.node_tab.setChecked(True)
        self.stack.setCurrentIndex(1)

    def show_environment(self) -> None:
        """Settings gear: the Environment tab (works with or without a selected node)."""
        self.stack.setCurrentIndex(1)
        self.node_panel.tabs.setCurrentIndex(3)
        self.node_panel._refresh_env()
        if self.node_panel.node is not None:
            self.node_tab.setChecked(True)
        else:
            self._group.setExclusive(False)
            self.graph_tab.setChecked(False)
            self._group.setExclusive(True)

    def current_tab(self) -> str:
        return "Graph" if self.stack.currentIndex() == 0 else self.node_tab.text()

    def close_node(self) -> None:
        self.node_tab.hide()
        self.node_panel.show_node(None, self.graph.handles)
        self.graph.select(None)
        self.show_graph()

    def show_node(self, node: Node | None, handles: dict[str, str]) -> None:
        """Fill the node panel and the node pill; stays on the current tab (click the pill to open it)."""
        self.node_panel.show_node(node, handles)
        if node is None:
            self.node_tab.hide()
            if self.stack.currentIndex() == 1:
                self.show_graph()
            return
        full = node_pill_text(handles.get(node.id, "earlier version"), node.title)
        fm = QFontMetrics(self.node_tab.font())
        self.node_tab.setText(fm.elidedText(full, Qt.TextElideMode.ElideRight, NODE_PILL_MAX_W))
        self.node_tab.setToolTip(full)
        self.node_tab.show()

    # status -------------------------------------------------------------------------
    def set_live(self, live: bool) -> None:
        t = self.theme
        self.live = live
        self.live_btn.setText("Live" if live else "Idle")
        self.live_btn.setIcon(icon("dot", t.hex("live") if live else t.hex("text_faint")))
        self.live_btn.setToolTip("The controller is running" if live else "Nothing is running")
        self.stop_action.setEnabled(live)
