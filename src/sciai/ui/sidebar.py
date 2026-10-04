"""Left card, mirroring the reference: wordmark, project row, New / Customize / Files,
the "Active" session list (hollow bullets, medium-weight selection, running-count badge),
a collapsible knowledge base with green checks, and a settings gear pinned bottom-left."""
from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import QSize, Qt, pyqtSignal
from PyQt6.QtGui import QAction, QEnterEvent, QFont, QMouseEvent, QPainter, QPaintEvent
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QMenu,
    QPushButton,
    QScrollArea,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from sciai.store.repository import Repository
from sciai.ui.chat_bar import PRODUCT_NAME
from sciai.ui.icons import icon
from sciai.ui.layout_util import clear_layout
from sciai.ui.mathtext import prose_text, value_text
from sciai.ui.shell import ElidedLabel
from sciai.ui.theme.theme import Theme

FILES_DIR = Path("~/.sciai/files").expanduser()
SUBTITLE = "Local research workspace"
ROW_H = 32


class ListRow(QWidget):
    """A plain sidebar row: small icon, elided text, optional count badge. Selected = medium-weight
    text with no fill (as in the reference); hover = soft pill."""

    clicked = pyqtSignal()

    def __init__(self, theme: Theme, icon_name: str, icon_color: str, text: str, tooltip: str = "") -> None:
        super().__init__()
        self.theme = theme
        self.selected = False
        self._hover = False
        self.setFixedHeight(ROW_H)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setToolTip(tooltip or text)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(5, 0, 6, 0)
        lay.setSpacing(6)
        mark = QLabel()
        mark.setPixmap(icon(icon_name, theme.hex(icon_color)).pixmap(QSize(14, 14)))
        mark.setFixedWidth(14)
        lay.addWidget(mark)
        self.label = ElidedLabel(text)
        lay.addWidget(self.label, 1)
        self.badge = QLabel()
        self.badge.setObjectName("badge")
        self.badge.setFixedHeight(18)  # a small gray count, not a row-high box
        self.badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.badge.hide()
        lay.addWidget(self.badge, 0, Qt.AlignmentFlag.AlignVCenter)

    def set_badge(self, count: int | None) -> None:
        self.badge.setVisible(count is not None)
        self.badge.setText(str(count) if count is not None else "")

    def set_selected(self, on: bool) -> None:
        self.selected = on
        f = self.label.font()
        f.setWeight(QFont.Weight.Medium if on else QFont.Weight.Normal)
        self.label.setFont(f)
        self.label._elide()
        self.update()

    def enterEvent(self, event: QEnterEvent) -> None:  # noqa: N802
        self._hover = True
        self.update()

    def leaveEvent(self, event) -> None:  # noqa: N802, ANN001
        self._hover = False
        self.update()

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit()

    def paintEvent(self, event: QPaintEvent) -> None:  # noqa: N802
        if not self._hover:
            return
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(self.theme.c("hover"))
        p.drawRoundedRect(self.rect().adjusted(0, 1, 0, -1), 8, 8)
        p.end()


class Sidebar(QWidget):
    new_session = pyqtSignal()
    open_session = pyqtSignal(str)
    open_node = pyqtSignal(str, str)  # session_id, node_id
    settings = pyqtSignal()
    toggle_workspace = pyqtSignal()
    add_file = pyqtSignal()

    def __init__(self, repo: Repository, theme: Theme) -> None:
        super().__init__()
        self.setObjectName("cardBody")
        self.repo, self.theme = repo, theme
        self.current: str | None = None
        self._history: list[str] = []
        self._running: tuple[str, int] | None = None
        self.session_rows: dict[str, ListRow] = {}
        t = theme
        ink = t.hex("text")

        lay = QVBoxLayout(self)
        lay.setContentsMargins(10, 12, 10, 10)
        lay.setSpacing(0)
        brand = QLabel(PRODUCT_NAME)
        brand.setObjectName("brand")
        brand.setContentsMargins(1, 0, 0, 0)
        lay.addWidget(brand)
        tag = QLabel(SUBTITLE)
        tag.setObjectName("secondary")
        tag.setContentsMargins(1, 0, 0, 0)
        lay.addWidget(tag)
        lay.addSpacing(12)

        # Project row: back arrow, current session name, chevron with the session menu.
        proj = QHBoxLayout()
        proj.setContentsMargins(0, 0, 0, 0)
        proj.setSpacing(2)
        self.back_btn = QToolButton()
        self.back_btn.setObjectName("iconButton")
        self.back_btn.setIcon(icon("arrow_left", ink))
        self.back_btn.setToolTip("Back to the previous session")
        self.back_btn.clicked.connect(self._back)
        self.project_name = ElidedLabel("New session")
        self.project_name.setObjectName("projectName")
        self.chevron = QToolButton()
        self.chevron.setObjectName("iconButton")
        self.chevron.setIcon(icon("chevron_down", ink))
        self.chevron.setToolTip("Switch session")
        self.chevron.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self.session_menu = QMenu(self.chevron)
        self.chevron.setMenu(self.session_menu)
        proj.addWidget(self.back_btn)
        proj.addWidget(self.project_name, 1)
        proj.addWidget(self.chevron)
        lay.addLayout(proj)
        lay.addSpacing(6)

        def nav(text: str, icon_name: str) -> QPushButton:
            b = QPushButton(text)
            b.setObjectName("nav")
            b.setIcon(icon(icon_name, ink))
            b.setIconSize(QSize(16, 16))
            lay.addWidget(b)
            return b

        self.new_btn = nav("New", "plus")
        self.new_btn.clicked.connect(self.new_session.emit)
        self.customize_btn = nav("Customize", "briefcase")
        cmenu = QMenu(self.customize_btn)
        self.workspace_action = QAction("Show workspace", cmenu)
        self.workspace_action.setCheckable(True)
        self.workspace_action.setChecked(True)
        self.workspace_action.setShortcut("Ctrl+G")
        self.workspace_action.triggered.connect(lambda _=False: self.toggle_workspace.emit())
        cmenu.addAction(self.workspace_action)
        cmenu.addAction("Settings…", self.settings.emit)
        self.customize_btn.clicked.connect(lambda: cmenu.exec(self.customize_btn.mapToGlobal(
            self.customize_btn.rect().bottomLeft())))
        self.files_btn = nav("Files", "files")
        self.files_menu = QMenu(self.files_btn)
        self.files_btn.clicked.connect(self._show_files)

        lay.addSpacing(10)
        rule = QFrame()
        rule.setObjectName("hairline")
        lay.addWidget(rule)
        lay.addSpacing(12)

        body = QWidget()
        body.setObjectName("cardBody")
        bl = QVBoxLayout(body)
        bl.setContentsMargins(0, 0, 0, 0)
        bl.setSpacing(0)
        active = QLabel("Active")
        active.setObjectName("secondary")
        active.setContentsMargins(0, 0, 0, 4)
        bl.addWidget(active)
        self.sessions_box = QVBoxLayout()
        self.sessions_box.setSpacing(0)
        bl.addLayout(self.sessions_box)
        bl.addSpacing(12)
        self.kb_toggle = QToolButton()
        self.kb_toggle.setObjectName("sectionToggle")
        self.kb_toggle.setCheckable(True)
        self.kb_toggle.setChecked(True)
        self.kb_toggle.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.kb_toggle.setIconSize(QSize(12, 12))
        self.kb_toggle.toggled.connect(self._kb_toggled)
        bl.addWidget(self.kb_toggle, 0, Qt.AlignmentFlag.AlignLeft)
        self.kb_wrap = QWidget()
        self.kb_box = QVBoxLayout(self.kb_wrap)
        self.kb_box.setContentsMargins(0, 2, 0, 0)
        self.kb_box.setSpacing(0)
        bl.addWidget(self.kb_wrap)
        bl.addStretch(1)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setWidget(body)
        lay.addWidget(scroll, 1)

        self.gear = QToolButton()
        self.gear.setObjectName("iconButton")
        self.gear.setIcon(icon("gear", t.hex("text_title")))
        self.gear.setIconSize(QSize(18, 18))
        self.gear.setToolTip("Settings and environment")
        self.gear.clicked.connect(self.settings.emit)
        lay.addWidget(self.gear, 0, Qt.AlignmentFlag.AlignLeft)
        self.kb_rows: list[ListRow] = []
        self._kb_toggled(True)

    # ------------------------------------------------------------------ state
    def refresh(self, current: str | None) -> None:
        t = self.theme
        if current and current != self.current:
            if self.current:
                self._history.append(self.current)
            self.current = current
        self.back_btn.setEnabled(bool(self._history))
        sessions = list(self.repo.list_sessions())
        title = next((r["title"] for r in sessions if r["id"] == self.current), "New session")
        self.project_name.set_full(title)
        self.session_menu.clear()
        clear_layout(self.sessions_box)
        self.session_rows = {}
        for r in sessions:
            sid = r["id"]
            act = self.session_menu.addAction(r["title"])
            act.triggered.connect(lambda _=False, s=sid: self.open_session.emit(s))
            row = ListRow(t, "hollow", "text_faint", r["title"])
            row.set_selected(sid == self.current)
            row.clicked.connect(lambda s=sid: self.open_session.emit(s))
            self.sessions_box.addWidget(row)
            self.session_rows[sid] = row
        self._apply_badge()

        clear_layout(self.kb_box)
        verified = self.repo.verified_results()
        self.kb_toggle.setText(f"Knowledge base · {len(verified)}")
        self.kb_rows = []
        for n in verified:
            value = prose_text(n.content) if n.type.value == "final" else value_text(n.display_result())
            row = ListRow(t, "check", "status_verified", f"{n.title}: {value[:80]}", f"{n.title}\n{value}")
            row.clicked.connect(lambda s=n.session_id, i=n.id: self.open_node.emit(s, i))
            self.kb_box.addWidget(row)
            self.kb_rows.append(row)
        if not verified:
            empty = QLabel("Verified results appear here")
            empty.setObjectName("secondary")
            empty.setContentsMargins(8, 4, 0, 0)
            self.kb_box.addWidget(empty)

    def set_running(self, sid: str | None, count: int = 0) -> None:
        """Badge on the running session: the number of tool calls so far (like the reference's "8")."""
        self._running = (sid, count) if sid else None
        self._apply_badge()

    def _apply_badge(self) -> None:
        for sid, row in self.session_rows.items():
            row.set_badge(self._running[1] if self._running and self._running[0] == sid else None)

    def _back(self) -> None:
        while self._history:
            sid = self._history.pop()
            if any(r["id"] == sid for r in self.repo.list_sessions()):
                self.current = None  # don't push the session we leave back onto the history
                self.open_session.emit(sid)
                return
        self.back_btn.setEnabled(False)

    def _kb_toggled(self, on: bool) -> None:
        self.kb_wrap.setVisible(on)
        self.kb_toggle.setIcon(icon("chevron_down" if on else "chevron_right", self.theme.hex("text_secondary")))

    def _show_files(self) -> None:
        m = self.files_menu
        m.clear()
        files = sorted(FILES_DIR.glob("*"))[:200] if FILES_DIR.exists() else []
        for p in files:
            a = m.addAction(icon("file", self.theme.hex("text_secondary")), p.name)
            a.setEnabled(False)
        if not files:
            a = m.addAction("No files yet")
            a.setEnabled(False)
        m.addSeparator()
        m.addAction(icon("plus", self.theme.hex("text")), "Add file…", self.add_file.emit)
        m.exec(self.files_btn.mapToGlobal(self.files_btn.rect().bottomLeft()))


__all__ = ["FILES_DIR", "SUBTITLE", "ListRow", "Sidebar"]
