"""Left card: serif wordmark, New, then collapsible icon lists for sessions, files and the knowledge base."""
from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import QSize, Qt, pyqtSignal
from PyQt6.QtGui import QFont, QMouseEvent, QPainter, QPaintEvent
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from sciai.store.repository import Repository
from sciai.ui.chat_bar import PRODUCT_NAME
from sciai.ui.icons import icon, paint_icon
from sciai.ui.mathtext import prose_text, value_text
from sciai.ui.theme.theme import Theme

FILES_DIR = Path("~/.sciai/files").expanduser()


class SectionHeader(QWidget):
    """11px uppercase, letter-spaced caption with a chevron; click to collapse.

    Painted directly: QSS has no letter-spacing or text-transform, and the global
    font rule would override a font set on a QLabel."""

    toggled = pyqtSignal(bool)

    def __init__(self, text: str, theme: Theme) -> None:
        super().__init__()
        self.theme = theme
        self.text = text
        self.expanded = True
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFixedHeight(28)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def set_text(self, text: str) -> None:
        self.text = text
        self.update()

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton:
            self.expanded = not self.expanded
            self.update()
            self.toggled.emit(self.expanded)

    def paintEvent(self, event: QPaintEvent) -> None:  # noqa: N802
        t = self.theme
        p = QPainter(self)
        font = t.ui_font("size_caption_px", bold=True)
        font.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, 0.8)
        font.setCapitalization(QFont.Capitalization.AllUppercase)
        p.setFont(font)
        p.setPen(t.c("text_secondary"))
        p.drawText(self.rect().adjusted(8, 0, -24, 0), Qt.AlignmentFlag.AlignVCenter, self.text)
        paint_icon(p, "chevron_down" if self.expanded else "chevron_right", t.c("text_faint"),
                   self.width() - 22, (self.height() - 16) / 2)
        p.end()


class FitList(QListWidget):
    """A plain list that is exactly as tall as its rows, so sections stack with no dead space."""

    def __init__(self) -> None:
        super().__init__()
        self.setFrameShape(QListWidget.Shape.NoFrame)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setTextElideMode(Qt.TextElideMode.ElideRight)
        self.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.setIconSize(QSize(16, 16))
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def fit(self) -> None:
        h = sum(self.sizeHintForRow(i) for i in range(self.count()))
        self.setFixedHeight(h + 2 * self.frameWidth())


class Section(QWidget):
    def __init__(self, title: str, theme: Theme) -> None:
        super().__init__()
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        self.header = SectionHeader(title, theme)
        self.list = FitList()
        lay.addWidget(self.header)
        lay.addWidget(self.list)
        self.header.toggled.connect(self.list.setVisible)


class Sidebar(QWidget):
    new_session = pyqtSignal()
    open_session = pyqtSignal(str)
    open_node = pyqtSignal(str, str)  # session_id, node_id
    collapsed_changed = pyqtSignal(bool)

    def __init__(self, repo: Repository, theme: Theme) -> None:
        super().__init__()
        self.setObjectName("cardBody")
        self.repo = repo
        self.theme = theme
        self.collapsed = False
        t = self.theme
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        self.full = QWidget()
        self.full.setObjectName("cardBody")
        self.rail = self._build_rail()
        self.rail.hide()
        outer.addWidget(self.full)
        outer.addWidget(self.rail)

        lay = QVBoxLayout(self.full)
        lay.setContentsMargins(12, 18, 12, 12)
        lay.setSpacing(0)
        brand_row = QHBoxLayout()
        brand_row.setContentsMargins(0, 0, 0, 0)
        brand = QLabel(PRODUCT_NAME)
        brand.setObjectName("brand")
        brand.setContentsMargins(8, 0, 0, 0)
        collapse = QToolButton()
        collapse.setObjectName("iconButton")
        collapse.setIcon(icon("chevron_left", t.hex("text_secondary")))
        collapse.setToolTip("Collapse sidebar")
        collapse.clicked.connect(lambda: self.set_collapsed(True))
        brand_row.addWidget(brand, 1)
        brand_row.addWidget(collapse, 0, Qt.AlignmentFlag.AlignTop)
        lay.addLayout(brand_row)
        tag = QLabel("Local research workspace")
        tag.setObjectName("secondary")
        tag.setContentsMargins(8, 0, 0, 0)
        lay.addWidget(tag)
        lay.addSpacing(14)
        new = QPushButton("New")
        new.setObjectName("nav")
        new.setIcon(icon("plus", t.hex("text")))
        new.setIconSize(QSize(16, 16))
        new.clicked.connect(self.new_session.emit)
        lay.addWidget(new)
        lay.addSpacing(8)

        body = QWidget()
        body.setObjectName("cardBody")
        bl = QVBoxLayout(body)
        bl.setContentsMargins(0, 0, 0, 0)
        bl.setSpacing(6)
        self.sessions_sec = Section("Sessions", t)
        self.files_sec = Section("Files", t)
        self.kb_sec = Section("Knowledge base", t)
        for sec in (self.sessions_sec, self.files_sec, self.kb_sec):
            bl.addWidget(sec)
        bl.addStretch(1)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setWidget(body)
        lay.addWidget(scroll, 1)

        self.sessions, self.files, self.kb = self.sessions_sec.list, self.files_sec.list, self.kb_sec.list
        self.kb_caption = self.kb_sec.header
        self.sessions.itemClicked.connect(lambda it: self.open_session.emit(it.data(Qt.ItemDataRole.UserRole)))
        self.kb.itemClicked.connect(self._kb_clicked)

    def _build_rail(self) -> QWidget:
        t = self.theme
        rail = QWidget()
        rail.setObjectName("cardBody")
        rl = QVBoxLayout(rail)
        rl.setContentsMargins(8, 18, 8, 12)
        rl.setSpacing(8)
        for name, tip, slot in (("chevron_right", "Expand sidebar", lambda: self.set_collapsed(False)),
                                ("plus", "New session", self.new_session.emit)):
            b = QToolButton()
            b.setObjectName("iconButton")
            b.setIcon(icon(name, t.hex("text")))
            b.setToolTip(tip)
            b.clicked.connect(slot)
            rl.addWidget(b, 0, Qt.AlignmentFlag.AlignHCenter)
        rl.addStretch(1)
        return rail

    def set_collapsed(self, collapsed: bool) -> None:
        self.collapsed = collapsed
        self.full.setVisible(not collapsed)
        self.rail.setVisible(collapsed)
        self.collapsed_changed.emit(collapsed)

    def refresh(self, current: str | None) -> None:
        t = self.theme
        chat_icon = icon("chat", t.hex("text_secondary"))
        self.sessions.clear()
        for row in self.repo.list_sessions():
            it = QListWidgetItem(chat_icon, row["title"])
            it.setToolTip(row["title"])
            it.setData(Qt.ItemDataRole.UserRole, row["id"])
            self.sessions.addItem(it)
            if row["id"] == current:
                it.setSelected(True)
        self.files.clear()
        file_icon = icon("file", t.hex("text_secondary"))
        if FILES_DIR.exists():
            for p in sorted(FILES_DIR.glob("*"))[:200]:
                self.files.addItem(QListWidgetItem(file_icon, p.name))
        if self.files.count() == 0:
            empty = QListWidgetItem(icon("empty", t.hex("text_faint")), "No files yet")
            empty.setFlags(Qt.ItemFlag.NoItemFlags)
            empty.setForeground(t.c("text_faint"))
            self.files.addItem(empty)
        self.kb.clear()
        verified = self.repo.verified_results()
        self.kb_caption.set_text(f"Knowledge base · {len(verified)}")
        check = icon("check", t.hex("status_verified"))
        for n in verified:
            value = prose_text(n.content) if n.type.value == "final" else value_text(n.display_result())
            it = QListWidgetItem(check, f"{n.title}: {value[:80]}")
            it.setToolTip(f"{n.title}\n{value}")
            it.setData(Qt.ItemDataRole.UserRole, (n.session_id, n.id))
            self.kb.addItem(it)
        for lst in (self.sessions, self.files, self.kb):
            lst.fit()

    def _kb_clicked(self, it: QListWidgetItem) -> None:
        sid, nid = it.data(Qt.ItemDataRole.UserRole)
        self.open_node.emit(sid, nid)
