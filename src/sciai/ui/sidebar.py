"""Left card: brand, New, then plain icon lists for sessions, files and the knowledge base."""
from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import QAbstractItemView, QLabel, QListWidget, QListWidgetItem, QPushButton, QVBoxLayout, QWidget

from sciai.store.repository import Repository
from sciai.ui.chat_bar import PRODUCT_NAME

FILES_DIR = Path("~/.sciai/files").expanduser()


def _plain_list() -> QListWidget:
    lst = QListWidget()
    lst.setFrameShape(QListWidget.Shape.NoFrame)
    lst.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    lst.setTextElideMode(Qt.TextElideMode.ElideRight)
    lst.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
    lst.setUniformItemSizes(True)
    return lst


class Sidebar(QWidget):
    new_session = pyqtSignal()
    open_session = pyqtSignal(str)
    open_node = pyqtSignal(str, str)  # session_id, node_id

    def __init__(self, repo: Repository) -> None:
        super().__init__()
        self.setObjectName("cardBody")
        self.repo = repo
        lay = QVBoxLayout(self)
        lay.setContentsMargins(12, 16, 12, 12)
        lay.setSpacing(2)
        brand = QLabel(PRODUCT_NAME)
        brand.setObjectName("brand")
        lay.addWidget(brand)
        tag = QLabel("Local research workspace")
        tag.setObjectName("muted")
        lay.addWidget(tag)
        lay.addSpacing(10)
        new = QPushButton("＋  New")
        new.setObjectName("nav")
        new.clicked.connect(self.new_session.emit)
        lay.addWidget(new)

        lay.addWidget(self._caption("Sessions"))
        self.sessions = _plain_list()
        self.sessions.itemClicked.connect(lambda it: self.open_session.emit(it.data(Qt.ItemDataRole.UserRole)))
        lay.addWidget(self.sessions, 3)

        lay.addWidget(self._caption("Files"))
        self.files = _plain_list()
        lay.addWidget(self.files, 1)

        self.kb_caption = self._caption("Knowledge base")
        lay.addWidget(self.kb_caption)
        self.kb = _plain_list()
        self.kb.itemClicked.connect(self._kb_clicked)
        lay.addWidget(self.kb, 2)

    @staticmethod
    def _caption(text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setObjectName("caption")
        return lbl

    def refresh(self, current: str | None) -> None:
        self.sessions.clear()
        for row in self.repo.list_sessions():
            it = QListWidgetItem(f"○  {row['title']}")
            it.setToolTip(row["title"])
            it.setData(Qt.ItemDataRole.UserRole, row["id"])
            self.sessions.addItem(it)
            if row["id"] == current:
                it.setSelected(True)
        self.files.clear()
        if FILES_DIR.exists():
            for p in sorted(FILES_DIR.glob("*"))[:200]:
                self.files.addItem(QListWidgetItem(f"▤  {p.name}"))
        if self.files.count() == 0:
            empty = QListWidgetItem("No files yet")
            empty.setFlags(Qt.ItemFlag.NoItemFlags)
            empty.setForeground(self.palette().placeholderText())
            self.files.addItem(empty)
        self.kb.clear()
        verified = self.repo.verified_results()
        self.kb_caption.setText(f"Knowledge base · {len(verified)} verified")
        for n in verified:
            value = n.content if n.type.value == "final" else n.display_result()
            it = QListWidgetItem(f"✓  {n.title}: {value[:80]}")
            it.setToolTip(f"{n.title}\n{value}")
            it.setData(Qt.ItemDataRole.UserRole, (n.session_id, n.id))
            self.kb.addItem(it)

    def _kb_clicked(self, it: QListWidgetItem) -> None:
        sid, nid = it.data(Qt.ItemDataRole.UserRole)
        self.open_node.emit(sid, nid)
