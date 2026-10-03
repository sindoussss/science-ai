"""Left pane: New, sessions, Files and the Knowledge base."""
from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import QAbstractScrollArea, QLabel, QListWidget, QListWidgetItem, QPushButton, QVBoxLayout, QWidget

from sciai.store.repository import Repository

FILES_DIR = Path("~/.sciai/files").expanduser()


class Sidebar(QWidget):
    new_session = pyqtSignal()
    open_session = pyqtSignal(str)
    open_node = pyqtSignal(str, str)  # session_id, node_id

    def __init__(self, repo: Repository) -> None:
        super().__init__()
        self.setObjectName("sidebar")
        self.repo = repo
        lay = QVBoxLayout(self)
        lay.setContentsMargins(10, 12, 10, 12)
        lay.setSpacing(6)
        new = QPushButton("＋ New")
        new.clicked.connect(self.new_session.emit)
        lay.addWidget(new)

        lay.addWidget(self._title("Sessions"))
        self.sessions = QListWidget()
        self.sessions.itemClicked.connect(lambda it: self.open_session.emit(it.data(Qt.ItemDataRole.UserRole)))
        lay.addWidget(self.sessions, 3)

        lay.addWidget(self._title("Files"))
        self.files = QListWidget()
        lay.addWidget(self.files, 1)

        lay.addWidget(self._title("Knowledge base"))
        self.kb_count = QLabel()
        self.kb_count.setObjectName("muted")
        lay.addWidget(self.kb_count)
        self.kb = QListWidget()
        self.kb.itemClicked.connect(self._kb_clicked)
        lay.addWidget(self.kb, 2)

    def showEvent(self, event) -> None:  # noqa: N802, ANN001
        for lst in (self.sessions, self.files, self.kb):
            lst.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
            lst.setTextElideMode(Qt.TextElideMode.ElideRight)
            lst.setSizeAdjustPolicy(QAbstractScrollArea.SizeAdjustPolicy.AdjustIgnored)
        super().showEvent(event)

    @staticmethod
    def _title(text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setObjectName("sectionTitle")
        return lbl

    def refresh(self, current: str | None) -> None:
        self.sessions.clear()
        for row in self.repo.list_sessions():
            it = QListWidgetItem(row["title"])
            it.setData(Qt.ItemDataRole.UserRole, row["id"])
            self.sessions.addItem(it)
            if row["id"] == current:
                it.setSelected(True)
        self.files.clear()
        if FILES_DIR.exists():
            for p in sorted(FILES_DIR.glob("*"))[:200]:
                self.files.addItem(p.name)
        if self.files.count() == 0:
            empty = QListWidgetItem("No files yet")
            empty.setFlags(Qt.ItemFlag.NoItemFlags)
            self.files.addItem(empty)
        self.kb.clear()
        verified = self.repo.verified_results()
        self.kb_count.setText(f"{len(verified)} verified results, reused across sessions")
        for n in verified:
            value = n.content if n.type.value == "final" else n.display_result()
            it = QListWidgetItem(f"✓ {n.title}: {value[:80]}")
            it.setToolTip(n.content)
            it.setData(Qt.ItemDataRole.UserRole, (n.session_id, n.id))
            self.kb.addItem(it)

    def _kb_clicked(self, it: QListWidgetItem) -> None:
        sid, nid = it.data(Qt.ItemDataRole.UserRole)
        self.open_node.emit(sid, nid)
