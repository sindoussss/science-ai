"""Confirm the formalized problem: its statement, its givens, and the assumption checklist.

Every assumption starts ticked. The dialog answers the controller with
``{"statement": text, "rejected": [unticked assumption texts]}``, or ``False`` on cancel.
"""
from __future__ import annotations

from typing import Any

from PyQt6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QLabel,
    QPlainTextEdit,
    QVBoxLayout,
    QWidget,
)

from sciai.graph.model import Node


def _given_text(name: str, q: dict[str, Any]) -> str:
    kind = f" ({q['kind'].replace('_', ' ')})" if q.get("kind") else ""
    return f"{name} = {q.get('value')} {q.get('unit', '')}".rstrip() + kind


class ConfirmProblemDialog(QDialog):
    def __init__(self, node: Node, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Confirm the problem")
        self.setMinimumWidth(460)
        inputs = node.tool_inputs or {}
        lay = QVBoxLayout(self)
        lay.setSpacing(8)
        intro = QLabel("This is how the question was formalized. Edit it if needed, then confirm.")
        intro.setWordWrap(True)
        lay.addWidget(intro)
        self.statement = QPlainTextEdit(node.content)
        self.statement.setMinimumHeight(80)
        lay.addWidget(self.statement)

        givens = inputs.get("givens") or {}
        if givens:
            cap = QLabel("Given values")
            cap.setObjectName("caption")
            lay.addWidget(cap)
            unsourced = set(inputs.get("unsourced_givens") or [])
            for name, q in givens.items():
                row = QLabel(_given_text(name, q) + ("  ⚠ not in the question" if name in unsourced else ""))
                row.setObjectName("mono")
                lay.addWidget(row)

        self.boxes: list[QCheckBox] = []
        items = inputs.get("checklist") or []
        if items:
            cap = QLabel("Assumptions (untick any that don't hold)")
            cap.setObjectName("caption")
            lay.addWidget(cap)
            for item in items:
                box = QCheckBox(item["text"] + ("" if item.get("source") == "default" else "  · suggested"))
                box.setProperty("assumption", item["text"])
                box.setChecked(True)
                lay.addWidget(box)
                self.boxes.append(box)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Confirm")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        lay.addWidget(buttons)

    def answer(self) -> dict[str, Any]:
        return {"statement": self.statement.toPlainText(),
                "rejected": [b.property("assumption") for b in self.boxes if not b.isChecked()]}
