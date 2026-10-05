"""The Molecule tab of the node view.

Shows the structure a node is, or the one it was computed from: the depiction, the identity, a
descriptor table with the tool behind each row, the drug-likeness verdict with its failing terms
named, and the hypothesis banner.

The banner sits above the scroll area rather than inside it, so it cannot be scrolled out of
view. That is the one piece of this tab that is not decoration: everything else on screen is a
number, and the banner is what says the numbers describe a candidate nobody has tested.

The badge in the banner says whether this node's required checks have run and passed. It never
contradicts the hypothesis sentence beside it: a chemistry node is a hypothesis whether or not
its checks pass, and what passing checks buy is reuse, not belief.
"""
from __future__ import annotations

from typing import Any

from PyQt6.QtCore import QSize, Qt
from PyQt6.QtGui import QFontMetricsF
from PyQt6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from sciai.domains.chem import descriptors as desc
from sciai.graph.model import Evidence, Node
from sciai.ui.chat.inline import Column, TableCard
from sciai.ui.icons import icon
from sciai.ui.mol_image import MolView, export
from sciai.ui.theme.theme import Theme

HYPOTHESIS_TEXT = ("Hypothesis. Computed values for a candidate, not a tested result: nothing "
                   "here says the molecule works.")

# What each chemistry check establishes, for the banner's one-line summary.
CHECK_SAID = {
    "reread": "identity re-read from the canonical form",
    "recomputed": "masses and counts recomputed from a pinned element table",
    "invariance": "value unchanged under a round trip and a renumbering",
    "re_evaluated": "filters re-evaluated from the stored descriptors",
    "alt_fingerprint": "ranking recomputed with a different fingerprint",
    "atom_by_atom": "matches re-walked atom by atom",
    "partition": "clusters checked against their cutoff",
    "re_sorted": "ranking re-sorted from the stored values",
    "quote_match": "quotes re-read from the cited files",
}

# Which tool a descriptor came from, for the table's last column.
SOURCE_TOOL = {name: "chem.descriptors" for name in desc.NAMES}
SOURCE_TOOL["logp"] = "chem.logp"


def molecule_of(node: Node | None) -> dict[str, Any] | None:
    """The molecule a node is, or the one it was computed from; None for every other node.

    Read from the node's own result, never by walking the graph: every chem result that is
    about one structure carries that structure's canonical SMILES, which is also what makes
    the depiction reproducible from the stored node alone.
    """
    if node is None:
        return None
    result = node.result or {}
    if result.get("kind") == "molecule":
        return result
    smiles = result.get("canonical_smiles") or (
        result.get("query") if result.get("kind") == "neighbours" else "")
    if smiles:
        # a descriptors, logp, druglike, substructure or similarity result
        return {"kind": "molecule", "canonical_smiles": smiles,
                "inchikey": result.get("inchikey", ""), "formula": result.get("formula", ""),
                "derived": True}
    return None


def descriptor_rows(result: dict[str, Any]) -> list[list[str]]:
    """One row per descriptor: name, value, unit, and the tool that produced it."""
    values = result.get("values") or {}
    units = result.get("units") or desc.UNITS
    rows = []
    for name in (*desc.NAMES, "logp"):
        if name not in values:
            continue
        rows.append([name.replace("_", " "), _shown(name, float(values[name])),
                     units.get(name, ""), SOURCE_TOOL.get(name, "")])
    return rows


# Counts are integers; the two masses keep four decimals, because a monoisotopic mass printed
# as "180" is the one number on this table that is useless without them.
COUNTS = ("heavy_atoms", "rings", "aromatic_rings", "hbd", "hba", "rotatable_bonds",
          "formal_charge")


def _shown(name: str, value: float) -> str:
    if name in COUNTS:
        return str(int(round(value)))
    if name in ("mw", "exact_mass"):
        return f"{value:.4f}"
    return f"{value:.4g}"


def check_badge(evidence: list[Evidence]) -> tuple[str, str]:
    """The banner's badge and the line under it, from a node's evidence.

    Chemistry checks are all required, so one failure is the whole story and is said first.
    """
    if not evidence:
        return "unchecked", "No check has run on this node yet."
    failed = [e for e in evidence if e.outcome == "fail"]
    if failed:
        said = CHECK_SAID.get(failed[0].method, failed[0].method.replace("_", " "))
        return "check failed", f"A required check disagreed: {said}."
    passed = [e for e in evidence if e.outcome == "pass"]
    if not passed:
        return "inconclusive", "No check reached a verdict; the values here are unconfirmed."
    said = [CHECK_SAID.get(e.method, e.method.replace("_", " ")) for e in passed]
    seen = list(dict.fromkeys(said))
    return "checks passed", _sentence(seen)


def _sentence(parts: list[str]) -> str:
    # Semicolons, no trailing "and": several of these clauses contain an "and" of their own.
    body = "; ".join(parts)
    return body[0].upper() + body[1:] + "."


def verdict_text(result: dict[str, Any]) -> str:
    verdicts = result.get("verdicts") or {}
    if not verdicts:
        return ""
    parts = []
    for rule, body in verdicts.items():
        name = "Lipinski" if rule == "lipinski" else "Veber"
        failing = body.get("failing") or []
        if body.get("passed") and not failing:
            parts.append(f"{name}: passes")
        elif body.get("passed"):
            parts.append(f"{name}: passes with one miss ({', '.join(failing)})")
        else:
            parts.append(f"{name}: fails on {', '.join(failing)}")
    return "; ".join(parts)


class MoleculeTab(QWidget):
    def __init__(self, theme: Theme) -> None:
        super().__init__()
        self.theme = theme
        self.molecule: dict[str, Any] | None = None
        self.name = "molecule"
        self._rows = 0
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # Pinned, outside the scroll area, so it cannot scroll away.
        self.banner = QWidget()
        self.banner.setObjectName("molBanner")
        strip = QVBoxLayout(self.banner)
        strip.setContentsMargins(16, 8, 16, 8)
        strip.setSpacing(4)
        self.hypothesis = QLabel(HYPOTHESIS_TEXT)
        self.hypothesis.setObjectName("molHypothesis")
        self.hypothesis.setWordWrap(True)
        strip.addWidget(self.hypothesis)
        checks = QHBoxLayout()
        checks.setSpacing(8)
        self.badge = QLabel()
        self.badge.setStyleSheet(theme.chip_css("hypothesis"))
        self.checks = QLabel()
        self.checks.setObjectName("molChecks")
        self.checks.setWordWrap(True)
        checks.addWidget(self.badge, 0, Qt.AlignmentFlag.AlignTop)
        checks.addWidget(self.checks, 1)
        strip.addLayout(checks)
        outer.addWidget(self.banner)

        body = QWidget()
        lay = QVBoxLayout(body)
        lay.setContentsMargins(16, 12, 16, 16)
        lay.setSpacing(6)

        row = QHBoxLayout()
        row.setSpacing(6)
        self.export_png = QPushButton("Export PNG")
        self.export_svg = QPushButton("Export SVG")
        for button, fmt in ((self.export_png, "png"), (self.export_svg, "svg")):
            button.setObjectName("outline")
            button.setIcon(icon("download", theme.hex("text_secondary")))
            button.setIconSize(QSize(14, 14))
            button.clicked.connect(lambda _=False, f=fmt: self._export(f))
            row.addWidget(button)
        row.addStretch(1)
        lay.addLayout(row)

        self.view = MolView(theme)
        self.view.setMinimumHeight(220)
        lay.addWidget(self.view)

        self.identity = QLabel()
        self.identity.setObjectName("molIdentity")
        self.identity.setWordWrap(True)
        self.smiles = QLabel()
        self.smiles.setObjectName("secondary")
        self.smiles.setWordWrap(True)
        self.standardized = QLabel()
        self.standardized.setObjectName("secondary")
        self.standardized.setWordWrap(True)
        for label in (self.identity, self.smiles, self.standardized):
            label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            lay.addWidget(label)

        lay.addSpacing(8)
        self.table_host = QVBoxLayout()
        self.table_host.setContentsMargins(0, 0, 0, 0)
        lay.addLayout(self.table_host)
        self.table: TableCard | None = None

        self.verdict = QLabel()
        self.verdict.setObjectName("secondary")
        self.verdict.setWordWrap(True)
        lay.addWidget(self.verdict)
        lay.addStretch(1)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setWidget(body)
        outer.addWidget(scroll)
        self.set_checks([])
        self.show_molecule(None)

    def set_checks(self, evidence: list[Evidence]) -> None:
        """Show what this node's checks established. Called with the panel's own evidence list."""
        badge, line = check_badge(evidence)
        self.badge.setText(badge)
        self.checks.setText(line)

    def show_molecule(self, molecule: dict[str, Any] | None,
                      result: dict[str, Any] | None = None, name: str = "molecule") -> None:
        self.molecule, self.name = molecule, name
        if self.table is not None:
            # removeWidget first: setParent(None) alone leaves the layout holding an item that
            # points at the widget, and the next layout pass calls minimumSizeHint() on a
            # C++ object deleteLater has already freed, which segfaults rather than raising.
            self.table_host.removeWidget(self.table)
            self.table.setParent(None)
            self.table.deleteLater()
            self.table = None
        self._rows = 0
        if molecule is None:
            self.view.set_structure(None)
            self.identity.setText("No structure")
            self.smiles.setText("Select a molecule, or a step that read one, to see it drawn.")
            self.standardized.hide()
            self.verdict.hide()
            for button in (self.export_png, self.export_svg):
                button.setEnabled(False)
            return

        smiles = molecule.get("canonical_smiles", "")
        self.view.set_structure(smiles)
        formula = molecule.get("formula", "")
        key = molecule.get("inchikey", "")
        atoms = molecule.get("atoms")
        bits = [b for b in (formula, f"{atoms} atoms" if atoms else "", key) if b]
        self.identity.setText("  ·  ".join(bits))
        self.smiles.setText(smiles)
        steps = molecule.get("standardized") or []
        self.standardized.setText("Standardized: " + ", ".join(steps))
        self.standardized.setVisible(bool(steps))
        for button in (self.export_png, self.export_svg):
            button.setEnabled(bool(smiles))

        rows = descriptor_rows(result or {})
        self._rows = len(rows)
        if rows:
            # The descriptor name is the column a reader scans, so its minimum is measured from
            # the names themselves: it takes the spare width when the card is wide and never
            # elides when it is narrow. "from" is the only column allowed to give width up,
            # because it repeats one string on every row; the unit is data, so it stays whole.
            metrics = QFontMetricsF(self.theme.ui_font("size_ui_px"))
            name_w = max(metrics.horizontalAdvance(r[0]) for r in rows) + 2 * TableCard.PAD
            self.table = TableCard(self.theme,
                                   [Column("descriptor", "text", name_w, flex=True),
                                    Column("value", "code", 40),
                                    Column("unit", "text", 24),
                                    Column("from", "text", 48, shrink=True)], rows)
            self.table_host.addWidget(self.table)
        text = verdict_text(result or {})
        self.verdict.setText(text)
        self.verdict.setVisible(bool(text))

    def descriptor_count(self) -> int:
        """How many descriptor rows the table is showing (0 when there is no table)."""
        return self._rows if self.table is not None else 0

    def _export(self, fmt: str) -> None:
        if self.molecule is None:
            return
        filt = "PNG image (*.png)" if fmt == "png" else "SVG image (*.svg)"
        path, _ = QFileDialog.getSaveFileName(self, f"Export {fmt.upper()}",
                                              f"{self.name}.{fmt}", filt)
        if path:
            if not path.lower().endswith(f".{fmt}"):
                path += f".{fmt}"
            export(self.molecule.get("canonical_smiles", ""), path)


__all__ = ["CHECK_SAID", "HYPOTHESIS_TEXT", "MoleculeTab", "check_badge",
           "descriptor_rows", "molecule_of", "verdict_text"]
