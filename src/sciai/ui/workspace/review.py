"""Review tab: ladder stepper, fired risk rules, and evidence cards (no raw JSON by default)."""
from __future__ import annotations

import json
from typing import Any

from PyQt6.QtCore import QRectF, Qt
from PyQt6.QtGui import QBrush, QPainter, QPen
from PyQt6.QtWidgets import QFrame, QHBoxLayout, QLabel, QPushButton, QScrollArea, QVBoxLayout, QWidget

from sciai.graph.model import Evidence, Node
from sciai.ui.layout_util import FlowLayout, clear_layout
from sciai.ui.theme.theme import Theme

LADDER = (("retried", "Retry"), ("backtracked", "Backtrack"), ("escalated", "Escalate"))
METHOD_LABEL = {
    "symbolic_vs_numeric": "Symbolic vs numeric",
    "alt_algorithm": "Different algorithm",
    "known_value": "Known value",
    "units": "Unit check",
    "dimensional": "Dimensional analysis",
    "plausibility": "Physical range",
    "residual": "Residual",
    "kirchhoff": "Kirchhoff and Ohm's law",
    "power_balance": "Power balance",
    "series_parallel": "Series/parallel reduction",
    "inputs_verified": "All answer values verified",
    "reread": "Re-read from the canonical form",
    "recomputed": "Recomputed from the element table",
    "invariance": "Invariance under a round trip",
    "re_evaluated": "Filters re-evaluated",
    "alt_fingerprint": "Different fingerprint",
    "atom_by_atom": "Atom by atom",
    "partition": "Cluster partition",
    "re_sorted": "Re-sorted from the stored values",
    "quote_match": "Quotes re-read from the files",
}
RULE_LABEL = {"stakes": "Stakes", "surprise": "Surprise", "confidence": "Confidence", "step_type": "Step type",
              "units": "Units"}


def _short(v: Any, n: int = 18) -> str:
    s = str(v)
    try:
        f = complex(s.replace(" ", ""))
        s = f"{f.real:.10g}" if abs(f.imag) < 1e-15 else f"{f.real:.6g}{f.imag:+.6g}i"
    except ValueError:
        pass
    return s if len(s) <= n else s[: n - 1] + "…"


def _chem_agrees(d: dict[str, Any]) -> str:
    """What a passing chemistry check actually established, in the reader's terms."""
    if "verified_quotes" in d:
        n = d["verified_quotes"]
        return f"Every quoted passage ({n}) was found again at its recorded file and offset."
    if "verified_matches" in d:
        n = d["verified_matches"]
        return f"Every reported match ({n}) was re-walked atom by atom and bond by bond."
    if "checked" in d:
        rest = ", ".join(d.get("not_recomputed") or [])
        tail = f"; {rest} checked for invariance only" if rest else ""
        return (f"{len(d['checked'])} values recomputed from a pinned element table and a "
                f"bond walk{tail}.")
    if "shared_members" in d:
        return (f"A path-based fingerprint ranks the same {d['shared_members']} members in the "
                "same order.")
    if "clusters" in d:
        return (f"Every structure sits in exactly one of {d['clusters']} clusters, none wider "
                "than its cutoff allows.")
    if isinstance(d.get("recomputed"), dict):
        return "Re-evaluating the filters from the stored descriptor values gives the same verdicts."
    if isinstance(d.get("recomputed"), list):
        return "Re-sorting from the stored values gives the same order and the same scores."
    if "invariants" in d:
        return "The value is unchanged by a canonical round trip and by renumbering the atoms."
    return "An independent recomputation agrees."


def _chem_disagrees(first: dict[str, Any]) -> str:
    """The first disagreement a chemistry check found."""
    if first.get("problem") and "reported" not in first:
        where = first.get("descriptor") or first.get("rule") or first.get("field") or ""
        return f"{where}: {first['problem']}" if where else str(first["problem"])
    name = (first.get("descriptor") or first.get("field") or first.get("rule")
            or first.get("candidate") or first.get("invariant") or "the result")
    if "under_invariant" in first:
        return (f"{name} changes under {first.get('invariant', 'an invariant')}: "
                f"{_short(first.get('reported'))} became {_short(first['under_invariant'])}.")
    if "reread" in first:
        return (f"Re-reading gives a different {name}: {_short(first['reread'])} against the "
                f"stored {_short(first.get('stored'))}.")
    return (f"{name}: the result says {_short(first.get('reported'))}, an independent "
            f"recomputation gives {_short(first.get('recomputed'))}.")


def summarize(ev: Evidence) -> str:
    d = ev.detail or {}
    if ev.method == "inputs_verified":
        n = len(d.get("answer_nodes", []))
        return f"Every value in the answer comes from a verified node ({n})."
    if d.get("error"):
        return f"Check could not run: {d['error']}"
    if ev.method == "dimensional" and "derived" in d and ev.outcome == "pass":
        dims = d["derived"] or {}
        shown = " ".join(f"{k.strip('[]')}^{v:g}" if v != 1 else k.strip("[]") for k, v in dims.items())
        return f"Units are consistent: the formula gives {shown or 'a dimensionless value'}."
    if ev.method == "known_value" and "pint_value" in d:
        return (f"CODATA {d.get('codata')} via SciPy {d.get('scipy')} matches pint ({d['pint_value']}) "
                f"within {d.get('tolerance', 0):.0e}.")
    if "sources_deliver_W" in d:
        return (f"Sources deliver {_short(d['sources_deliver_W'])} W; resistors dissipate "
                f"{_short(d.get('resistors_dissipate_W'))} W.")
    if "equivalent_resistance" in d and ev.outcome == "pass":
        return f"Reduces to {_short(d['equivalent_resistance'])} ohm; every current agrees."
    if "max_error" in d and ev.outcome != "inconclusive":
        how = f"{d['method']} rerun" if d.get("method") else "closed form"
        return f"{how} differs by at most {_short(d['max_error'])}."
    if "reference_si" in d:
        return f"30-digit SI reference {_short(d['reference_si'])} vs result {_short(d.get('claimed_si'))}."
    if "shared_members" in d:
        # the similarity ranking check records two orders rather than a disagreement list
        return _chem_agrees(d) if ev.outcome == "pass" else str(d.get("note") or
                                                               "The two fingerprints disagree.")
    if "disagreements" in d:
        bad = d["disagreements"]
        return _chem_disagrees(bad[0]) if bad else _chem_agrees(d)
    if d.get("mismatches"):
        m = d["mismatches"][0]
        where = m.get("x", m.get("point", m.get("interval")))
        got = m.get("numeric", m.get("a"))
        claimed = m.get("claimed", m.get("b"))
        return (f"Disagrees at {_short(where, 24)}: independent method gives {_short(got)}, "
                f"result gives {_short(claimed)}.")
    if d.get("problems"):
        p = d["problems"][0]
        return p.get("reason") or f"Residual {_short(p.get('residual'))} at {p.get('solution')}."
    if "samples" in d:
        return f"Agrees at {d['samples']} independent sample points."
    if "reference" in d:
        return f"30-digit reference {_short(d['reference'])} vs result {_short(d.get('claimed'))}."
    if "numeric" in d:
        return f"Numeric value {_short(d['numeric'])} vs result {_short(d.get('claimed'))}."
    if "residual" in d:
        return f"Residual {d['residual']}, sign change: {'yes' if d.get('sign_change') else 'no'}."
    if "errors" in d:
        return f"Truncation error shrinks as expected ({', '.join(d['errors'])})."
    if "approach" in d:
        return f"Numerical approach converges to {_short(d.get('claimed', d['approach'][-1]))}."
    if "base_a" in d:
        return f"Both sides reduce to {d['base_a']}."
    if d.get("reason"):
        return str(d["reason"])
    return "Check completed."


class LadderStepper(QWidget):
    """Three steps (retry, backtrack, escalate); the current stage is highlighted."""

    def __init__(self, theme: Theme) -> None:
        super().__init__()
        self.theme = theme
        self.stage = "none"
        self.setMinimumHeight(54)

    def set_stage(self, stage: str) -> None:
        self.stage = stage
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802, ANN001
        t = self.theme
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        keys = [k for k, _ in LADDER]
        current = keys.index(self.stage) if self.stage in keys else -1
        w = self.width()
        xs = [w * (i + 0.5) / 3 for i in range(3)]
        y = 16
        for i in range(2):
            done = i < current
            p.setPen(QPen(t.c("text_secondary") if done else t.c("border"), 1.5))
            p.drawLine(int(xs[i] + 12), y, int(xs[i + 1] - 12), y)
        for i, (key, label) in enumerate(LADDER):
            if i < current:
                fill, ring, txt = t.c("panel"), t.c("text_secondary"), t.c("text_secondary")
            elif i == current:
                fill, ring, txt = t.c("text"), t.c("text"), t.c("panel")
            else:
                fill, ring, txt = t.c("panel"), t.c("border_strong"), t.c("text_faint")
            p.setPen(QPen(ring, 1.5))
            p.setBrush(QBrush(fill))
            p.drawEllipse(QRectF(xs[i] - 11, y - 11, 22, 22))
            p.setPen(txt)
            p.setFont(t.ui_font("size_small_px", bold=True))
            p.drawText(QRectF(xs[i] - 11, y - 11, 22, 22), Qt.AlignmentFlag.AlignCenter,
                       str(i + 1))
            p.setPen(t.c("text") if i == current else t.c("text_muted"))
            p.setFont(t.ui_font("size_small_px", bold=i == current))
            p.drawText(QRectF(xs[i] - 60, y + 15, 120, 18), Qt.AlignmentFlag.AlignCenter, label)
        p.end()


class EvidenceCard(QFrame):
    def __init__(self, ev: Evidence, theme: Theme, show_details: bool) -> None:
        super().__init__()
        self.setObjectName("evidenceCard")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(12, 10, 12, 10)
        lay.setSpacing(4)
        head = QHBoxLayout()
        title = QLabel(METHOD_LABEL.get(ev.method, ev.method.replace("_", " ").capitalize()))
        title.setStyleSheet("font-weight:600;")
        status = {"pass": "verified", "fail": "failed"}.get(ev.outcome, "inconclusive")
        badge = QLabel({"pass": "pass", "fail": "fail"}.get(ev.outcome, "inconclusive"))
        badge.setStyleSheet(theme.chip_css(status))
        head.addWidget(title)
        head.addStretch(1)
        head.addWidget(badge)
        lay.addLayout(head)
        tool = QLabel(ev.tool_name)
        tool.setObjectName("muted")
        tool.setStyleSheet(f"font-size:{theme.css('size_small_px')};")
        lay.addWidget(tool)
        result = QLabel(summarize(ev))
        result.setWordWrap(True)
        lay.addWidget(result)
        if show_details:
            detail = {k: v for k, v in (ev.detail or {}).items() if k not in ("kind", "outcome", "answer_nodes")}
            raw = QLabel(json.dumps({"inputs": ev.inputs, "detail": detail}, indent=1, default=str)[:1500])
            raw.setObjectName("mono")
            raw.setWordWrap(True)
            raw.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            lay.addWidget(raw)


class ReviewPanel(QScrollArea):
    def __init__(self, theme: Theme) -> None:
        super().__init__()
        self.theme = theme
        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        body = QWidget()
        self.lay = QVBoxLayout(body)
        self.lay.setContentsMargins(0, 8, 6, 12)
        self.lay.setSpacing(8)
        self.setWidget(body)
        self.show_details = False
        self._last: tuple[Node, list[Evidence]] | None = None

    def _section(self, text: str) -> None:
        lbl = QLabel(text)
        lbl.setObjectName("caption")
        lbl.setStyleSheet("padding:6px 0 0 0;")
        self.lay.addWidget(lbl)

    def clear(self) -> None:
        clear_layout(self.lay)

    def show_node(self, node: Node, evidence: list[Evidence]) -> None:
        self._last = (node, evidence)
        self.clear()
        t = self.theme
        self._section("Failure ladder")
        stepper = LadderStepper(t)
        stepper.set_stage(node.ladder_stage.value)
        self.lay.addWidget(stepper)
        if node.ladder_stage.value == "none":
            note = QLabel("Not on the ladder: no check on this node has failed.")
            note.setObjectName("muted")
            self.lay.addWidget(note)

        self._section("Risk rules")
        fired = node.risk.get("fired", [])
        if not fired:
            none = QLabel("None fired.")
            none.setObjectName("muted")
            self.lay.addWidget(none)
        if fired:
            host = QWidget()
            flow = FlowLayout(host, spacing=6)
            for f in fired:
                chip = QLabel(RULE_LABEL.get(f["rule"], f["rule"]))
                chip.setObjectName("ruleChip")
                chip.setToolTip(f["reason"])
                flow.addWidget(chip)
            self.lay.addWidget(host)
            for f in fired:
                why = QLabel(f"{RULE_LABEL.get(f['rule'], f['rule'])}: {f['reason']}")
                why.setObjectName("secondary")
                why.setWordWrap(True)
                self.lay.addWidget(why)
        if node.risk.get("pending_checks"):
            pend = QLabel("Waiting for a check: " + ", ".join(
                METHOD_LABEL.get(m, m) for m in node.risk["pending_checks"]))
            pend.setStyleSheet(f"color:{t.hex('warning')};")
            self.lay.addWidget(pend)
        if node.risk.get("unverifiable"):
            un = QLabel("No independent check exists for this tool.")
            un.setObjectName("muted")
            self.lay.addWidget(un)

        head = QHBoxLayout()
        head.setContentsMargins(0, 0, 0, 0)
        cap = QLabel("Evidence")
        cap.setObjectName("caption")
        cap.setStyleSheet("padding:6px 0 0 0;")
        head.addWidget(cap)
        head.addStretch(1)
        toggle = QPushButton("Hide details" if self.show_details else "Show details")
        toggle.setObjectName("link")
        toggle.clicked.connect(self._toggle_details)
        head.addWidget(toggle)
        wrap = QWidget()
        wrap.setLayout(head)
        self.lay.addWidget(wrap)
        if not evidence:
            none = QLabel("No checks have run on this node.")
            none.setObjectName("muted")
            self.lay.addWidget(none)
        for ev in evidence:
            self.lay.addWidget(EvidenceCard(ev, t, self.show_details))
        self.lay.addStretch(1)

    def _toggle_details(self) -> None:
        self.show_details = not self.show_details
        if self._last is not None:
            self.show_node(*self._last)
