"""Declining a chemistry request, decided by capability and never by keyword.

The rule: a request is declined when no registered tool can serve the operation it asks for.
That is the whole test. ``OPERATION_TOOLS`` maps an operation to the tool that performs it and
``served`` asks the registry whether that tool exists, so the set of things this system refuses
is exactly the set of things it cannot do. A synthesis route, a reaction procedure or a dose is
declined because no tool produces one, not because a word was spotted in the question.

Keyword rules live in ``WORDING`` and do one job: pick which sentence the decline prints. They
cannot cause a decline. A question that says "synthesis" while asking for something a tool can
do (the melting point of a compound named in a synthesis paper, say) is served normally, and a
question no tool can serve is declined even when it contains no flagged word at all. The
false-positive tests in ``tests/unit/test_chem_decline.py`` pin that down in both directions.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

# Operation -> the tool that performs it. An operation absent here is served by nothing.
# Adding a capability means adding a tool and an entry; there is no other way to open one up.
OPERATION_TOOLS: dict[str, str] = {
    "identity": "chem.parse",
    "standardize": "chem.parse",
    "descriptors": "chem.descriptors",
    "logp": "chem.logp",
    "druglike": "chem.druglike",
    "similarity": "chem.similar",
    "substructure": "chem.substructure",
    "cluster": "chem.cluster",
    "literature": "chem.lit",
    "ranking": "chem.rank",
}

# The sentence a decline prints when no wording rule matches. One sentence, no lecture.
GENERIC = ("This system does computational screening only, so it cannot do that; synthesis, "
           "laboratory procedures and dosing are work for a qualified lab.")

# (rule id, pattern, sentence). Wording only: reaching this list means the request was already
# declined for want of a tool. Order matters only for which sentence reads best.
WORDING: tuple[tuple[str, re.Pattern[str], str], ...] = (
    ("route", re.compile(r"\b(synthes\w+|retrosynth\w+|reaction route|synthetic route|"
                         r"how (?:to|do i|would i) (?:make|prepare|produce)|precursor)\b", re.I),
     "This system compares and screens molecules but does not produce synthesis routes; "
     "a route is work for a qualified lab."),
    ("procedure", re.compile(r"\b(procedure|protocol|reaction conditions|reflux|workup|"
                             r"purif\w+|scale[- ]?up|yield)\b", re.I),
     "This system works on structures and their computed properties, not laboratory "
     "procedures; the bench work belongs to a qualified lab."),
    ("compounding", re.compile(r"\b(compound(?:ing)?\s+(?:instructions|procedure)|formulat\w+|"
                               r"excipient|tablet press|capsule fill)\b", re.I),
     "This system screens candidate structures and does not give compounding or formulation "
     "instructions; those belong to a licensed pharmacy or lab."),
    ("dosing", re.compile(r"\b(dose|dosage|dosing|mg/kg|administer\w*|how much .{0,40}take|"
                          r"posolog\w+)\b", re.I),
     "This system predicts molecular properties and cannot give a dose; dosing is a decision "
     "for a clinician."),
)


@dataclass(frozen=True)
class Decline:
    """A refused request, as it is shown and as it is recorded on the decline node."""

    operation: str          # the operation that nothing serves
    wording: str            # which wording rule chose the sentence ("generic" if none did)
    message: str            # the one sentence the user reads
    asked: str              # what was asked, stored for the record
    served: tuple[str, ...] = field(default=())   # the operations that were available instead

    def node_content(self) -> str:
        return (f"Declined: no tool serves {self.operation!r}.\n"
                f"Asked: {self.asked}\n"
                f"Wording rule: {self.wording}\n"
                f"Available operations: {', '.join(self.served) or 'none'}")


def served(operation: str, *, tool_names: frozenset[str] | None = None) -> bool:
    """True when a registered tool performs ``operation``."""
    tool = OPERATION_TOOLS.get((operation or "").strip().lower())
    if tool is None:
        return False
    return tool in (tool_names if tool_names is not None else _registered())


def served_operations(*, tool_names: frozenset[str] | None = None) -> tuple[str, ...]:
    names = tool_names if tool_names is not None else _registered()
    return tuple(sorted({op for op, tool in OPERATION_TOOLS.items() if tool in names}))


def wording_for(asked: str) -> tuple[str, str]:
    """(rule id, sentence) for a declined request. Never decides *whether* to decline."""
    for rule, pattern, sentence in WORDING:
        if pattern.search(asked or ""):
            return rule, sentence
    return "generic", GENERIC


def decline_for(operation: str, asked: str, *,
                tool_names: frozenset[str] | None = None) -> Decline | None:
    """A ``Decline`` when nothing serves ``operation``, else None.

    ``asked`` only steers the wording, so a served operation is never declined over its text.
    """
    if served(operation, tool_names=tool_names):
        return None
    rule, sentence = wording_for(asked)
    return Decline(operation=(operation or "").strip().lower() or "unspecified", wording=rule,
                   message=sentence, asked=asked or "",
                   served=served_operations(tool_names=tool_names))


def _registered() -> frozenset[str]:
    from sciai.tools.registry import all_tools

    return frozenset(t.name for t in all_tools())


__all__ = ["Decline", "GENERIC", "OPERATION_TOOLS", "WORDING", "decline_for", "served",
           "served_operations", "wording_for"]
