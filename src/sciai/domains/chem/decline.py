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
    # "how much ... take" spans whatever the question put in between, which may be a whole
    # SMILES string: 40 characters was not enough for one, and the dose question then printed
    # the generic sentence instead of the one about dosing.
    ("dosing", re.compile(r"\b(dose|dosage|dosing|mg/kg|administer\w*|"
                          r"how much [^.?!]{0,160}?take|posolog\w+)\b", re.I),
     "This system predicts molecular properties and cannot give a dose; dosing is a decision "
     "for a clinician."),
)


# ---------------------------------------------------------------------------- asked in code
# Deciding before any model call whether a request can be served at all. The rule does not
# change: these patterns only name the operation a question *asks for*, and ``served`` still
# decides. An operation named here that is absent from OPERATION_TOOLS is declined because
# nothing performs it, and the day a tool is registered for one, the same code serves it.
#
# Two lists, in this order. SERVED_REQUESTS recognises a question asking for something a tool
# does; one match and nothing is pre-declined, which is what keeps "the logP of the product of
# this synthesis" a logP question. UNSERVED_REQUESTS recognises a question *requesting* an
# operation nothing performs, and only in a requesting construction: "how do I synthesize X",
# not the word "synthesis" in passing, and "what dose", not "dose-limiting toxicity".
SERVED_REQUESTS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("identity", re.compile(r"\b(canonical smiles|inchi\w*|formula|standardi[sz]\w+|"
                           r"what (?:molecule|compound|structure) is)\b", re.I)),
    ("descriptors", re.compile(r"\b(molecular weight|molar mass|tpsa|polar surface|rotatable|"
                              r"h-?bond|heavy atoms?|descriptors?|monoisotopic|"
                              r"sp3 fraction|rings?)\b", re.I)),
    ("logp", re.compile(r"\b(logp|clogp|lipophilic\w*)\b", re.I)),
    ("druglike", re.compile(r"\b(drug-?like\w*|lipinski|veber|rule of five)\b", re.I)),
    ("similarity", re.compile(r"\b(similar\w*|tanimoto|nearest|neighbou?rs?|compare)\b", re.I)),
    ("substructure", re.compile(r"\b(substructure|smarts)\b", re.I)),
    ("cluster", re.compile(r"\bcluster\w*\b", re.I)),
    ("literature", re.compile(r"\b(literature|papers?|corpus|passages?)\b", re.I)),
    ("ranking", re.compile(r"\b(rank\w*|shortlist|best candidates?)\b", re.I)),
)

UNSERVED_REQUESTS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("synthesis_route", re.compile(
        r"how (?:do|would|can|should|to)\s*(?:i|we|you)?\s*"
        r"(?:synthesi[sz]e|synthesise|make|prepare|produce)\b|"
        r"\b(?:synthesi[sz]e|prepare|produce)\s+[^.?!]{0,40}?\bfrom\b|"
        r"\b(?:synthetic|synthesis|reaction|retrosynthetic)\s+route\b|"
        r"\bretrosynthe\w+\b|\bsteps?\s+to\s+(?:synthesi[sz]e|make|prepare)\b", re.I)),
    ("procedure", re.compile(
        r"\breaction conditions\b|\b(?:procedure|protocol)\s+(?:for|to)\b|"
        r"how (?:do|would|should)\s*(?:i|we)?\s*(?:run|carry out|perform)\b", re.I)),
    ("compounding", re.compile(
        r"\bcompounding\s+(?:instructions|procedure)\b|"
        r"how (?:do|would|should)\s*(?:i|we)?\s*(?:formulate|compound)\b", re.I)),
    ("dosing", re.compile(
        r"\bwhat(?:'s| is)?\s+(?:the\s+)?(?:dose|dosage)\b|"
        r"\b(?:dose|dosage)\s+(?:of|for)\b|\bdosing regimen\b|\bmg/kg\b|"
        r"\bhow much\b[^.?!]{0,60}?\b(?:should|do|would)\b[^.?!]{0,30}?"
        r"\b(?:take|takes|be given|receive|administer|ingest)\b|"
        r"\bhow (?:much|many)\s+(?:mg|milligrams?|grams?|tablets?|capsules?)\b", re.I)),
)


def requested_operation(question: str) -> str | None:
    """The operation a question asks for, when it plainly asks for one nothing performs.

    None means "do not pre-decline": either the question asks for something a tool does, or it
    is not asking for one of these operations at all. Being unsure is always None, because a
    false decline refuses work the system can do, and the model's own ``operation`` is checked
    by the same capability rule one call later.
    """
    text = question or ""
    if any(pattern.search(text) for _, pattern in SERVED_REQUESTS):
        return None
    return next((op for op, pattern in UNSERVED_REQUESTS if pattern.search(text)), None)


def decline_for_question(question: str, *,
                         tool_names: frozenset[str] | None = None) -> Decline | None:
    """A ``Decline`` for a question that asks for an operation nothing serves, before any model
    call is made. Capability still decides: the operation goes through ``decline_for``, so an
    operation some tool performs is never declined here, whatever the wording."""
    operation = requested_operation(question)
    if operation is None:
        return None
    return decline_for(operation, question, tool_names=tool_names)


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


__all__ = ["Decline", "GENERIC", "OPERATION_TOOLS", "SERVED_REQUESTS", "UNSERVED_REQUESTS",
           "WORDING", "decline_for", "decline_for_question", "requested_operation", "served",
           "served_operations", "wording_for"]
