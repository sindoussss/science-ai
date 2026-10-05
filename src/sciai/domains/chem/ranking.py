"""Ranking candidates by criteria the question names, and refusing criteria that aim at harm.

The ranking is arithmetic over values that are already stored on descriptor nodes: each
criterion is normalized across the candidates, signed by its direction, weighted, and summed.
Nothing here asks the model which molecule it likes. The weights come from the question, the
order is reproducible, and ties stay ties rather than being broken by position in the input.

Criteria are screened first. Maximizing toxicity or lethality is refused outright; minimizing
toxicity is ordinary ADMET work and passes, which is the distinction ``restricted`` draws and
the one the tests pin down.

A ranking is a hypothesis about which candidates are worth a look. It is not a prediction that
any of them works, and ``report`` says so.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

MAX_CANDIDATES = 2000
MAX_CRITERIA = 8
DIRECTIONS = ("higher_is_better", "lower_is_better")


class RankingError(ValueError):
    pass


class CriteriaRefused(ValueError):
    """The criteria optimize for harm; the request stops rather than being answered."""

    def __init__(self, rule: str, message: str) -> None:
        super().__init__(message)
        self.rule, self.message = rule, message


@dataclass(frozen=True)
class Ranked:
    name: str
    score: float
    terms: dict[str, float]


def check_criteria(criteria: list[dict[str, Any]]) -> None:
    """Raise ``CriteriaRefused`` when any criterion aims at harm."""
    from sciai.domains.chem import restricted

    for c in criteria:
        text = " ".join(str(c.get(k, "")) for k in ("descriptor", "direction", "note"))
        refusal = restricted.criteria_refusal(text)
        if refusal is not None:
            raise CriteriaRefused(refusal.rule, refusal.message)


def rank(candidates: list[dict[str, Any]], criteria: list[dict[str, Any]]) -> list[Ranked]:
    """Candidates ordered best first. ``candidates`` is [{name, values}]."""
    if not candidates:
        raise RankingError("no candidates to rank")
    if len(candidates) > MAX_CANDIDATES:
        raise RankingError(f"{len(candidates)} candidates is past the {MAX_CANDIDATES} limit")
    if not criteria:
        raise RankingError("no criteria given, so there is nothing to rank by")
    if len(criteria) > MAX_CRITERIA:
        raise RankingError(f"{len(criteria)} criteria is past the {MAX_CRITERIA} limit")
    check_criteria(criteria)

    for c in criteria:
        if c.get("direction") not in DIRECTIONS:
            raise RankingError(f"criterion {c.get('descriptor')!r} needs a direction, one of "
                               f"{', '.join(DIRECTIONS)}")
    names = [str(c["name"]) for c in candidates]
    if len(set(names)) != len(names):
        raise RankingError("two candidates share a name, so the ranking would be ambiguous")

    ranked: list[Ranked] = []
    spans = {str(c["descriptor"]): _span(candidates, str(c["descriptor"])) for c in criteria}
    for candidate in candidates:
        terms: dict[str, float] = {}
        for c in criteria:
            key = str(c["descriptor"])
            value = (candidate.get("values") or {}).get(key)
            if value is None:
                raise RankingError(f"{candidate['name']} has no value for {key}")
            low, high = spans[key]
            scaled = 0.5 if high == low else (float(value) - low) / (high - low)
            if c["direction"] == "lower_is_better":
                scaled = 1.0 - scaled
            terms[key] = round(scaled * float(c.get("weight", 1.0)), 9)
        ranked.append(Ranked(name=str(candidate["name"]),
                             score=round(sum(terms.values()), 9), terms=terms))
    return sorted(ranked, key=lambda r: (-r.score, r.name))


def _span(candidates: list[dict[str, Any]], key: str) -> tuple[float, float]:
    values = [float((c.get("values") or {})[key]) for c in candidates
              if (c.get("values") or {}).get(key) is not None]
    if not values:
        raise RankingError(f"no candidate carries a value for {key}")
    return min(values), max(values)


def as_result(ranked: list[Ranked], criteria: list[dict[str, Any]]) -> dict[str, Any]:
    return {"kind": "ranking",
            "criteria": [{"descriptor": str(c["descriptor"]), "direction": c["direction"],
                          "weight": float(c.get("weight", 1.0))} for c in criteria],
            "ranking": [{"name": r.name, "score": r.score, "terms": r.terms} for r in ranked]}


def report(ranked: list[Ranked]) -> str:
    head = ", ".join(f"{r.name} ({r.score:.3f})" for r in ranked[:3])
    return (f"Ranked {len(ranked)} candidates; the top of the list is {head}. The order follows "
            "the criteria given and the stored descriptor values, and it is a hypothesis about "
            "what is worth testing, not a prediction that any of these works.")


def check(candidates: list[dict[str, Any]], reported: dict[str, Any]) -> dict[str, Any]:
    """Re-sort from the stored values and require the same order and the same scores."""
    criteria = list(reported.get("criteria") or [])
    if not criteria:
        return {"kind": "check", "outcome": "fail", "method": "re-sorted from stored values",
                "disagreements": [{"problem": "the result records no criteria"}]}
    recomputed = rank(candidates, criteria)
    theirs = [(r["name"], float(r["score"])) for r in (reported.get("ranking") or [])]
    ours = [(r.name, r.score) for r in recomputed]
    disagreements: list[dict[str, Any]] = []
    if [n for n, _ in theirs] != [n for n, _ in ours]:
        disagreements.append({"problem": "different order",
                              "reported": [n for n, _ in theirs], "recomputed": [n for n, _ in ours]})
    for (name, score), (_, ours_score) in zip(theirs, ours):
        if abs(score - ours_score) > 1e-9:
            disagreements.append({"candidate": name, "reported": score, "recomputed": ours_score})
    return {"kind": "check", "outcome": "fail" if disagreements else "pass",
            "method": "re-sorted from stored values", "disagreements": disagreements,
            "recomputed": [{"name": n, "score": s} for n, s in ours]}


__all__ = ["DIRECTIONS", "MAX_CANDIDATES", "MAX_CRITERIA", "CriteriaRefused", "Ranked",
           "RankingError", "as_result", "check", "check_criteria", "rank", "report"]
