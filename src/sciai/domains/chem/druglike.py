"""Drug-likeness filters, and the lipophilicity estimate they lean on.

The filters are arithmetic over descriptors that were already computed and checked, so the
check here re-evaluates the rules from the stored descriptor node rather than from any text the
model produced. The thresholds are constants in this file; nothing reads them from a prompt.

Lipophilicity is a different kind of number and is labelled as one. RDKit ships a single logP
model (Crippen's atom-contribution method) and there is no second method here that would be a
genuinely independent opinion rather than a copy of the same table, so, following Yeri's rule
for this case, the result carries ``method_count: 1`` and reads as a single-method estimate. It
is checked for the invariances a correct value must hold, and it never becomes anything firmer
than an estimate.

A filter verdict is not a prediction that a molecule is a drug. Lipinski's and Veber's rules
describe where oral drugs have historically clustered; plenty of approved drugs fail them.
``report`` says that in the text rather than leaving a bare pass or fail to be over-read.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

LIPINSKI = {"mw": ("<=", 500.0), "logp": ("<=", 5.0), "hbd": ("<=", 5.0), "hba": ("<=", 10.0)}
VEBER = {"rotatable_bonds": ("<=", 10.0), "tpsa": ("<=", 140.0)}
RULES = {"lipinski": LIPINSKI, "veber": VEBER}
LIPINSKI_ALLOWED_FAILURES = 1   # the rule of five is conventionally read as "at most one miss"

LOGP_ABS_TOL = 1e-6


@dataclass(frozen=True)
class Verdict:
    rule: str
    passed: bool
    failing: tuple[str, ...]
    detail: dict[str, Any]


def evaluate(values: dict[str, float], *, rule: str) -> Verdict:
    """Apply one rule set to descriptor values. Missing inputs are a failure to evaluate."""
    thresholds = RULES[rule]
    missing = sorted(k for k in thresholds if values.get(k) is None)
    if missing:
        raise KeyError(f"{rule} needs {', '.join(missing)}, which the descriptors do not carry")
    failing = tuple(k for k, (_, limit) in thresholds.items() if float(values[k]) > limit)
    allowed = LIPINSKI_ALLOWED_FAILURES if rule == "lipinski" else 0
    return Verdict(rule=rule, passed=len(failing) <= allowed, failing=failing,
                   detail={k: {"value": float(values[k]), "limit": limit, "op": op}
                           for k, (op, limit) in thresholds.items()})


def evaluate_all(values: dict[str, float]) -> dict[str, Verdict]:
    return {rule: evaluate(values, rule=rule) for rule in RULES}


def as_result(verdicts: dict[str, Verdict]) -> dict[str, Any]:
    return {"kind": "druglike",
            "verdicts": {r: {"passed": v.passed, "failing": list(v.failing), "detail": v.detail}
                         for r, v in verdicts.items()},
            "allowed_failures": {"lipinski": LIPINSKI_ALLOWED_FAILURES, "veber": 0}}


def report(verdicts: dict[str, Verdict]) -> str:
    parts = []
    for rule, v in verdicts.items():
        name = "Lipinski" if rule == "lipinski" else "Veber"
        if v.passed and not v.failing:
            parts.append(f"{name}: passes")
        elif v.passed:
            parts.append(f"{name}: passes with one miss ({', '.join(v.failing)})")
        else:
            parts.append(f"{name}: fails on {', '.join(v.failing)}")
    return ("; ".join(parts) + ". These rules describe where oral drugs have historically "
            "clustered, so a miss is a flag to look at, not a verdict on the molecule.")


def check(values: dict[str, float], reported: dict[str, Any]) -> dict[str, Any]:
    """Re-evaluate every rule from the descriptor values and compare with what was reported."""
    recomputed = evaluate_all(values)
    disagreements = []
    for rule, v in recomputed.items():
        theirs = (reported.get("verdicts") or {}).get(rule)
        if theirs is None:
            disagreements.append({"rule": rule, "problem": "missing from the result"})
            continue
        if bool(theirs.get("passed")) != v.passed or sorted(theirs.get("failing") or ()) != sorted(v.failing):
            disagreements.append({"rule": rule, "reported": theirs.get("passed"),
                                  "recomputed": v.passed,
                                  "reported_failing": sorted(theirs.get("failing") or ()),
                                  "recomputed_failing": sorted(v.failing)})
    return {"kind": "check", "outcome": "fail" if disagreements else "pass",
            "method": "re-evaluated from the descriptor node",
            "disagreements": disagreements,
            "recomputed": {r: {"passed": v.passed, "failing": list(v.failing)}
                           for r, v in recomputed.items()}}


# ---------------------------------------------------------------------------- lipophilicity
def logp(mol: Any) -> dict[str, Any]:
    """Crippen's logP, labelled as the single-method estimate it is."""
    from rdkit.Chem import Crippen

    value = float(Crippen.MolLogP(mol))
    return {"kind": "estimate", "property": "logp", "value": value, "unit": "",
            "method": "Crippen atom contributions", "method_count": 1,
            "label": "single-method estimate",
            "caveat": ("One method only: RDKit ships a single logP model, so this value has no "
                       "independent second opinion behind it and stays an estimate.")}


def check_logp(mol: Any, reported: dict[str, Any]) -> dict[str, Any]:
    """The invariances a correct value holds: a canonical round trip and a renumbering.

    This cannot audit Crippen's table, and does not claim to. It catches the wrong molecule
    reaching the tool, a stale cache, and a value that drifted between computing and storing.
    """
    from rdkit import Chem
    from rdkit.Chem import Crippen

    value = reported.get("value")
    if value is None:
        return {"kind": "check", "outcome": "fail", "method": "invariance",
                "disagreements": [{"problem": "the result carries no value"}]}
    round_trip = Chem.MolFromSmiles(Chem.MolToSmiles(mol))
    renumbered = Chem.RenumberAtoms(mol, list(range(mol.GetNumAtoms()))[::-1])
    others = {"round_trip": float(Crippen.MolLogP(round_trip)),
              "renumbered": float(Crippen.MolLogP(renumbered))}
    disagreements = [{"invariant": k, "reported": float(value), "under_invariant": v}
                     for k, v in others.items() if abs(float(value) - v) > LOGP_ABS_TOL]
    return {"kind": "check", "outcome": "fail" if disagreements else "pass",
            "method": "invariance", "invariants": others, "disagreements": disagreements,
            "note": ("Invariance only. A single-method estimate cannot be checked against a "
                     "second method, because there is not one.")}


__all__ = ["LIPINSKI", "LIPINSKI_ALLOWED_FAILURES", "RULES", "VEBER", "Verdict", "as_result",
           "check", "check_logp", "evaluate", "evaluate_all", "logp", "report"]
