"""One line of text per chemistry result, for the graph digest, the answer and the node list.

Every renderer here reads the stored result dict and nothing else, so what the model is shown
and what a reused node shows later are the same text, in the same order, whatever order SQLite
handed the JSON back in.
"""
from __future__ import annotations

from typing import Any

KINDS = ("molecule", "descriptors", "estimate", "druglike", "neighbours", "substructure",
         "clusters", "ranking", "literature")


def molecule_text(r: dict[str, Any]) -> str:
    bits = [str(r.get("formula") or ""), f"{r.get('atoms')} atoms" if r.get("atoms") else "",
            str(r.get("inchikey") or "")]
    head = ", ".join(b for b in bits if b)
    steps = r.get("standardized") or []
    return head + (f"; standardized: {', '.join(steps)}" if steps else "")


def descriptors_text(r: dict[str, Any]) -> str:
    v = r.get("values") or {}

    def n(name: str, digits: int = 2) -> str:
        return "?" if v.get(name) is None else f"{float(v[name]):.{digits}f}"

    def i(name: str) -> str:
        return "?" if v.get(name) is None else str(int(round(float(v[name]))))

    out = (f"MW {n('mw')}, exact mass {n('exact_mass', 4)}, {i('heavy_atoms')} heavy atoms, "
           f"{i('rings')} rings, TPSA {n('tpsa')} A^2, HBD {i('hbd')}, HBA {i('hba')}, "
           f"{i('rotatable_bonds')} rotatable bonds")
    return out + (f", logP {n('logp')}" if v.get("logp") is not None else "")


def estimate_text(r: dict[str, Any]) -> str:
    name = str(r.get("property") or "estimate")
    value = r.get("value")
    shown = "?" if value is None else f"{float(value):.2f}"
    label = r.get("label")
    return f"{name} {shown}" + (f" ({label})" if label else "")


def druglike_text(r: dict[str, Any]) -> str:
    parts = []
    for rule, body in (r.get("verdicts") or {}).items():
        name = "Lipinski" if rule == "lipinski" else "Veber"
        failing = body.get("failing") or []
        if body.get("passed") and not failing:
            parts.append(f"{name}: passes")
        elif body.get("passed"):
            parts.append(f"{name}: passes with one miss ({', '.join(failing)})")
        else:
            parts.append(f"{name}: fails on {', '.join(failing)}")
    return "; ".join(parts) or "no rule was applied"


def neighbours_text(r: dict[str, Any]) -> str:
    hits = r.get("neighbours") or []
    if not hits:
        return "no neighbour passed the screen"
    head = ", ".join(f"{h['name']} {float(h['similarity']):.3f}" for h in hits[:3])
    refused = len(r.get("refused_members") or [])
    return (f"{len(hits)} neighbours by {r.get('fingerprint', 'morgan')} Tanimoto: {head}"
            + (f"; {refused} members refused by the screen" if refused else ""))


def substructure_text(r: dict[str, Any]) -> str:
    count = int(r.get("count", 0))
    return f"{count} match{'es' if count != 1 else ''} for {r.get('smarts', '')}"


def clusters_text(r: dict[str, Any]) -> str:
    sizes = r.get("sizes") or []
    refused = len(r.get("refused_members") or [])
    return (f"{int(r.get('count', 0))} clusters at cutoff {r.get('cutoff')}"
            + (f"; largest holds {sizes[0]}" if sizes else "")
            + (f"; {refused} members refused by the screen" if refused else ""))


def ranking_text(r: dict[str, Any]) -> str:
    ranked = r.get("ranking") or []
    if not ranked:
        return "nothing to rank"
    head = ", ".join(f"{x['name']} ({float(x['score']):.3f})" for x in ranked[:3])
    return f"{len(ranked)} candidates ranked; top: {head}"


def literature_text(r: dict[str, Any]) -> str:
    dropped = len(r.get("dropped_passages") or [])
    return (f"{len(r.get('hits') or [])} passages from {len(r.get('files') or [])} files"
            + (f"; {dropped} dropped by the classifier" if dropped else "")
            + (f"; {r['excluded_sections']} procedure sections were never indexed"
               if r.get("excluded_sections") else ""))


TEXT = {"molecule": molecule_text, "descriptors": descriptors_text, "estimate": estimate_text,
        "druglike": druglike_text, "neighbours": neighbours_text,
        "substructure": substructure_text, "clusters": clusters_text, "ranking": ranking_text,
        "literature": literature_text}


def text_for(result: dict[str, Any]) -> str:
    return TEXT[result["kind"]](result)


__all__ = ["KINDS", "TEXT", "text_for"]
