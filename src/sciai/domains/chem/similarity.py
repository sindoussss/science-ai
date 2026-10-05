"""Similarity, substructure and clustering over a library, each with a second opinion.

Similarity ranking is checked by recomputing it with a different fingerprint. Two fingerprints
that disagree about which molecules are the nearest neighbours mean the ranking is an artefact
of the descriptor, not a fact about the molecules, so the check reports both orders and fails
rather than quietly picking one.

Every structure that comes back from a neighbour search is screened again. A library member
reached this way never passed through import, which is where the first screen runs, so this is
the second place Yeri's change 4 asks for.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable

MORGAN_RADIUS = 2
MORGAN_BITS = 2048
TOP_K_DEFAULT = 10
MAX_LIBRARY = 5000

# The check's fingerprint. A different algorithm, not the same one resized: Morgan is circular
# and atom-environment based, the RDKit fingerprint is path based.
CHECK_FP = "rdkit_path"


class LibraryError(ValueError):
    pass


@dataclass(frozen=True)
class Neighbour:
    index: int
    name: str
    smiles: str
    similarity: float


def morgan(mol: Any) -> Any:
    from rdkit.Chem import rdFingerprintGenerator

    gen = rdFingerprintGenerator.GetMorganGenerator(radius=MORGAN_RADIUS, fpSize=MORGAN_BITS)
    return gen.GetFingerprint(mol)


def path_fp(mol: Any) -> Any:
    from rdkit.Chem import rdFingerprintGenerator

    gen = rdFingerprintGenerator.GetRDKitFPGenerator(fpSize=MORGAN_BITS)
    return gen.GetFingerprint(mol)


def similarities(query: Any, members: list[Any], *, fp: Callable[[Any], Any]) -> list[float]:
    from rdkit import DataStructs

    q = fp(query)
    return [float(DataStructs.TanimotoSimilarity(q, fp(m))) for m in members]


def rank(query: Any, members: list[Any], names: list[str], smiles: list[str], *,
         top_k: int = TOP_K_DEFAULT, fp: Callable[[Any], Any] | None = None) -> list[Neighbour]:
    """The ``top_k`` nearest members, ties broken by name so the order is reproducible."""
    if len(members) > MAX_LIBRARY:
        raise LibraryError(f"{len(members)} members is past the {MAX_LIBRARY} limit for one search")
    scores = similarities(query, members, fp=fp or morgan)
    order = sorted(range(len(members)), key=lambda i: (-scores[i], names[i]))
    return [Neighbour(index=i, name=names[i], smiles=smiles[i], similarity=scores[i])
            for i in order[:top_k]]


def as_result(neighbours: list[Neighbour], *, query_smiles: str, fingerprint: str = "morgan") -> dict[str, Any]:
    return {"kind": "neighbours", "query": query_smiles, "fingerprint": fingerprint,
            "neighbours": [{"name": n.name, "smiles": n.smiles,
                            "similarity": round(n.similarity, 6)} for n in neighbours]}


def check_rank(query: Any, members: list[Any], names: list[str], smiles: list[str],
               reported: dict[str, Any], *, top_k: int = TOP_K_DEFAULT) -> dict[str, Any]:
    """Recompute the ranking with a path-based fingerprint and compare the two orders."""
    theirs = [n["name"] for n in (reported.get("neighbours") or [])]
    ours = [n.name for n in rank(query, members, names, smiles, top_k=top_k, fp=path_fp)]
    agree = theirs == ours
    overlap = len(set(theirs) & set(ours))
    return {"kind": "check", "outcome": "pass" if agree else "fail", "method": CHECK_FP,
            "reported_order": theirs, "recomputed_order": ours,
            "shared_members": overlap,
            "note": ("" if agree else
                     "Two fingerprints rank these differently, so the order is a property of "
                     "the descriptor rather than of the molecules.")}


# ---------------------------------------------------------------------------- substructure
def substructure(mol: Any, smarts: str) -> dict[str, Any]:
    from rdkit import Chem

    patt = Chem.MolFromSmarts(smarts)
    if patt is None:
        raise LibraryError(f"this is not a pattern the toolkit can read: {smarts[:60]!r}")
    matches = mol.GetSubstructMatches(patt)
    return {"kind": "substructure", "smarts": smarts, "count": len(matches),
            "atoms": [list(m) for m in matches]}


def check_substructure(mol: Any, reported: dict[str, Any]) -> dict[str, Any]:
    """Re-verify every reported match atom by atom, without asking the matcher again."""
    from rdkit import Chem

    smarts = reported.get("smarts") or ""
    patt = Chem.MolFromSmarts(smarts)
    if patt is None:
        return {"kind": "check", "outcome": "fail", "method": "atom walk",
                "disagreements": [{"problem": f"the stored pattern no longer compiles: {smarts!r}"}]}
    problems: list[dict[str, Any]] = []
    for match in reported.get("atoms") or []:
        if len(match) != patt.GetNumAtoms():
            problems.append({"match": match, "problem": "wrong number of atoms for the pattern"})
            continue
        for q_idx, m_idx in enumerate(match):
            q_atom, m_atom = patt.GetAtomWithIdx(q_idx), mol.GetAtomWithIdx(int(m_idx))
            if not q_atom.Match(m_atom):
                problems.append({"match": match, "atom": int(m_idx),
                                 "problem": "this atom does not satisfy the pattern"})
        for bond in patt.GetBonds():
            a, b = match[bond.GetBeginAtomIdx()], match[bond.GetEndAtomIdx()]
            if mol.GetBondBetweenAtoms(int(a), int(b)) is None:
                problems.append({"match": match, "problem": f"no bond between {a} and {b}"})
    count = reported.get("count")
    if count is not None and int(count) != len(reported.get("atoms") or []):
        problems.append({"problem": "the count and the match list disagree",
                         "count": int(count), "matches": len(reported.get("atoms") or [])})
    return {"kind": "check", "outcome": "fail" if problems else "pass", "method": "atom walk",
            "disagreements": problems, "verified_matches": len(reported.get("atoms") or [])}


# ---------------------------------------------------------------------------- clustering
def butina(members: list[Any], *, cutoff: float = 0.4) -> dict[str, Any]:
    """Butina clustering at a distance cutoff, over Morgan fingerprints."""
    from rdkit import DataStructs
    from rdkit.ML.Cluster import Butina

    fps = [morgan(m) for m in members]
    n = len(fps)
    distances: list[float] = []
    for i in range(1, n):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
        distances.extend(1.0 - s for s in sims)
    clusters = Butina.ClusterData(distances, n, cutoff, isDistData=True)
    assignment = [0] * n
    for cluster_id, members_of in enumerate(clusters):
        for i in members_of:
            assignment[i] = cluster_id
    return {"kind": "clusters", "cutoff": cutoff, "count": len(clusters),
            "assignment": assignment,
            "sizes": sorted((len(c) for c in clusters), reverse=True)}


def check_clusters(members: list[Any], reported: dict[str, Any]) -> dict[str, Any]:
    """Consistency, not a second algorithm: every member placed once, and the cutoff honoured.

    Two members in one cluster must be within the cutoff of that cluster's first member, which
    is what Butina's construction guarantees; a pair that is not means the assignment does not
    match the distances it claims to come from.
    """
    from rdkit import DataStructs

    assignment = list(reported.get("assignment") or [])
    problems: list[dict[str, Any]] = []
    if len(assignment) != len(members):
        problems.append({"problem": "the assignment does not cover every member",
                         "members": len(members), "assigned": len(assignment)})
        return {"kind": "check", "outcome": "fail", "method": "cutoff consistency",
                "disagreements": problems}
    cutoff = float(reported.get("cutoff", 0.4))
    fps = [morgan(m) for m in members]
    groups: dict[int, list[int]] = {}
    for i, c in enumerate(assignment):
        groups.setdefault(int(c), []).append(i)
    if sorted(len(g) for g in groups.values()) != sorted(reported.get("sizes") or [],):
        problems.append({"problem": "the reported sizes do not match the assignment"})
    for cluster_id, idx in groups.items():
        centroid = idx[0]
        for other in idx[1:]:
            distance = 1.0 - float(DataStructs.TanimotoSimilarity(fps[centroid], fps[other]))
            if distance > cutoff + 1e-9:
                problems.append({"cluster": cluster_id, "members": [centroid, other],
                                 "distance": round(distance, 4), "cutoff": cutoff,
                                 "problem": "further apart than the cutoff allows"})
    return {"kind": "check", "outcome": "fail" if problems else "pass",
            "method": "cutoff consistency", "disagreements": problems,
            "clusters": len(groups)}


def screen_members(smiles: Iterable[str]) -> tuple[list[str], list[dict[str, Any]]]:
    """Screen each member, returning the ones that pass and a record of the ones refused."""
    from sciai.domains.chem import restricted, standardize

    kept: list[str] = []
    refused: list[dict[str, Any]] = []
    for s in smiles:
        try:
            ident = standardize.screened(s)
        except standardize.MoleculeError as exc:
            refused.append({"input": s[:120], "reason": str(exc)})
            continue
        kept.append(ident.canonical_smiles)
    _ = restricted  # imported for the side-effect-free dependency, kept explicit
    return kept, refused


__all__ = ["CHECK_FP", "MAX_LIBRARY", "MORGAN_BITS", "MORGAN_RADIUS", "TOP_K_DEFAULT",
           "LibraryError", "Neighbour", "as_result", "butina", "check_clusters", "check_rank",
           "check_substructure", "morgan", "path_fp", "rank", "screen_members", "similarities",
           "substructure"]
