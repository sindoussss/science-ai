"""Chemistry solvers and checkers.

Every solver here is deterministic and every one has a checker that reaches its answer by a
different route: masses and counts recomputed from a pinned element table, drug-likeness
re-evaluated from the stored descriptor node, a similarity ranking recomputed with a path-based
fingerprint instead of a circular one, substructure matches re-verified atom by atom.

Two things this module does not have, and cannot be given from outside it: a tool that plans a
reaction or writes a procedure, and a way to mark a result verified. The first is absent from
the registry, which is what ``domains.chem.decline`` consults when it refuses a request. The
second is enforced by the store, the graph engine and the verifier, each independently.

Every structure that enters through one of these tools passes ``standardize.screened``, so the
restricted screen runs on the way in and again on every neighbour that comes back out of a
library search.
"""
from __future__ import annotations

from typing import Any

from sciai.domains.chem import descriptors as desc
from sciai.domains.chem import druglike, ranking, similarity, standardize
from sciai.graph.model import Domain
from sciai.tools.registry import ToolSpec, register, schema

STRUCTURE = {"type": "string", "maxLength": 1000}
FORMAT = {"enum": ["smiles", "molblock", "inchi"]}
SMARTS = {"type": "string", "maxLength": 400}
LIBRARY = {
    "type": "object",
    "properties": {
        "name": {"type": "string", "maxLength": 200},
        "members": {"type": "array", "maxItems": similarity.MAX_LIBRARY,
                    "items": {"type": "object",
                              "properties": {"name": {"type": "string", "maxLength": 200},
                                             "structure": STRUCTURE},
                              "required": ["name", "structure"], "additionalProperties": False}},
    },
    "required": ["members"],
    "additionalProperties": False,
}
CANDIDATES = {"type": "array", "maxItems": ranking.MAX_CANDIDATES,
              "items": {"type": "object",
                        "properties": {"name": {"type": "string", "maxLength": 200},
                                       "values": {"type": "object",
                                                  "additionalProperties": {"type": "number"}}},
                        "required": ["name", "values"], "additionalProperties": False}}
CRITERIA = {"type": "array", "maxItems": ranking.MAX_CRITERIA,
            "items": {"type": "object",
                      "properties": {"descriptor": {"type": "string", "maxLength": 60},
                                     "direction": {"enum": list(ranking.DIRECTIONS)},
                                     "weight": {"type": "number", "minimum": 0, "maximum": 10},
                                     "note": {"type": "string", "maxLength": 200}},
                      "required": ["descriptor", "direction"], "additionalProperties": False}}


def _result(value: dict[str, Any], **meta: Any) -> dict[str, Any]:
    # No confidence on a chem result: it is a hypothesis whatever the checks say, and a number
    # here would read as a probability that the molecule works.
    return {"result": value, "confidence": None, "meta": meta}


def _check(outcome: str, **detail: Any) -> dict[str, Any]:
    return {"result": {"kind": "check", "outcome": outcome, **detail}, "confidence": None,
            "meta": {}}


def _mol(structure: str, fmt: str = "smiles") -> Any:
    """A screened, standardized molecule from a structure string."""
    ident = standardize.screened(structure, fmt=fmt)
    return standardize.parse(ident.canonical_smiles), ident


# ================================================================== chem.parse
def parse_fn(args: dict[str, Any]) -> dict[str, Any]:
    ident = standardize.screened(args["structure"], fmt=args.get("format", "smiles"))
    return _result(ident.as_result(), summary=ident.summary())


register(ToolSpec(
    name="chem.parse", domain=Domain.CHEM, kind="solver",
    description="standardize one structure and give its canonical SMILES, InChIKey and formula",
    schema=schema({"structure": STRUCTURE, "format": FORMAT}, ["structure"]),
    fn=parse_fn, always_check=True,
))


def check_parse_fn(args: dict[str, Any]) -> dict[str, Any]:
    """Read the canonical SMILES back from scratch and require the same identity."""
    stored = args["molecule"]
    again = standardize.read_twice(stored["canonical_smiles"])
    fields = ("canonical_smiles", "inchikey", "formula", "atoms", "bonds", "charge")
    disagreements = [{"field": f, "stored": stored.get(f), "reread": getattr(again, f)}
                     for f in fields if stored.get(f) != getattr(again, f)]
    return _check("fail" if disagreements else "pass", method="re-read from the canonical form",
                  disagreements=disagreements)


register(ToolSpec(
    name="chem.check_parse", domain=Domain.CHEM, kind="checker",
    description="re-read a stored structure and require the same InChIKey, formula and counts",
    schema=schema({"molecule": {"type": "object"}}, ["molecule"]), fn=check_parse_fn,
))


# ================================================================== chem.descriptors
def descriptors_fn(args: dict[str, Any]) -> dict[str, Any]:
    mol, ident = _mol(args["structure"], args.get("format", "smiles"))
    values = desc.compute(mol)
    return _result({**values.as_result(), "inchikey": ident.inchikey,
                    "canonical_smiles": ident.canonical_smiles},
                   summary=values.summary())


register(ToolSpec(
    name="chem.descriptors", domain=Domain.CHEM, kind="solver",
    description="molecular weight, exact mass, heavy atoms, rings, HBD, HBA, TPSA, "
                "rotatable bonds and sp3 fraction of one structure",
    schema=schema({"structure": STRUCTURE, "format": FORMAT}, ["structure"]),
    fn=descriptors_fn, always_check=True,
))


def check_descriptors_fn(args: dict[str, Any]) -> dict[str, Any]:
    stored = args["descriptors"]
    mol = standardize.parse(stored["canonical_smiles"])
    return {"result": desc.check(mol, stored.get("values") or {}), "confidence": None, "meta": {}}


register(ToolSpec(
    name="chem.check_descriptors", domain=Domain.CHEM, kind="checker",
    description="recompute masses and counts from a pinned element table and check TPSA's "
                "invariance under a round trip and a renumbering",
    schema=schema({"descriptors": {"type": "object"}}, ["descriptors"]), fn=check_descriptors_fn,
))


# ================================================================== chem.logp
def logp_fn(args: dict[str, Any]) -> dict[str, Any]:
    mol, ident = _mol(args["structure"], args.get("format", "smiles"))
    value = druglike.logp(mol)
    return _result({**value, "canonical_smiles": ident.canonical_smiles,
                    "inchikey": ident.inchikey},
                   summary=f"logP {value['value']:.2f} ({value['label']})")


register(ToolSpec(
    name="chem.logp", domain=Domain.CHEM, kind="solver",
    description="Crippen logP of one structure, as a single-method estimate",
    schema=schema({"structure": STRUCTURE, "format": FORMAT}, ["structure"]),
    fn=logp_fn, always_check=True,
))


def check_logp_fn(args: dict[str, Any]) -> dict[str, Any]:
    stored = args["estimate"]
    mol = standardize.parse(stored["canonical_smiles"])
    return {"result": druglike.check_logp(mol, stored), "confidence": None, "meta": {}}


register(ToolSpec(
    name="chem.check_logp", domain=Domain.CHEM, kind="checker",
    description="check a logP estimate for invariance under a canonical round trip and a "
                "renumbering (there is no second method to compare against)",
    schema=schema({"estimate": {"type": "object"}}, ["estimate"]), fn=check_logp_fn,
))


# ================================================================== chem.druglike
def druglike_fn(args: dict[str, Any]) -> dict[str, Any]:
    values = dict((args["descriptors"].get("values") or {}))
    if args.get("logp") is not None:
        values["logp"] = float(args["logp"])
    verdicts = druglike.evaluate_all(values)
    return _result({**druglike.as_result(verdicts), "values": values,
                    "canonical_smiles": args["descriptors"].get("canonical_smiles", "")},
                   summary=druglike.report(verdicts))


register(ToolSpec(
    name="chem.druglike", domain=Domain.CHEM, kind="solver",
    description="apply the Lipinski and Veber filters to a descriptor result, naming what fails",
    schema=schema({"descriptors": {"type": "object"}, "logp": {"type": "number"}},
                  ["descriptors"]),
    fn=druglike_fn, always_check=True,
))


def check_druglike_fn(args: dict[str, Any]) -> dict[str, Any]:
    stored = args["verdict"]
    return {"result": druglike.check(stored.get("values") or {}, stored), "confidence": None,
            "meta": {}}


register(ToolSpec(
    name="chem.check_druglike", domain=Domain.CHEM, kind="checker",
    description="re-evaluate the filters from the stored descriptor values, not from the text",
    schema=schema({"verdict": {"type": "object"}}, ["verdict"]), fn=check_druglike_fn,
))


# ================================================================== chem.similar
def _library(args: dict[str, Any]) -> tuple[list[Any], list[str], list[str], list[dict[str, Any]]]:
    """Screen every member on the way in; refused members are reported, never silently dropped."""
    mols, names, smiles, refused = [], [], [], []
    for member in args["library"]["members"]:
        try:
            ident = standardize.screened(member["structure"])
        except standardize.MoleculeError as exc:
            refused.append({"name": member["name"], "reason": str(exc)})
            continue
        mols.append(standardize.parse(ident.canonical_smiles))
        names.append(member["name"])
        smiles.append(ident.canonical_smiles)
    return mols, names, smiles, refused


def similar_fn(args: dict[str, Any]) -> dict[str, Any]:
    mol, ident = _mol(args["structure"])
    members, names, smiles, refused = _library(args)
    if not members:
        raise similarity.LibraryError("no library member passed the structure screen")
    top_k = int(args.get("top_k", similarity.TOP_K_DEFAULT))
    neighbours = similarity.rank(mol, members, names, smiles, top_k=top_k)
    value = similarity.as_result(neighbours, query_smiles=ident.canonical_smiles)
    best = neighbours[0]
    return _result({**value, "refused_members": refused, "top_k": top_k},
                   summary=(f"Nearest of {len(members)} members: {best.name} at Tanimoto "
                            f"{best.similarity:.3f}"
                            + (f"; {len(refused)} members were refused by the screen"
                               if refused else "")))


register(ToolSpec(
    name="chem.similar", domain=Domain.CHEM, kind="solver",
    description="the nearest members of a library to one structure, by Tanimoto on Morgan "
                "fingerprints",
    schema=schema({"structure": STRUCTURE, "library": LIBRARY,
                   "top_k": {"type": "integer", "minimum": 1, "maximum": 50}},
                  ["structure", "library"]),
    fn=similar_fn, always_check=True, entity_arg="library",
    describe_entity=lambda lib: f"{len(lib.get('members') or [])} structures",
))


def check_similar_fn(args: dict[str, Any]) -> dict[str, Any]:
    stored = args["neighbours"]
    mol = standardize.parse(stored["query"])
    members = [standardize.parse(n["smiles"]) for n in stored["neighbours"]]
    names = [n["name"] for n in stored["neighbours"]]
    smiles = [n["smiles"] for n in stored["neighbours"]]
    result = similarity.check_rank(mol, members, names, smiles, stored, top_k=len(names))
    return {"result": result, "confidence": None, "meta": {}}


register(ToolSpec(
    name="chem.check_similar", domain=Domain.CHEM, kind="checker",
    description="recompute the ranking with a path-based fingerprint and compare the orders",
    schema=schema({"neighbours": {"type": "object"}}, ["neighbours"]), fn=check_similar_fn,
))


# ================================================================== chem.substructure
def substructure_fn(args: dict[str, Any]) -> dict[str, Any]:
    mol, ident = _mol(args["structure"])
    value = similarity.substructure(mol, args["smarts"])
    return _result({**value, "canonical_smiles": ident.canonical_smiles},
                   summary=f"{value['count']} match{'es' if value['count'] != 1 else ''} "
                           f"for {args['smarts']}")


register(ToolSpec(
    name="chem.substructure", domain=Domain.CHEM, kind="solver",
    description="count and locate the matches of a SMARTS pattern in one structure",
    schema=schema({"structure": STRUCTURE, "smarts": SMARTS}, ["structure", "smarts"]),
    fn=substructure_fn, always_check=True,
))


def check_substructure_fn(args: dict[str, Any]) -> dict[str, Any]:
    stored = args["matches"]
    mol = standardize.parse(stored["canonical_smiles"])
    return {"result": similarity.check_substructure(mol, stored), "confidence": None, "meta": {}}


register(ToolSpec(
    name="chem.check_substructure", domain=Domain.CHEM, kind="checker",
    description="re-verify every reported match atom by atom and bond by bond",
    schema=schema({"matches": {"type": "object"}}, ["matches"]), fn=check_substructure_fn,
))


# ================================================================== chem.cluster
def cluster_fn(args: dict[str, Any]) -> dict[str, Any]:
    members, names, smiles, refused = _library(args)
    if not members:
        raise similarity.LibraryError("no library member passed the structure screen")
    value = similarity.butina(members, cutoff=float(args.get("cutoff", 0.4)))
    return _result({**value, "names": names, "smiles": smiles, "refused_members": refused},
                   summary=(f"{value['count']} clusters over {len(members)} structures at "
                            f"cutoff {value['cutoff']}; largest holds {value['sizes'][0]}"))


register(ToolSpec(
    name="chem.cluster", domain=Domain.CHEM, kind="solver",
    description="Butina clusters of a library at a Tanimoto distance cutoff",
    schema=schema({"library": LIBRARY,
                   "cutoff": {"type": "number", "minimum": 0.05, "maximum": 0.95}},
                  ["library"]),
    fn=cluster_fn, always_check=True, entity_arg="library",
    describe_entity=lambda lib: f"{len(lib.get('members') or [])} structures",
))


def check_cluster_fn(args: dict[str, Any]) -> dict[str, Any]:
    stored = args["clusters"]
    members = [standardize.parse(s) for s in stored["smiles"]]
    return {"result": similarity.check_clusters(members, stored), "confidence": None, "meta": {}}


register(ToolSpec(
    name="chem.check_cluster", domain=Domain.CHEM, kind="checker",
    description="check that every member is placed once and that no cluster is wider than its "
                "cutoff allows",
    schema=schema({"clusters": {"type": "object"}}, ["clusters"]), fn=check_cluster_fn,
))


# ================================================================== chem.rank
def rank_fn(args: dict[str, Any]) -> dict[str, Any]:
    candidates, criteria = args["candidates"], args["criteria"]
    ranked = ranking.rank(candidates, criteria)
    return _result(ranking.as_result(ranked, criteria), summary=ranking.report(ranked))


register(ToolSpec(
    name="chem.rank", domain=Domain.CHEM, kind="solver",
    description="order candidates by the question's criteria over stored descriptor values",
    schema=schema({"candidates": CANDIDATES, "criteria": CRITERIA}, ["candidates", "criteria"]),
    fn=rank_fn, always_check=True,
))


def check_rank_fn(args: dict[str, Any]) -> dict[str, Any]:
    stored = args["ranking"]
    candidates = args["candidates"]
    return {"result": ranking.check(candidates, stored), "confidence": None, "meta": {}}


register(ToolSpec(
    name="chem.check_rank", domain=Domain.CHEM, kind="checker",
    description="re-sort the ranking from the stored values and require the same order",
    schema=schema({"ranking": {"type": "object"}, "candidates": CANDIDATES},
                  ["ranking", "candidates"]), fn=check_rank_fn,
))


__all__ = ["CANDIDATES", "CRITERIA", "LIBRARY", "STRUCTURE"]
