"""Turning a structure a user supplied into one canonical record, or refusing it.

The Phase 3 rule for datasets was that nothing enters the graph on a single reading. The same
rule applies here. A structure is parsed, standardized and written as a canonical SMILES; then
that canonical SMILES is read back from scratch and has to produce the same InChIKey and the
same atom, bond and charge counts. Two readings that disagree mean the import fails and both
readings are shown, because a structure the toolkit reads two ways is not a structure anyone
should compute on.

The InChIKey is the identity. Its first block of 14 characters is the skeleton without
stereochemistry or protonation, so a salt form and a stereoisomer of one substance share it,
which is what the restricted screen keys on and what lets the graph reuse a molecule it has
already seen while keeping a stereoisomer separate.

Standardization is recorded, never silent: ``Identity.steps`` lists what was changed (a salt
stripped, a charge neutralized), so an answer can say which structure was actually computed on
when it was not the one typed in.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from contextlib import contextmanager
from functools import lru_cache
from typing import Any, Iterator

MAX_ATOMS = 400          # a structure larger than this is a polymer or a protein, not a candidate


class MoleculeError(ValueError):
    """A structure that cannot be read, is too large, or reads two different ways."""


@dataclass(frozen=True)
class Identity:
    """One standardized structure, as it is stored and as the model is allowed to see it."""

    canonical_smiles: str
    inchi: str
    inchikey: str
    formula: str
    atoms: int
    bonds: int
    charge: int
    skeleton: str                              # the InChIKey's first block
    steps: tuple[str, ...] = field(default=())  # what standardization changed, in order
    input_smiles: str = ""

    def as_result(self) -> dict[str, Any]:
        """The tool result shape. Deliberately has no status field: chem nodes are hypotheses."""
        return {"kind": "molecule", "canonical_smiles": self.canonical_smiles,
                "inchikey": self.inchikey, "inchi": self.inchi, "formula": self.formula,
                "atoms": self.atoms, "bonds": self.bonds, "charge": self.charge,
                "skeleton": self.skeleton, "standardized": list(self.steps),
                "input": self.input_smiles}

    def summary(self) -> str:
        changed = f"; standardized: {', '.join(self.steps)}" if self.steps else ""
        return (f"{self.formula}, {self.atoms} atoms, charge {self.charge}, "
                f"InChIKey {self.inchikey}{changed}")


def parse(text: str, *, fmt: str = "smiles") -> Any:
    """Read one structure. ``fmt`` is smiles, molblock or inchi."""
    from rdkit import Chem

    raw = (text or "").strip()
    if not raw:
        raise MoleculeError("no structure given")
    if fmt == "smiles":
        mol = Chem.MolFromSmiles(raw)
    elif fmt == "molblock":
        mol = Chem.MolFromMolBlock(raw)
    elif fmt == "inchi":
        mol = Chem.MolFromInchi(raw)
    else:
        raise MoleculeError(f"unknown structure format {fmt!r}")
    if mol is None:
        raise MoleculeError(f"this is not a structure the toolkit can read: {raw[:60]!r}")
    if mol.GetNumAtoms() == 0:
        raise MoleculeError("the structure has no atoms")
    if mol.GetNumAtoms() > MAX_ATOMS:
        raise MoleculeError(f"{mol.GetNumAtoms()} atoms is past the {MAX_ATOMS}-atom limit for "
                            "a small-molecule candidate")
    return mol


def standardize(mol: Any) -> tuple[Any, tuple[str, ...]]:
    """Clean up a structure, returning it and the list of what was changed."""
    from rdkit import Chem
    from rdkit.Chem.MolStandardize import rdMolStandardize

    steps: list[str] = []
    out = rdMolStandardize.Cleanup(mol)
    if Chem.MolToSmiles(out) != Chem.MolToSmiles(mol):
        steps.append("normalized")

    fragments = Chem.GetMolFrags(out, asMols=True, sanitizeFrags=True)
    if len(fragments) > 1:
        out = rdMolStandardize.FragmentParent(out)
        steps.append(f"kept the largest of {len(fragments)} fragments")

    if any(a.GetFormalCharge() for a in out.GetAtoms()):
        before = Chem.MolToSmiles(out)
        neutral = rdMolStandardize.Uncharger().uncharge(out)
        if Chem.MolToSmiles(neutral) != before:
            out, _ = neutral, steps.append("neutralized")
    Chem.SanitizeMol(out)
    return out, tuple(steps)


def identity(mol: Any, *, steps: tuple[str, ...] = (), input_smiles: str = "") -> Identity:
    """The canonical record for a standardized structure."""
    from rdkit import Chem
    from rdkit.Chem import rdMolDescriptors

    inchi = Chem.MolToInchi(mol)
    key = Chem.MolToInchiKey(mol)
    if not inchi or not key:
        raise MoleculeError("the toolkit could not produce an InChI for this structure, so it "
                            "has no identity to store")
    return Identity(canonical_smiles=Chem.MolToSmiles(mol), inchi=inchi, inchikey=key,
                    formula=rdMolDescriptors.CalcMolFormula(mol), atoms=mol.GetNumAtoms(),
                    bonds=mol.GetNumBonds(),
                    charge=sum(a.GetFormalCharge() for a in mol.GetAtoms()),
                    skeleton=key.split("-")[0], steps=steps, input_smiles=input_smiles)


def read_twice(text: str, *, fmt: str = "smiles") -> Identity:
    """Parse, standardize, then read the canonical form back and require the same answer.

    This is the import gate. A disagreement raises ``MoleculeError`` naming both readings.
    """
    first, steps = standardize(parse(text, fmt=fmt))
    ident = identity(first, steps=steps, input_smiles=(text or "").strip())

    second = parse(ident.canonical_smiles, fmt="smiles")
    again = identity(second)
    mismatch = [f"{name}: {a!r} then {b!r}"
                for name, a, b in (("InChIKey", ident.inchikey, again.inchikey),
                                   ("formula", ident.formula, again.formula),
                                   ("atoms", ident.atoms, again.atoms),
                                   ("bonds", ident.bonds, again.bonds),
                                   ("charge", ident.charge, again.charge))
                if a != b]
    if mismatch:
        raise MoleculeError("the two readings of this structure disagree, so it was not "
                            "imported (" + "; ".join(mismatch) + ")")
    return ident


def screened(text: str, *, fmt: str = "smiles", threshold: float | None = None) -> Identity:
    """``read_twice`` plus the restricted screen. Raises ``MoleculeError`` on a refusal.

    Every path that brings a structure into the graph goes through here, so no structure is
    stored that the screen has not seen.
    """
    from sciai.domains.chem import restricted

    ident = read_twice(text, fmt=fmt)
    mol = parse(ident.canonical_smiles, fmt="smiles")
    kw = {} if threshold is None else {"threshold": threshold}
    result = restricted.screen(inchikey=ident.inchikey, mol=mol,
                               fingerprint=restricted.fingerprint_of(mol), **kw)
    if result.refused:
        raise MoleculeError(result.note)
    if result.flagged:
        return Identity(**{**ident.__dict__, "steps": (*ident.steps, f"flagged: {result.note}")})
    return ident


__all__ = ["MAX_ATOMS", "Identity", "MoleculeError", "identity", "parse", "read_twice",
           "screened", "standardize"]


# ---------------------------------------------------------------- provenance
# Yeri's rule (2026-10-07) is that a slot's value must be traceable to the question. For a
# structure, "traceable" cannot be string equality: the second live run answered five of eight
# chemistry questions wrongly because the router had picked a physics recipe, and the fix -- a
# text match -- then refused a *right* answer, because qwen3:8b rewrote the question's own
# "Cn1cnc2c1c(=O)n(C)c(=O)n2C" as the Kekule form "CN1C=NC2=C1C(=O)N(C)C(=O)N2C". String
# equality is also too generous in the other direction: "C" is a substring of almost any
# question that contains a SMILES, so methane passed as the molecule asked about.
#
# So a structure is sourced when the question itself contains a structure that standardizes to
# the same InChIKey. The comparison is deterministic chemistry, never the model's word, and a
# molecule the question does not contain still has nothing to match.
SMILES_TEXT = re.compile(r"[A-Za-z0-9@+\-\[\]()=#$%/\\.*]+")
# A SMILES never begins with any of these, and never ends with one either.
_CANNOT_START = "()=#-+/\\.%0123456789"
_CANNOT_END = "(=#-+/\\.%"


# Outside square brackets a SMILES may only name the organic subset, so a run holding any
# other letter is a word, not a structure, and is never handed to the toolkit.
_BARE_ATOM_LETTERS = set("BCNOPSFIlrHbcnops")


@contextmanager
def _quiet() -> Iterator[None]:
    """Silence the toolkit while a candidate is probed: a word that is not a structure is the
    expected answer here, not a failure worth printing to the user's console."""
    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.*")
    try:
        yield
    finally:
        RDLogger.EnableLog("rdApp.*")


def _could_be_smiles(text: str) -> bool:
    if not text:
        return False
    if "[" in text:            # a bracket atom may name anything, so let the toolkit decide
        return True
    return all(not c.isalpha() or c in _BARE_ATOM_LETTERS for c in text)


@lru_cache(maxsize=1024)
def inchikey_of(text: str) -> str | None:
    """The standard InChIKey of one structure, or None if it is not a readable structure.

    Cached, because provenance asks the same question of the same text repeatedly (a re-prompt
    re-plans the whole reply) and reading a structure twice is the expensive part.
    """
    if not _could_be_smiles(text):
        return None
    try:
        with _quiet():
            return read_twice(text).inchikey
    except Exception:  # noqa: BLE001 - "not a structure" is the answer here, not a failure
        return None


def _variants(run: str) -> list[str]:
    """One run of SMILES-legal characters, and the trims worth trying.

    A question writes a structure in prose -- "aspirin (CC(=O)Oc1ccccc1C(=O)O)?" -- so the run
    can carry the brackets around it. A SMILES may itself end in ")", so both are tried.
    """
    out = [run]
    trimmed = run.lstrip(_CANNOT_START).rstrip(_CANNOT_END)
    if trimmed and trimmed != run:
        out.append(trimmed)
    if trimmed.endswith(")"):
        out.append(trimmed[:-1])
    return [v for v in out if len(v) >= 1]


@lru_cache(maxsize=256)
def keys_in_text(text: str) -> frozenset[str]:
    """The InChIKeys of every structure the text itself contains.

    Runs of SMILES-legal characters are the candidates, so a JSON library in the question is
    split on its quotes and braces and each member is found on its own. A run that is not a
    structure simply contributes nothing.
    """
    keys = set()
    for run in SMILES_TEXT.findall(text or ""):
        for candidate in _variants(run):
            key = inchikey_of(candidate)
            if key:
                keys.add(key)
                break
    return frozenset(keys)


def in_text(structure: str, text: str) -> bool:
    """Does ``text`` contain a structure that is this same molecule?"""
    key = inchikey_of((structure or "").strip())
    return bool(key) and key in keys_in_text(text or "")
