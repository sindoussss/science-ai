"""Molecular descriptors, and the independent recomputation that checks them.

Two legs, because the descriptors are not all the same kind of thing:

* **Recomputed.** Masses and counts are computed a second time by this module's own code over
  a pinned element table, with no RDKit descriptor function involved. If RDKit's molecular
  weight and a sum over atoms of IUPAC standard weights disagree, something is wrong with the
  molecule that reached the tool, and the node fails. This leg is a real second implementation.

* **Invariance.** TPSA and anything else with a fitted contribution table cannot honestly be
  reimplemented here; a copy of the same table is not a second opinion. Those are checked for
  the properties a correct value must have anyway: unchanged through a canonical SMILES round
  trip, and unchanged when the atoms are renumbered. That catches the failures this layer
  actually suffers (the wrong molecule reaching the tool, a stale cache, two descriptors
  swapped) and it does not pretend to audit RDKit's chemistry. Published reference values are
  compared in the live check, where Phase 3 keeps its published statistics.

Element data is pinned here on purpose. Standard atomic weights move in their last digits
between IUPAC releases, and a fingerprint that shifts with a dependency upgrade would
invalidate stored nodes for no chemical reason.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# IUPAC standard atomic weights (2021), for average molecular weight.
ATOMIC_WEIGHT: dict[str, float] = {
    "H": 1.008, "Li": 6.94, "B": 10.81, "C": 12.011, "N": 14.007, "O": 15.999,
    "F": 18.998403162, "Na": 22.98976928, "Mg": 24.305, "Al": 26.9815384, "Si": 28.085,
    "P": 30.973761998, "S": 32.06, "Cl": 35.45, "K": 39.0983, "Ca": 40.078, "Fe": 55.845,
    "Zn": 65.38, "As": 74.921595, "Se": 78.971, "Br": 79.904, "I": 126.90447, "Pt": 195.084,
}

# Mass of the most abundant isotope, for monoisotopic (exact) mass.
PRINCIPAL_ISOTOPE: dict[str, float] = {
    "H": 1.0078250319, "Li": 7.0160040, "B": 11.0093055, "C": 12.0, "N": 14.0030740052,
    "O": 15.9949146221, "F": 18.99840320, "Na": 22.98976928, "Mg": 23.9850417,
    "Al": 26.98153844, "Si": 27.9769265327, "P": 30.97376151, "S": 31.97207069,
    "Cl": 34.96885271, "K": 38.9637069, "Ca": 39.9625912, "Fe": 55.9349421, "Zn": 63.9291466,
    "As": 74.9215964, "Se": 79.9165218, "Br": 78.9183376, "I": 126.904468, "Pt": 194.9647911,
}

# A cation is lighter than its neutral atoms by the electrons it lost, and an anion heavier.
# The element tables above are neutral-atom masses, so a charged species needs this correction;
# without it a quaternary ammonium disagrees with RDKit by exactly one electron mass, which is
# how this constant came to be here.
ELECTRON_MASS = 0.000548579909

# Average molecular weight compares two tables of *conventional* values, and those legitimately
# differ: RDKit carries selenium at 78.96 where IUPAC 2021 has 78.971, which is 1e-4 relative on
# a small selenide. So this leg is the loose one. It still catches any real error, because a
# miscounted atom moves the weight by a whole unit, twenty times this tolerance.
MW_REL_TOL = 5e-4
# Monoisotopic mass compares measured physical constants, not conventions, so it is the sharp
# leg and the one that decides whether the molecule in hand is the molecule that was asked for.
EXACT_REL_TOL = 1e-6
TPSA_ABS_TOL = 1e-6      # the invariance leg: the same value, not a similar one

NAMES = ("mw", "exact_mass", "heavy_atoms", "rings", "aromatic_rings", "hbd", "hba", "tpsa",
         "rotatable_bonds", "fraction_csp3", "formal_charge")

UNITS = {"mw": "g/mol", "exact_mass": "g/mol", "tpsa": "A^2"}


class DescriptorError(ValueError):
    """A molecule holding an element this module has no pinned mass for."""


@dataclass(frozen=True)
class Descriptors:
    values: dict[str, float]

    def as_result(self) -> dict[str, Any]:
        return {"kind": "descriptors", "values": dict(self.values), "units": dict(UNITS)}

    def summary(self) -> str:
        v = self.values
        return (f"MW {v['mw']:.2f}, exact mass {v['exact_mass']:.4f}, "
                f"{int(v['heavy_atoms'])} heavy atoms, {int(v['rings'])} rings, "
                f"TPSA {v['tpsa']:.2f} A^2, HBD {int(v['hbd'])}, HBA {int(v['hba'])}, "
                f"{int(v['rotatable_bonds'])} rotatable bonds")


def compute(mol: Any) -> Descriptors:
    """Every descriptor, from RDKit. ``check`` then recomputes the ones that can be."""
    from rdkit.Chem import Descriptors as D
    from rdkit.Chem import rdMolDescriptors as MD

    ring_info = mol.GetRingInfo()
    return Descriptors({
        "mw": float(D.MolWt(mol)),
        "exact_mass": float(D.ExactMolWt(mol)),
        "heavy_atoms": float(mol.GetNumHeavyAtoms()),
        "rings": float(ring_info.NumRings()),
        "aromatic_rings": float(MD.CalcNumAromaticRings(mol)),
        "hbd": float(MD.CalcNumLipinskiHBD(mol)),
        "hba": float(MD.CalcNumLipinskiHBA(mol)),
        "tpsa": float(MD.CalcTPSA(mol)),
        # NonStrict is Veber's definition, the one the drug-likeness filter is defined against;
        # RDKit's default additionally discounts amide and ester linkages.
        "rotatable_bonds": float(MD.CalcNumRotatableBonds(
            mol, MD.NumRotatableBondsOptions.NonStrict)),
        "fraction_csp3": float(MD.CalcFractionCSP3(mol)),
        "formal_charge": float(sum(a.GetFormalCharge() for a in mol.GetAtoms())),
    })


# ------------------------------------------------------------------ the recomputation leg
def recompute(mol: Any) -> dict[str, float]:
    """Masses and counts again, from this module's own tables and a walk over the atoms."""
    average = exact = 0.0
    heavy = hbd = hba = charge = 0
    for atom in mol.GetAtoms():
        symbol = atom.GetSymbol()
        hydrogens = atom.GetTotalNumHs()
        try:
            average += ATOMIC_WEIGHT[symbol] + hydrogens * ATOMIC_WEIGHT["H"]
            exact += PRINCIPAL_ISOTOPE[symbol] + hydrogens * PRINCIPAL_ISOTOPE["H"]
        except KeyError:
            raise DescriptorError(
                f"no pinned mass for element {symbol!r}, so the molecular weight of this "
                "structure cannot be checked independently") from None
        if atom.GetAtomicNum() > 1:
            heavy += 1
        if symbol in ("N", "O"):
            hba += 1                 # Lipinski's acceptor count is the N and O atoms
            hbd += hydrogens         # and its donor count is the hydrogens on them
        charge += atom.GetFormalCharge()
    average -= charge * ELECTRON_MASS
    exact -= charge * ELECTRON_MASS
    return {"mw": average, "exact_mass": exact, "heavy_atoms": float(heavy), "hbd": float(hbd),
            "hba": float(hba), "formal_charge": float(charge),
            "rings": float(_ring_count(mol)), "rotatable_bonds": float(_rotatable(mol))}


def _ring_count(mol: Any) -> int:
    """The smallest set of smallest rings, by Euler's formula over each connected fragment."""
    from rdkit import Chem

    fragments = Chem.GetMolFrags(mol)
    return mol.GetNumBonds() - mol.GetNumAtoms() + len(fragments)


def _rotatable(mol: Any) -> int:
    """Veber's rotatable bonds, counted by walking the bonds rather than by a SMARTS match.

    The definition: a single, acyclic bond whose two atoms each have more than one heavy
    neighbour and neither of which is part of a triple bond. ``compute`` asks RDKit for the
    same quantity through its NonStrict option, so the two paths share no code. Note this is
    the Veber count the drug-likeness filter uses, not RDKit's stricter default, which also
    discounts amide and ester linkages.
    """
    from rdkit.Chem import BondType

    count = 0
    for bond in mol.GetBonds():
        if bond.GetBondType() is not BondType.SINGLE or bond.IsInRing():
            continue
        ends = (bond.GetBeginAtom(), bond.GetEndAtom())
        if any(_heavy_degree(a) < 2 for a in ends):
            continue
        if any(b.GetBondType() is BondType.TRIPLE for a in ends for b in a.GetBonds()):
            continue
        count += 1
    return count


def _heavy_degree(atom: Any) -> int:
    return sum(1 for n in atom.GetNeighbors() if n.GetAtomicNum() > 1)


# ------------------------------------------------------------------ the invariance leg
def invariants(mol: Any) -> dict[str, float]:
    """TPSA from a canonical round trip and from a renumbered copy of the same molecule."""
    from rdkit import Chem
    from rdkit.Chem import rdMolDescriptors as MD

    round_trip = Chem.MolFromSmiles(Chem.MolToSmiles(mol))
    if round_trip is None:
        raise DescriptorError("the canonical SMILES of this structure does not read back")
    order = list(range(mol.GetNumAtoms()))[::-1]
    renumbered = Chem.RenumberAtoms(mol, order)
    return {"tpsa_round_trip": float(MD.CalcTPSA(round_trip)),
            "tpsa_renumbered": float(MD.CalcTPSA(renumbered))}


def check(mol: Any, values: dict[str, float]) -> dict[str, Any]:
    """Compare a descriptor result against both legs. Returns a checker result payload."""
    disagreements: list[dict[str, Any]] = []
    recomputed = recompute(mol)
    for name, theirs in recomputed.items():
        reported = values.get(name)
        if reported is None:
            disagreements.append({"descriptor": name, "problem": "missing from the result"})
            continue
        tol = {"mw": MW_REL_TOL, "exact_mass": EXACT_REL_TOL}.get(name, 0.0)
        if not _close(float(reported), theirs, tol):
            disagreements.append({"descriptor": name, "reported": float(reported),
                                  "recomputed": theirs, "rel_tol": tol})

    inv = invariants(mol)
    tpsa = values.get("tpsa")
    if tpsa is None:
        disagreements.append({"descriptor": "tpsa", "problem": "missing from the result"})
    else:
        for name, other in inv.items():
            if abs(float(tpsa) - other) > TPSA_ABS_TOL:
                disagreements.append({"descriptor": "tpsa", "invariant": name,
                                      "reported": float(tpsa), "under_invariant": other})

    outcome = "fail" if disagreements else "pass"
    return {"kind": "check", "outcome": outcome, "method": "recompute+invariance",
            "recomputed": recomputed, "invariants": inv, "disagreements": disagreements,
            "checked": sorted(set(recomputed) | {"tpsa"}),
            "not_recomputed": ["tpsa", "aromatic_rings", "fraction_csp3"]}


def _close(a: float, b: float, rel_tol: float) -> bool:
    if rel_tol <= 0:
        return a == b
    scale = max(abs(a), abs(b), 1.0)
    return abs(a - b) <= rel_tol * scale


__all__ = ["ATOMIC_WEIGHT", "NAMES", "PRINCIPAL_ISOTOPE", "UNITS", "DescriptorError",
           "Descriptors", "check", "compute", "invariants", "recompute"]
