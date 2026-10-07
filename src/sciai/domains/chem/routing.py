"""Is this question about a molecule? A code guard on the router, not a prompt instruction.

The first live run of the recipe branch answered "what are the molecular weight and TPSA of
caffeine" with 3.9728917e-19 J: the router had sent the question to ``photon_energy`` and the
model filled that recipe's slots from the worked example in its own prompt. The prompt already
said not to; a prompt cannot be relied on for this.

So the question's own text decides what a molecule question may become. A question that names a
structure, a molecule or a chemistry quantity may only be routed to a chemistry recipe, to the
``molecule_question`` path, or out of scope. It is never sent to a recipe about photons,
projectiles or circuits, whatever the model replies.

This does not decline anything and cannot: declining is ``decline.py``'s job and is decided by
capability. This only narrows which route may answer.
"""
from __future__ import annotations

import re

# Chemistry quantities, identifiers and operations, as a question writes them. "compound" and
# "structure" are left out on purpose: "compound interest" and "the structure of the beam" are
# not chemistry, and nothing here needs them -- a molecule question that names none of these
# still carries a structure, which STRUCTURE_MARKERS finds.
TERMS = re.compile(
    r"\b(molecular weight|molar mass|molecular formula|monoisotopic|tpsa|"
    r"polar surface area|logp|clogp|lipophilic\w*|lipinski|veber|rule of five|"
    r"drug-?like\w*|tanimoto|morgan fingerprint|fingerprint|inchi\w*|smiles|smarts|"
    r"substructure|rotatable bonds?|hydrogen bond (?:donors?|acceptors?)|"
    r"molecules?|drug candidates?|pharmacophore|stereocent\w+|tautomers?)\b", re.I)

# Molecules by name. Partial by construction, which is why it is one signal of three.
NAMES = re.compile(
    r"\b(aspirin|acetylsalicylic acid|salicylic acid|caffeine|ibuprofen|paracetamol|"
    r"acetaminophen|penicillin|morphine|codeine|nicotine|ethanol|methanol|benzene|toluene|"
    r"phenol|aniline|acetone|glucose|sucrose|cholesterol|testosterone|oestradiol|estradiol|"
    r"adrenaline|dopamine|serotonin|histamine|atorvastatin|metformin|omeprazole|sildenafil|"
    r"warfarin|diazepam|amoxicillin|sertraline|lidocaine)\b", re.I)

# Fragments that appear in SMILES and essentially never in prose: a branch opening on a double
# bond, a bracket atom, a chiral marker, an aromatic ring-closure. One is enough.
STRUCTURE_MARKERS = re.compile(r"\(=|=\)|\[[A-Za-z@+\-]|@@|\b[cnops]\d|\)[cnops]\d?|"
                               r"[A-Za-z]\d\)|InChI=", re.I)


def looks_chemical(question: str) -> bool:
    """True when the question is about a molecule, by its own text.

    Three independent signals: a chemistry term, a molecule by name, or a fragment of a written
    structure. Any one is enough, because the cost of a false positive is only that the
    question must go to a chemistry recipe or out of scope, while the cost of a false negative
    is an answer computed from a physics recipe's worked example.
    """
    text = question or ""
    return bool(TERMS.search(text) or NAMES.search(text) or STRUCTURE_MARKERS.search(text))


__all__ = ["NAMES", "STRUCTURE_MARKERS", "TERMS", "looks_chemical"]
