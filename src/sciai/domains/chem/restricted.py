"""Screening a structure before it enters the graph, and screening a request's criteria.

Two separate jobs, deliberately kept apart from the capability decline in ``decline``:

* ``screen`` looks at a structure. It refuses the chemical-weapon families, flags chemotypes
  that are both legitimate medicine and worth labelling, and allows everything else. It runs on
  import and again on every similarity or ranking request, because a library member reached
  through a neighbour search never passed through import.
* ``criteria_refusal`` looks at what the request is optimizing for. Maximizing toxicity or
  lethality is refused; minimizing it is ordinary ADMET work and is not.

Three rules, in the order they fire:

1. **Identity.** A salted SHA-256 of the InChIKey skeleton block, matched against a stored list.
   Hashing means this file is not itself a list of agent structures, and keying on the skeleton
   block catches salts and stereoisomers of the same substance.
2. **Close analog.** Tanimoto against stored Morgan fingerprints, over a threshold. A packed
   fingerprint is not practically invertible back to a structure, so the mechanism screens for
   neighbours without shipping the neighbours.
3. **Structural alert.** Generic SMARTS for whole chemical families, each one carrying its own
   action and a note saying why, because some of these patterns are legitimate medicine.

Where this screen is partial, and it is partial: the shipped identity and fingerprint lists are
empty, since populating them means authoring a list of agent identifiers, and the alert set
covers families rather than individual substances. So the screen is a safety net, not a
guarantee. The protection that does not depend on coverage is that no tool in this phase plans a
reaction, writes a procedure or gives a dose, so there is nothing for a missed structure to be
used with.

False positives are the cost this file is tuned against. An alert that would catch a working
drug class is a ``flag``, which labels the node and lets the work proceed; only families with no
therapeutic use are a ``refuse``. ``tests/unit/test_chem_restricted.py`` pins both directions.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
from dataclasses import dataclass
from functools import lru_cache
from importlib import resources
from typing import Any

import re

DATA_FILE = "restricted.json"
FP_RADIUS = 2
FP_BITS = 2048
# Measured on Morgan radius 2, 2048 bits, with this module's own generator:
#   one substituent changed, a true close analog ..... 0.77
#   one methyl removed (caffeine to theophylline) .... 0.46
#   same chemotype, different drug (two mustards) .... 0.54
#   unrelated molecules .............................. 0.09 to 0.13
# So 0.70 catches a single-substituent analog while staying clear of the band where two
# different working drugs merely share a scaffold. It does not catch a remoter analogue, and
# the tests say so rather than implying the screen is tight.
DEFAULT_ANALOG_TANIMOTO = 0.70

ALLOW = "allow"
FLAG = "flag"
REFUSE = "refuse"

REFUSAL_MESSAGE = ("That structure is in a family this system will not work with, so the "
                   "request stops here.")
CRITERIA_MESSAGE = ("This system ranks candidates for therapeutic promise and will not rank "
                    "them for harm, so the request stops here.")

# Criteria that optimize *for* harm. Minimizing toxicity is the opposite and must not match:
# "least toxic", "lowest toxicity" and "reduce toxicity" are everyday ADMET goals.
_HARM_CRITERIA: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("maximize_harm", re.compile(r"\b(maximi[sz]e|increase|raise|boost|optimi[sz]e for)\b"
                                 r"[^.;]{0,40}\b(toxicity|lethality|letha\w+|harm|potency as a "
                                 r"(?:poison|toxin)|poison\w*)\b", re.I)),
    ("most_toxic", re.compile(r"\b(most|highest|greatest)\b[^.;]{0,20}"
                              r"\b(toxic\w*|lethal\w*|deadly|poison\w*)\b", re.I)),
    ("weaponize", re.compile(r"\b(chemical weapon|nerve agent|weaponi[sz]\w+|incapacitating "
                             r"agent|lethal dose for)\b", re.I)),
)


@dataclass(frozen=True)
class Screen:
    """The outcome for one structure. ``action`` is allow, flag or refuse."""

    action: str
    rule: str = ""
    note: str = ""
    detail: dict[str, Any] | None = None

    @property
    def refused(self) -> bool:
        return self.action == REFUSE

    @property
    def flagged(self) -> bool:
        return self.action == FLAG


@dataclass(frozen=True)
class CriteriaRefusal:
    rule: str
    message: str
    asked: str


def criteria_refusal(criteria: str) -> CriteriaRefusal | None:
    """A refusal when the request optimizes for harm, else None."""
    text = criteria or ""
    for rule, pattern in _HARM_CRITERIA:
        if pattern.search(text):
            return CriteriaRefusal(rule=rule, message=CRITERIA_MESSAGE, asked=text)
    return None


@lru_cache(maxsize=1)
def table() -> dict[str, Any]:
    """The restricted table, read once from the package data."""
    raw = resources.files("sciai.domains.chem").joinpath(DATA_FILE).read_text(encoding="utf-8")
    data = json.loads(raw)
    data.setdefault("salt", "")
    data.setdefault("identity", {})
    data["identity"].setdefault("refuse", [])
    data["identity"].setdefault("flag", [])
    data.setdefault("fingerprints", {})
    data["fingerprints"].setdefault("refuse", [])
    data.setdefault("alerts", [])
    return data


def key_hash(inchikey: str, *, salt: str | None = None) -> str:
    """The salted hash of an InChIKey's skeleton block (the first 14 characters)."""
    block = (inchikey or "").strip().upper().split("-")[0]
    s = table()["salt"] if salt is None else salt
    return hmac.new(s.encode(), block.encode(), hashlib.sha256).hexdigest()


def pack_fingerprint(bits: Any) -> str:
    """An RDKit bit vector (or an iterable of on-bit indices) as base64, for the table."""
    on = _on_bits(bits)
    buf = bytearray(FP_BITS // 8)
    for i in on:
        buf[i // 8] |= 1 << (i % 8)
    return base64.b64encode(bytes(buf)).decode()


def unpack_fingerprint(packed: str) -> set[int]:
    buf = base64.b64decode(packed)
    return {i for i in range(min(FP_BITS, len(buf) * 8)) if buf[i // 8] >> (i % 8) & 1}


def tanimoto(a: set[int], b: set[int]) -> float:
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)


def screen(*, inchikey: str = "", fingerprint: Any = None, mol: Any = None,
           threshold: float = DEFAULT_ANALOG_TANIMOTO,
           data: dict[str, Any] | None = None) -> Screen:
    """Screen one structure. Any of the three inputs may be omitted; each enables one rule.

    Refusals win over flags, and the first rule to refuse is the one reported.
    """
    t = data if data is not None else table()
    salt = t.get("salt", "")
    ident = t.get("identity", {})
    flags: list[Screen] = []

    if inchikey:
        h = key_hash(inchikey, salt=salt)
        if h in set(ident.get("refuse", [])):
            return Screen(REFUSE, "identity", REFUSAL_MESSAGE, {"matched": "identity"})
        if h in set(ident.get("flag", [])):
            flags.append(Screen(FLAG, "identity", "A listed controlled substance.",
                                {"matched": "identity"}))

    if fingerprint is not None:
        on = _on_bits(fingerprint)
        best, stored = 0.0, t.get("fingerprints", {}).get("refuse", [])
        for packed in stored:
            best = max(best, tanimoto(on, unpack_fingerprint(packed)))
        if best >= threshold:
            return Screen(REFUSE, "analog", REFUSAL_MESSAGE,
                          {"matched": "analog", "tanimoto": round(best, 3)})

    if mol is not None:
        for alert in t.get("alerts", []):
            if not _matches(mol, alert.get("smarts", "")):
                continue
            action = alert.get("action", FLAG)
            if action == REFUSE:
                return Screen(REFUSE, alert.get("id", "alert"), REFUSAL_MESSAGE,
                              {"matched": "alert", "alert": alert.get("id", "")})
            flags.append(Screen(FLAG, alert.get("id", "alert"),
                                alert.get("note", "A flagged chemotype."),
                                {"matched": "alert", "alert": alert.get("id", "")}))

    if flags:
        return flags[0]
    return Screen(ALLOW)


def fingerprint_of(mol: Any) -> Any:
    """The Morgan fingerprint this module screens with (RDKit, imported lazily)."""
    from rdkit.Chem import rdFingerprintGenerator

    gen = rdFingerprintGenerator.GetMorganGenerator(radius=FP_RADIUS, fpSize=FP_BITS)
    return gen.GetFingerprint(mol)


def _on_bits(bits: Any) -> set[int]:
    if bits is None:
        return set()
    if isinstance(bits, (set, frozenset)):
        return set(bits)
    get = getattr(bits, "GetOnBits", None)
    if get is not None:
        return set(get())
    nonzero = getattr(bits, "GetNonzeroElements", None)
    if nonzero is not None:
        return set(nonzero())
    return set(bits)


@lru_cache(maxsize=64)
def _smarts(pattern: str) -> Any:
    from rdkit import Chem

    return Chem.MolFromSmarts(pattern)


def _matches(mol: Any, pattern: str) -> bool:
    if not pattern:
        return False
    patt = _smarts(pattern)
    if patt is None:  # a bad pattern must not silently pass a structure
        raise ValueError(f"unparsable SMARTS in the restricted table: {pattern!r}")
    return bool(mol.HasSubstructMatch(patt))


__all__ = ["ALLOW", "CRITERIA_MESSAGE", "FLAG", "REFUSAL_MESSAGE", "REFUSE", "CriteriaRefusal",
           "Screen", "criteria_refusal", "fingerprint_of", "key_hash", "pack_fingerprint",
           "screen", "table", "tanimoto", "unpack_fingerprint"]
