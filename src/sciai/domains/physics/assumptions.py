"""Default modelling assumptions per problem type, defined in code.

formalize names a ``problem_type``; its checklist is always offered when the user
confirms the problem, with the model's own assumptions below it (duplicates removed).
Every item starts ticked. A ticked item becomes a confirmed assumption node; an
unticked one is recorded as rejected, so nothing can rely on it.
"""
from __future__ import annotations

import re
from typing import Any

G0 = "constant g = 9.80665 m/s^2"

DEFAULTS: dict[str, tuple[str, ...]] = {
    "projectile": ("no air resistance", G0, "launch and landing at the same height"),
    "kinematics": ("point mass", "no friction unless given", G0),
    "dynamics": ("point mass", "no friction unless given", G0),
    "energy": ("no losses unless given", "closed system"),
    "thermo": ("ideal gas", "quasi-static process", "closed system"),
    "dc_circuit": ("ideal wires", "ideal sources", "linear, constant resistors", "steady state"),
    "rc_rl_transient": ("ideal components", "switch acts instantly", "initial conditions as given"),
    "ode": ("constant parameters", "initial conditions as given"),
    "general": ("constants from CODATA", "given values are exact"),
}
PROBLEM_TYPES = tuple(DEFAULTS)
NO_CHECKLIST = ("math",)  # pure mathematics: no modelling assumptions to offer
MAX_MODEL_ASSUMPTIONS = 8
MAX_ASSUMPTION_CHARS = 120


def normalize_type(problem_type: str | None) -> str | None:
    """None or "math" -> None (no defaults); a known type -> itself; anything else -> general."""
    if problem_type is None:
        return None
    t = str(problem_type).strip().lower().replace("-", "_").replace(" ", "_")
    if not t or t in NO_CHECKLIST:
        return None
    return t if t in DEFAULTS else "general"


def _key(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def checklist(problem_type: str | None, model_assumptions: list[str] | None = None) -> list[dict[str, Any]]:
    """The items to confirm: the type's defaults first, then the model's, without duplicates."""
    items: list[dict[str, Any]] = []
    seen: set[str] = set()
    t = normalize_type(problem_type)
    sources = [(text, "default") for text in DEFAULTS.get(t, ()) if t] + \
              [(text, "model") for text in (model_assumptions or [])]
    for text, source in sources:
        text = " ".join(str(text).split())[:MAX_ASSUMPTION_CHARS]
        k = _key(text)
        if not k or k in seen:
            continue
        seen.add(k)
        items.append({"text": text, "source": source})
    return items
