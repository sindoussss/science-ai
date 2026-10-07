"""Core graph data types.

A Node row is one *version* of a logical node. Versions of the same logical
node share a ``lineage_id``. Edges connect versions, never lineages, so a
dependent always points at the exact value it consumed.
"""
from __future__ import annotations

import math
from decimal import ROUND_HALF_UP, Decimal
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class StrEnum(str, Enum):
    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return self.value


class Status(StrEnum):
    PROPOSED = "proposed"
    VERIFIED = "verified"
    FAILED = "failed"
    INVALIDATED = "invalidated"
    HYPOTHESIS = "hypothesis"


class Layer(StrEnum):
    REASONING = "reasoning"
    DOMAIN = "domain"
    PLOT = "plot"


class NodeType(StrEnum):
    PROBLEM = "problem"
    CLAIM = "claim"
    STEP = "step"
    TOOL_RESULT = "tool_result"
    CHECK = "check"
    FINAL = "final"
    ERROR = "error"
    HINT = "hint"
    ENTITY = "entity"
    PLOT = "plot"
    ASSUMPTION = "assumption"  # a modelling assumption; never verified, confirmed by the user


class EdgeKind(StrEnum):
    DEPENDS_ON = "depends_on"        # src was computed from dst
    CHECKS = "checks"                # src (a check) verifies dst
    CONFLICTS_WITH = "conflicts_with"
    RELATES = "relates"              # domain-layer relation
    RENDERS = "renders"              # src (plot) renders dst (data)


# Edge kinds along which invalidation propagates (from dst to src).
CASCADE_KINDS = (EdgeKind.DEPENDS_ON, EdgeKind.RENDERS)


class Domain(StrEnum):
    GENERAL = "general"
    MATH = "math"
    PHYSICS = "physics"
    DATA = "data"
    CHEM = "chem"


class LadderStage(StrEnum):
    NONE = "none"
    RETRIED = "retried"
    BACKTRACKED = "backtracked"
    ESCALATED = "escalated"


TERMINAL_BAD = (Status.FAILED, Status.INVALIDATED)


def new_id() -> str:
    return uuid.uuid4().hex


def now() -> float:
    return time.time()


@dataclass
class Node:
    session_id: str
    layer: Layer
    type: NodeType
    title: str
    domain: Domain = Domain.GENERAL
    content: str = ""
    content_canonical: str = ""
    fingerprint: str = ""
    tool_name: str | None = None
    tool_inputs: dict[str, Any] | None = None
    result: dict[str, Any] | None = None
    status: Status = Status.PROPOSED
    locked: bool = False
    needs_recheck: bool = False
    confidence: float | None = None
    risk: dict[str, Any] = field(default_factory=dict)
    flags: list[str] = field(default_factory=list)
    ladder_stage: LadderStage = LadderStage.NONE
    role: str | None = None
    id: str = field(default_factory=new_id)
    lineage_id: str = ""
    version: int = 1
    superseded_by: str | None = None
    created_at: float = field(default_factory=now)
    updated_at: float = field(default_factory=now)

    def __post_init__(self) -> None:
        if not self.lineage_id:
            self.lineage_id = self.id

    @property
    def warning(self) -> str | None:
        """Locked nodes keep their status, but a bad status earns a warning."""
        if self.locked and self.status in TERMINAL_BAD:
            return f"locked node is {self.status.value}"
        if self.needs_recheck:
            return "unverified result from an earlier session; re-check before use"
        return None

    def display_result(self) -> str:
        if not self.result:
            return ""
        return result_to_text(self.result)


# Chemistry result kinds, named here so the renderer below stays a flat dispatch. The import
# of the renderer itself is deferred: rdkit is slow to load and the graph model is not.
CHEM_KINDS = ("molecule", "descriptors", "estimate", "druglike", "neighbours", "substructure",
              "clusters", "ranking", "literature")


# What a reader sees. The stored value keeps full precision; this is the display only.
# Four significant figures is what a worked answer is quoted to, and it is what stops
# "35.32400580358996 m" reading as a measurement nobody made.
SIG_FIGS = 4
# A temperature is written to this many decimals instead of four significant figures: 25 degC
# is exactly 298.15 K, and "298.1 K" is a worse answer than the question's own precision.
TEMPERATURE_DECIMALS = 2
TEMPERATURE_UNITS = frozenset({
    "k", "kelvin", "degk", "degree_kelvin", "degc", "celsius", "degree_celsius",
    "degf", "fahrenheit", "degree_fahrenheit", "degr", "rankine", "degree_rankine",
    "°c", "°f", "°k",
})


def _round_half_up(v: float, digits: int) -> float:
    """Round away from zero on a tie, as arithmetic is taught. Python's own round() is
    half-to-even, which turned 298.15 into 298.1."""
    q = Decimal(1).scaleb(-digits)
    return float(Decimal(repr(v)).quantize(q, rounding=ROUND_HALF_UP))


def show_number(value: Any, decimals: int | None = None) -> str:
    """A value as the user reads it: four significant figures, rounded half up.

    ``decimals`` fixes the number of decimal places instead, for a quantity whose unit needs
    them (a temperature). Trailing zeros are dropped, so 25 degC stays "25".
    """
    try:
        v = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(v):
        return str(v)
    if decimals is not None and abs(v) >= 10 ** -decimals:
        v = _round_half_up(v, decimals)
        if v == int(v):
            return str(int(v))
        return f"{v:.{decimals}f}".rstrip("0").rstrip(".")
    if v == int(v) and abs(v) < 1e16:
        return str(int(v))
    exponent = math.floor(math.log10(abs(v)))
    if -5 < exponent < SIG_FIGS + 2:   # the range %g writes without an exponent
        v = _round_half_up(v, SIG_FIGS - 1 - exponent)
        if v == int(v):
            return str(int(v))
    return f"{v:.{SIG_FIGS}g}"


def temperature_decimals(unit: Any) -> int | None:
    """``TEMPERATURE_DECIMALS`` when the unit is a temperature, else None."""
    text = str(unit or "").strip().lower().replace(" ", "_")
    return TEMPERATURE_DECIMALS if text in TEMPERATURE_UNITS else None


def result_to_text(result: dict[str, Any]) -> str:
    kind = result.get("kind")
    value = result.get("value")
    units = result.get("units")
    if kind == "list":
        labels = result.get("labels")
        if labels and len(labels) == len(value):
            text = ", ".join(f"{lbl} = {result_to_text(v)}" for lbl, v in zip(labels, value))
        else:
            text = ", ".join(result_to_text(v) for v in value)
    elif kind == "quantity":
        # the unit is written once, here: a caller that adds it again is what produced
        # "35.3 m meters", so answer sentences never name the unit themselves
        unit = result.get("unit") or ""
        text = f"{show_number(value, temperature_decimals(unit))} {unit}".rstrip()
    elif kind == "series":
        t = result.get("t") or [0.0]
        finals = ", ".join(f"{f} = {show_number(v)}" for f, v in zip(result.get("funcs", []), value or []))
        text = f"{finals} at {result.get('var', 't')} = {show_number(t[-1])}"
    elif kind == "plotspec":
        if value.get("version") == 2:
            from sciai.graph.plotspec import summary

            text = summary(value)
        else:
            text = f"plot ({len(value.get('x', []))} points)"
    elif kind in CHEM_KINDS:
        from sciai.domains.chem import report as chem_report

        # units live inside each chemistry result, so none is appended below
        return chem_report.text_for(result)
    elif kind in ("dataset", "table", "stats", "adjusted"):
        from sciai.domains.data import report

        text = {"dataset": report.dataset_text, "table": report.table_text, "stats": report.stats_text,
                "adjusted": report.adjusted_text}[kind](result)
    elif kind == "number":
        # a converted temperature arrives as a plain number whose unit is in ``units``
        text = show_number(value, temperature_decimals(units))
    else:
        text = str(value)
    return f"{text} {units}" if units else text


@dataclass
class Edge:
    src: str
    dst: str
    kind: EdgeKind
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class Evidence:
    node_id: str
    method: str
    tool_name: str
    inputs: dict[str, Any]
    outcome: str  # pass | fail | inconclusive
    detail: dict[str, Any]
    check_node_id: str | None = None
    id: str = field(default_factory=new_id)
    created_at: float = field(default_factory=now)


@dataclass
class CascadeReport:
    """What an invalidation touched, including other sessions (item 6)."""

    origin_id: str
    invalidated: list[str] = field(default_factory=list)
    # session_id -> list of (node_id, title) of final answers that are now invalid
    affected_sessions: dict[str, list[tuple[str, str]]] = field(default_factory=dict)
    locked_warnings: list[str] = field(default_factory=list)

    @property
    def crosses_sessions(self) -> bool:
        return bool(self.affected_sessions)
