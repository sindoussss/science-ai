"""Core graph data types.

A Node row is one *version* of a logical node. Versions of the same logical
node share a ``lineage_id``. Edges connect versions, never lineages, so a
dependent always points at the exact value it consumed.
"""
from __future__ import annotations

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


def result_to_text(result: dict[str, Any]) -> str:
    kind = result.get("kind")
    value = result.get("value")
    units = result.get("units")
    if kind == "list":
        text = ", ".join(result_to_text(v) for v in value)
    elif kind == "plotspec":
        text = f"plot ({len(value.get('x', []))} points)"
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
