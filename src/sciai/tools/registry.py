"""Tool registry. A tool declares its name, JSON schema, domain and kind.

Tool functions take a validated ``args`` dict and return::

    {"result": {...typed result...}, "confidence": float | None, "meta": {...}}

Checkers return ``{"result": {"kind": "check", "outcome": "pass|fail|inconclusive", ...}}``.
They run inside the sandbox process; the main process only sees specs.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Literal

import jsonschema

from sciai.graph.model import Domain

ToolFn = Callable[[dict[str, Any]], dict[str, Any]]

ASSUMPTIONS_SCHEMA = {
    "type": "object",
    "additionalProperties": {
        "enum": ["real", "positive", "negative", "nonnegative", "nonpositive", "integer", "nonzero"]
    },
}


@dataclass(frozen=True)
class ToolSpec:
    name: str
    domain: Domain
    kind: Literal["solver", "checker"]
    description: str
    schema: dict[str, Any]
    fn: ToolFn
    # Args holding expressions: canonicalized for fingerprints, scanned for numbers.
    expr_args: tuple[str, ...] = ()
    # Step-type rule: True, or a predicate over args (e.g. "long algebra").
    always_check: bool | Callable[[dict[str, Any]], bool] = False
    methods: tuple[str, ...] = ("default",)
    cost: int = 1
    script: Callable[[dict[str, Any]], str] | None = field(default=None, compare=False)
    # Custom canonical form for fingerprints (quantities in SI, netlists sorted...); runs in the sandbox.
    canonical: Callable[[dict[str, Any]], dict[str, Any]] | None = field(default=None, compare=False)
    # An argument holding a domain entity (a circuit netlist): the controller stores it as an
    # entity node, and the result depends on that node.
    entity_arg: str | None = None
    describe_entity: Callable[[Any], str] | None = field(default=None, compare=False)
    # An argument naming a graph node (a handle like "n5"): the controller resolves it, the result
    # depends on that node, and the node's result is passed to the tool as "data".
    node_arg: str | None = None
    # An argument naming a dataset (a dataset node's handle, or an imported file's name): the
    # controller resolves it to that node, the result depends on it, and the tool receives the
    # node's dataset descriptor in place of the name. Checks receive the descriptor too.
    dataset_arg: str | None = None

    def validate(self, args: dict[str, Any]) -> None:
        jsonschema.validate(args, self.schema)

    def requires_check(self, args: dict[str, Any]) -> bool:
        if callable(self.always_check):
            return bool(self.always_check(args))
        return bool(self.always_check)

    def prompt_line(self) -> str:
        props = self.schema.get("properties", {})
        req = set(self.schema.get("required", []))
        parts = []
        for k, v in props.items():
            t = v.get("type", "any") if isinstance(v, dict) else "any"
            if k == self.dataset_arg:
                t = "dataset"
            elif isinstance(t, list):
                t = "|".join(t)
            if isinstance(v, dict) and "enum" in v:
                t = "|".join(map(str, v["enum"]))
            parts.append(f"{k}{'' if k in req else '?'}:{t}")
        return f"{self.name}({', '.join(parts)}) - {self.description}"


_REGISTRY: dict[str, ToolSpec] = {}


def register(spec: ToolSpec) -> ToolSpec:
    if spec.name in _REGISTRY:
        raise ValueError(f"tool {spec.name} registered twice")
    _REGISTRY[spec.name] = spec
    return spec


def get(name: str) -> ToolSpec:
    load_builtin_tools()
    try:
        return _REGISTRY[name]
    except KeyError:
        raise KeyError(f"unknown tool {name!r}") from None


def all_tools() -> list[ToolSpec]:
    load_builtin_tools()
    return list(_REGISTRY.values())


_loaded = False


def load_builtin_tools() -> None:
    global _loaded
    if _loaded:
        return
    _loaded = True
    from sciai.tools import (  # noqa: F401
        circuit_tools,
        data_tools,
        linalg_tools,
        numeric_tools,
        ode_tools,
        phys_tools,
        plot_tools,
        stats_tools,
        sympy_tools,
        units_tools,
    )


def schema(properties: dict[str, Any], required: list[str]) -> dict[str, Any]:
    return {"type": "object", "properties": properties, "required": required, "additionalProperties": False}
