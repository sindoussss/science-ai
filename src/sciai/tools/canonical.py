"""Canonical tool arguments (for fingerprints). Runs inside the sandbox because it parses."""
from __future__ import annotations

from typing import Any

from sciai.tools.parsing import canonical, parse, parse_relation
from sciai.tools.registry import get

RELATION_ARGS = {"equation"}


def _canon_value(key: str, value: Any, assumptions: dict[str, str] | None) -> Any:
    if isinstance(value, str):
        node = parse_relation(value, assumptions) if key in RELATION_ARGS else parse(value, assumptions)
        return canonical(node)
    if isinstance(value, list):
        return [_canon_value(key, v, assumptions) for v in value]
    if isinstance(value, dict):
        return {k: _canon_value(key, v, assumptions) for k, v in sorted(value.items())}
    return value


def canonicalize(tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
    spec = get(tool_name)
    assumptions = args.get("assumptions") or None
    out: dict[str, Any] = {}
    for key, value in sorted(args.items()):
        if key in spec.expr_args and value is not None:
            out[key] = _canon_value(key, value, assumptions)
        else:
            out[key] = value
    if "method" in spec.schema.get("properties", {}):
        out.setdefault("method", "default")
    return out
