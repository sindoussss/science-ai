"""Fault injection for tests and demos: corrupt chosen tool results on purpose."""
from __future__ import annotations

import copy
from typing import Any, Callable

from sciai.tools.parsing import expr_result, parse
from sciai.tools.sandbox import Runner, ToolOutcome


class FaultyRunner:
    """Wraps a runner; for ``tool``, corrupts the results of the given call numbers (1-based).
    ``calls=None`` corrupts every call. Checkers and canonicalization are never touched."""

    def __init__(self, inner: Runner, tool: str, wrong_expr: str | None = None, calls: set[int] | None = None,
                 corrupt: Callable[[dict[str, Any]], dict[str, Any]] | None = None) -> None:
        """``wrong_expr`` replaces the result with that expression; ``corrupt`` maps the real
        result to a wrong one instead (e.g. a quantity in the wrong unit)."""
        if (wrong_expr is None) == (corrupt is None):
            raise ValueError("give exactly one of wrong_expr and corrupt")
        self.inner = inner
        self.tool = tool
        self.wrong = wrong_expr
        self.corrupt = corrupt
        self.calls = calls
        self.count = 0
        self.corrupted: list[int] = []
        self.on_call: Callable[[str, dict[str, Any]], None] | None = None

    def run(self, tool: str, args: dict[str, Any], timeout: float | None = None) -> ToolOutcome:
        if self.on_call:
            self.on_call(tool, args)
        out = self.inner.run(tool, args, timeout)
        if tool != self.tool or not out.ok:
            return out
        self.count += 1
        if self.calls is None or self.count in self.calls:
            out = copy.deepcopy(out)
            res = out.value["result"]
            out.value["result"] = self.corrupt(res) if self.corrupt else expr_result(parse(self.wrong or ""))
            self.corrupted.append(self.count)
        return out

    def canonicalize(self, tool: str, args: dict[str, Any]) -> ToolOutcome:
        return self.inner.canonicalize(tool, args)

    def close(self) -> None:
        self.inner.close()
