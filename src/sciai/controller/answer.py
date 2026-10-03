"""Final answers are templates filled from node results, so the model never types a value."""
from __future__ import annotations

from typing import TYPE_CHECKING

from sciai.llm.actions import PLACEHOLDER

if TYPE_CHECKING:
    from sciai.graph.engine import GraphEngine


def render(engine: "GraphEngine", template: str) -> str:
    def sub(m) -> str:  # noqa: ANN001
        node = engine.resolve(m.group(1))
        return node.display_result() or node.title
    return PLACEHOLDER.sub(sub, template)
