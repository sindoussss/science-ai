"""One model, many roles. A role is a system prompt plus the tools it may call."""
from __future__ import annotations

from dataclasses import dataclass
from importlib import resources

from sciai.tools.registry import all_tools


@dataclass(frozen=True)
class Role:
    name: str
    prompt_file: str
    tools: tuple[str, ...]  # tool names, or prefixes ending in "."

    def allows(self, tool: str) -> bool:
        return any(tool == t or (t.endswith(".") and tool.startswith(t)) for t in self.tools)

    def tool_lines(self) -> str:
        return "\n".join(s.prompt_line() for s in all_tools() if s.kind == "solver" and self.allows(s.name))

    def system_prompt(self) -> str:
        text = resources.files("sciai.llm").joinpath("prompts", self.prompt_file).read_text(encoding="utf-8")
        return text.replace("{TOOLS}", self.tool_lines())


SOLVERS = ("sympy.", "numeric.evaluate", "numeric.quad", "numeric.root", "units.convert", "plot.")

ROLES: dict[str, Role] = {
    "formalizer": Role("formalizer", "formalize.md", SOLVERS),
    "controller": Role("controller", "controller.md", SOLVERS),
    "algebra": Role("algebra", "algebra.md", ("sympy.",)),
    "numeric": Role("numeric", "numeric.md", ("numeric.evaluate", "numeric.quad", "numeric.root", "units.convert")),
}

# For a retry, switch to a worker role different from the one that failed.
RETRY_ROLE = {"controller": "algebra", "algebra": "numeric", "numeric": "algebra"}
