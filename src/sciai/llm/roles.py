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


# Data tools the model may call. data.load (import) and stats.adjust (the family rule) are run by
# the app itself, never chosen by the model.
DATA_SOLVERS = ("data.describe", "data.filter", "data.derive", "data.group", "stats.ttest", "stats.mannwhitney",
                "stats.correlation", "stats.chi2", "stats.anova", "stats.kruskal", "stats.regression")
SOLVERS = ("sympy.", "numeric.evaluate", "numeric.quad", "numeric.root", "units.convert", "plot.",
           "phys.evaluate", "phys.constant", "ode.dsolve", "ode.solve_ivp", "linalg.solve", "circuit.dc",
           *DATA_SOLVERS)

ROLES: dict[str, Role] = {
    "formalizer": Role("formalizer", "formalize.md", SOLVERS),
    "controller": Role("controller", "controller.md", SOLVERS),
    "algebra": Role("algebra", "algebra.md", ("sympy.",)),
    "numeric": Role("numeric", "numeric.md", ("numeric.evaluate", "numeric.quad", "numeric.root", "units.convert")),
    "physics": Role("physics", "physics.md", ("phys.", "ode.", "units.", "sympy.", "numeric.", "plot.")),
    "circuits": Role("circuits", "circuits.md", ("circuit.", "linalg.", "ode.", "plot.")),
    "data": Role("data", "data.md", (*DATA_SOLVERS, "plot.")),
}

# After a failed check the retry goes to a specialist role for the failed tool, which
# must pick a different method or tool (a repeated fingerprint is refused).
RETRY_ROLE_BY_PREFIX = (("sympy.", "algebra"), ("phys.", "physics"), ("units.", "physics"), ("ode.", "physics"),
                        ("circuit.", "circuits"), ("linalg.", "circuits"), ("data.", "data"), ("stats.", "data"))


def retry_role(tool: str) -> str:
    return next((role for prefix, role in RETRY_ROLE_BY_PREFIX if tool.startswith(prefix)), "numeric")
