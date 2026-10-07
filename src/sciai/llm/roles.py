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
    # Tools listed in the prompt, when that is narrower than what the role may call. A tool a
    # recipe already covers needs no line in the router prompt, but a goal may still name it.
    shown: tuple[str, ...] = ()

    def allows(self, tool: str) -> bool:
        return self._matches(self.tools, tool)

    @staticmethod
    def _matches(allowed: tuple[str, ...], tool: str) -> bool:
        return any(tool == t or (t.endswith(".") and tool.startswith(t)) for t in allowed)

    def tool_lines(self) -> str:
        listed = self.shown or self.tools
        return "\n".join(s.prompt_line() for s in all_tools()
                          if s.kind == "solver" and self._matches(listed, s.name))

    def system_prompt(self) -> str:
        text = resources.files("sciai.llm").joinpath("prompts", self.prompt_file).read_text(encoding="utf-8")
        if "{RECIPES}" in text:
            from sciai.controller.recipes import prompt_block

            text = text.replace("{RECIPES}", prompt_block())
        return text.replace("{TOOLS}", self.tool_lines())


# Data tools the model may call. data.load (import) and stats.adjust (the family rule) are run by
# the app itself, never chosen by the model.
DATA_SOLVERS = ("data.describe", "data.filter", "data.derive", "data.group", "stats.ttest", "stats.mannwhitney",
                "stats.correlation", "stats.chi2", "stats.anova", "stats.kruskal", "stats.regression")
# Chemistry tools the model may call. Every one of them is a screening tool; the set is also
# what ``domains.chem.decline`` consults, so what this list omits is what the system declines.
CHEM_SOLVERS = ("chem.parse", "chem.descriptors", "chem.logp", "chem.druglike", "chem.similar",
                "chem.substructure", "chem.cluster", "chem.rank", "chem.lit")
# What the router prompt lists: the five recipes cover identity, descriptors, logP,
# drug-likeness and similarity, so those tools get no line of their own. A shorter router
# prompt is the point -- the model chooses a recipe, not a tool. A molecule_question goal may
# still name any chemistry tool, so ``allows`` is unchanged; only the listing is narrower.
CHEM_GOAL_SOLVERS = ("chem.substructure", "chem.cluster", "chem.rank", "chem.lit")
SOLVERS = ("sympy.", "numeric.evaluate", "numeric.quad", "numeric.root", "units.convert", "plot.",
           "phys.evaluate", "phys.constant", "ode.dsolve", "ode.solve_ivp", "linalg.solve", "circuit.dc",
           *DATA_SOLVERS, *CHEM_SOLVERS)

ROLES: dict[str, Role] = {
    # The router's tools are only the ones a delegated route may name in a goal: a maths or
    # physics question goes through a recipe, so no goal of its own can reach this role.
    "formalizer": Role("formalizer", "formalize.md", (*DATA_SOLVERS, *CHEM_SOLVERS),
                       shown=(*DATA_SOLVERS, *CHEM_GOAL_SOLVERS)),
    "controller": Role("controller", "controller.md", SOLVERS),
    "algebra": Role("algebra", "algebra.md", ("sympy.",)),
    "numeric": Role("numeric", "numeric.md", ("numeric.evaluate", "numeric.quad", "numeric.root", "units.convert")),
    "physics": Role("physics", "physics.md", ("phys.", "ode.", "units.", "sympy.", "numeric.", "plot.")),
    "circuits": Role("circuits", "circuits.md", ("circuit.", "linalg.", "ode.", "plot.")),
    "data": Role("data", "data.md", (*DATA_SOLVERS, "plot.")),
    "chem": Role("chem", "chem.md", CHEM_SOLVERS),
}

# After a failed check the retry goes to a specialist role for the failed tool, which
# must pick a different method or tool (a repeated fingerprint is refused).
RETRY_ROLE_BY_PREFIX = (("sympy.", "algebra"), ("phys.", "physics"), ("units.", "physics"), ("ode.", "physics"),
                        ("circuit.", "circuits"), ("linalg.", "circuits"), ("data.", "data"), ("stats.", "data"),
                        ("chem.", "chem"))


def retry_role(tool: str) -> str:
    return next((role for prefix, role in RETRY_ROLE_BY_PREFIX if tool.startswith(prefix)), "numeric")
