"""The recipe library: the shapes a question may be formalized into.

Why this exists. Formalization used to be open ended: the model wrote a free-form goal tool
call and described the question's values as quantity objects in ``givens``. Two things broke,
and they broke worse on larger models, because a larger model writes more:

- A "given" that is not a number has nowhere to go. The integrand of an integral, the right
  hand side of a differential equation and the curves of an area problem are expressions, and
  the quantity validator answered ``a quantity is {"value": number, "unit": "m/s"}`` or
  ``unknown kind``. Neither is something a model can repair by trying again, because the slot
  it was asked to fill cannot hold what the question contains.
- An open-ended goal means the model picks the formula, so the answer is only ever as good as
  the model's memory of that formula.

A recipe closes both. Each recipe is a fixed list of named slots; each slot has one kind; and
the recipe -- not the model -- fixes the physical kind of every numeric slot and the tool calls
that answer the question. The model's whole job is to name one recipe and fill its slots with
strings. Code coerces the strings, refuses the ones that are not what the slot is for, asks the
user when a required slot is genuinely missing, builds the tool calls, and writes the answer
sentence. Every number in the answer still comes from a tool, and every tool call still has its
independent check: the recipe decides *which* call to make, never what it returns.

Three router outcomes are not recipes. ``none`` is out of scope. ``dataset_question`` and
``molecule_question`` hand the question to the formalize path Phase 3 and Phase 4 are built on,
whose own tools, declines and hypothesis rules are unchanged by any of this.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Callable

from sciai.tools.parsing import ParseError, parse, parse_relation

Values = dict[str, Any]
# (tool, args, title); a step is built from the slot values and the results of the steps
# before it, so a second call can use the first call's answer without a model call.
StepFn = Callable[[Values, "list[dict[str, Any]]"], "tuple[str, dict[str, Any], str] | None"]
AnswerFn = Callable[[Values, "list[str]"], str]

IDENT = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,29}$")
UNIT_TEXT = re.compile(r"^[A-Za-z0-9_ */^().\-]+$")
_THOUSANDS = re.compile(r"(?<=\d),(?=\d{3}(?!\d))")
_NUMBER_THEN_REST = re.compile(r"^([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)\s*(.*)$")
# "y' = x*y", "dy/dx = x*y", "f(x) = x**2": the model labels the expression it was asked for.
_LABELLED = re.compile(r"^\s*(?:d\s*[A-Za-z]\w*\s*/\s*d\s*[A-Za-z]\w*|[A-Za-z]\w*\s*'+|"
                       r"[A-Za-z]\w*\s*\([^()]*\)|[A-Za-z]\w*)\s*=(?!=)\s*(?P<rhs>.+)$")
# Offset temperature units: a difference in them is pint's delta_ form, an absolute is not.
OFFSET_UNITS = {"degc", "celsius", "degree_celsius", "degf", "fahrenheit", "degree_fahrenheit"}

MAX_RESISTORS = 8


class SlotError(ValueError):
    """A slot holds something it cannot hold. The message is fed back to the model once."""


@dataclass(frozen=True)
class MissingSlots(Exception):
    """Required slots the question itself does not contain. The user is asked, not the model."""

    recipe: str
    slots: tuple["Slot", ...]

    def question(self) -> str:
        if len(self.slots) == 1:
            return f"What is {self.slots[0].about}?"
        # Slots that are individually optional only got here as a group of which one is
        # needed, so the question asks for one of them rather than for all of them.
        joiner = " or " if all(not s.required for s in self.slots) else " and "
        return "I need " + ", ".join(s.about for s in self.slots[:-1]) + \
               joiner + self.slots[-1].about + ". What are they?"


@dataclass(frozen=True)
class Slot:
    name: str
    kind: str            # number | unit | expr | equation | bound | var | choice | numbers
    about: str           # one phrase, used in the prompt and in the question put to the user
    required: bool = True
    default: Any = None
    choices: tuple[str, ...] = ()

    def line(self) -> str:
        shown = self.name if self.required else f"{self.name}?"
        if self.kind == "choice":
            return f"{shown}={'|'.join(self.choices)}"
        return f"{shown}:{self.kind}"


# ------------------------------------------------------------------- coercion
def _strip_label(text: str) -> str:
    m = _LABELLED.match(text)
    return m.group("rhs").strip() if m else text


def _number(slot: Slot, raw: Any) -> float:
    if isinstance(raw, bool):
        raise SlotError(f"{slot.name} must be a number ({slot.about}), not true/false")
    if isinstance(raw, (int, float)):
        value = float(raw)
    else:
        text = _THOUSANDS.sub("", str(raw).strip().replace("−", "-")).strip()
        try:
            value = float(text)
        except ValueError:
            # "3/4", "2*pi", "10**3": arithmetic a question can legitimately state. The
            # restricted parser evaluates it; anything with a symbol in it is not a number.
            try:
                value = float(parse(text).evalf())
            except (ParseError, TypeError, ValueError, AttributeError):
                raise SlotError(f"{slot.name} must be a plain number ({slot.about}), not "
                                f"{str(raw)[:60]!r}; units belong in their own slot") from None
    if not math.isfinite(value):
        raise SlotError(f"{slot.name} must be a finite number ({slot.about})")
    return value


def _unit(slot: Slot, raw: Any) -> str:
    text = str(raw).strip()
    if not text or not UNIT_TEXT.match(text):
        raise SlotError(f"{slot.name} must be a unit like m/s, kohm or degC ({slot.about}), not "
                        f"{text[:60]!r}")
    from sciai.tools.units_tools import ureg

    try:
        ureg().parse_units(text.replace("^", "**"))
    except Exception:  # noqa: BLE001 - pint raises several unrelated types for bad units
        raise SlotError(f"{slot.name}: {text!r} is not a unit I know ({slot.about})") from None
    return text


def _expr(slot: Slot, raw: Any) -> str:
    text = _strip_label(str(raw).strip()).replace("^", "**")
    if not text:
        raise SlotError(f"{slot.name} is empty; it holds {slot.about}")
    if "=" in text:
        raise SlotError(f"{slot.name} is an expression, not an equation ({slot.about}): write "
                        f"only the right-hand side")
    try:
        parse(text)
    except ParseError as exc:
        raise SlotError(f"{slot.name} ({slot.about}) is not an expression I can read: {exc}") from None
    return text


_APPLIED = re.compile(r"\b([A-Za-z]\w*)\s*\(")


def _derivative(slot: Slot, raw: Any) -> str:
    """The right-hand side of a first-order ODE.

    A model names the unknown function either ``y`` or ``y(x)``, and which one it writes is not
    worth a re-prompt, so both are accepted here and ``_dsolve_args`` normalizes it to the one
    form the solver takes. Accepting only one of them is precisely the kind of schema that made
    a correct answer unreachable.
    """
    from sciai.tools.parsing import FUNCTIONS

    text = _strip_label(str(raw).strip()).replace("^", "**")
    if not text:
        raise SlotError(f"{slot.name} is empty; it holds {slot.about}")
    if "=" in text:
        raise SlotError(f"{slot.name} is an expression, not an equation ({slot.about}): write "
                        f"only the right-hand side")
    declared = tuple(sorted({n for n in _APPLIED.findall(text) if n not in FUNCTIONS}))
    try:
        parse(text, functions=declared)
    except ParseError as exc:
        raise SlotError(f"{slot.name} ({slot.about}) is not an expression I can read: {exc}") from None
    return text


def _equation(slot: Slot, raw: Any) -> str:
    text = str(raw).strip().replace("^", "**")
    try:
        parse_relation(text)
    except ParseError as exc:
        raise SlotError(f"{slot.name} ({slot.about}) is not an equation I can read: {exc}") from None
    return text


def _bound(slot: Slot, raw: Any) -> str:
    text = str(raw).strip().replace("^", "**")
    try:
        parse(text)
    except ParseError as exc:
        raise SlotError(f"{slot.name} ({slot.about}) must be a number or a constant like pi or "
                        f"oo: {exc}") from None
    return text


def _var(slot: Slot, raw: Any) -> str:
    text = str(raw).strip()
    if not IDENT.match(text):
        raise SlotError(f"{slot.name} must be a single variable name like x or t ({slot.about})")
    return text


def _choice(slot: Slot, raw: Any) -> str:
    text = str(raw).strip().lower()
    if text not in slot.choices:
        raise SlotError(f"{slot.name} must be one of {', '.join(slot.choices)} ({slot.about})")
    return text


def _numbers(slot: Slot, raw: Any) -> list[float]:
    items = raw if isinstance(raw, list) else [p for p in re.split(r"[,;]", str(raw)) if p.strip()]
    if not items:
        raise SlotError(f"{slot.name} must be numbers separated by commas ({slot.about})")
    return [_number(slot, item) for item in items]


COERCE: dict[str, Callable[[Slot, Any], Any]] = {
    "number": _number, "unit": _unit, "expr": _expr, "equation": _equation, "bound": _bound,
    "var": _var, "choice": _choice, "numbers": _numbers, "derivative": _derivative,
}


# -------------------------------------------------------------------- recipes
@dataclass(frozen=True)
class Plan:
    recipe: str
    steps: tuple[StepFn, ...]
    answer: AnswerFn
    values: Values
    # Slots the model actually filled, as opposed to the ones code defaulted. Only the former
    # have to be in the question: a default is a constant this code chose, not a claim about
    # the problem, so it is a source rather than something to source.
    supplied: frozenset[str] = frozenset()


@dataclass(frozen=True)
class Recipe:
    name: str
    about: str
    slots: tuple[Slot, ...]
    steps: tuple[StepFn, ...]
    answer: AnswerFn
    examples: tuple[str, ...] = ()
    # Groups of slots of which at least one must be filled (a circuit needs resistors
    # somewhere, in series or in parallel, but not necessarily in both).
    require_any: tuple[tuple[str, ...], ...] = ()
    # The default-assumptions checklist this recipe's problems carry (Phase 2). The recipe
    # names it, not the model: the type of problem is exactly what choosing the recipe decided.
    # "math" means there is nothing to assume.
    problem_type: str = "math"

    def slot(self, name: str) -> Slot | None:
        return next((s for s in self.slots if s.name == name), None)

    def line(self) -> str:
        return f"{self.name}({', '.join(s.line() for s in self.slots)}) - {self.about}"

    def fill(self, raw: dict[str, Any]) -> Values:
        """Coerce what the model wrote into slot values.

        Order matters. ``_split_units`` runs first because a model that writes "20 m/s" into a
        number slot has given both halves of a quantity and simply put them in one place; that
        is not a type error and must not cost a re-prompt. Then every required slot that is
        still empty is collected, and the *user* is asked for those, because no amount of
        re-prompting will make the model invent a number the question never contained. A value
        that is present but wrong for its slot is the only real type error, and only that is
        fed back to the model.
        """
        raw = self._split_units({k: v for k, v in (raw or {}).items() if v not in (None, "")})
        values: Values = {}
        missing: list[Slot] = []
        for slot in self.slots:
            if slot.name not in raw:
                if slot.default is not None:
                    values[slot.name] = slot.default
                elif slot.required:
                    missing.append(slot)
                continue
            values[slot.name] = COERCE[slot.kind](slot, raw[slot.name])
        for group in self.require_any:
            if not any(values.get(n) for n in group):
                missing += [s for s in self.slots if s.name in group]
        if missing:
            raise MissingSlots(self.name, tuple(dict.fromkeys(missing)))
        unknown = sorted(set(raw) - {s.name for s in self.slots})
        if unknown:
            raise SlotError(f"{self.name} has no slot called {', '.join(unknown)}; its slots are "
                            f"{', '.join(s.name for s in self.slots)}")
        return values

    def _split_units(self, raw: dict[str, Any]) -> dict[str, Any]:
        """"20 m/s" in a number slot fills that slot and its unit slot."""
        out = dict(raw)
        for slot in self.slots:
            unit_slot = self.slot(f"{slot.name}_unit")
            if slot.kind != "number" or unit_slot is None or not isinstance(out.get(slot.name), str):
                continue
            m = _NUMBER_THEN_REST.match(out[slot.name].strip())
            if m is None or not m.group(2):
                continue
            out[slot.name] = m.group(1)
            out.setdefault(unit_slot.name, m.group(2).strip())
        return out

    def plan(self, raw: dict[str, Any]) -> Plan:
        values = self.fill(raw)
        supplied = frozenset(self._split_units(
            {k: v for k, v in (raw or {}).items() if v not in (None, "")})) & set(values)
        return Plan(self.name, self.steps, self.answer, values, frozenset(supplied))

    def sources(self, values: Values) -> dict[str, dict[str, Any]]:
        """The slot values as the problem's sources, which is what ``givens`` was for.

        Provenance does not change because formalization did: a number in a tool call still has
        to come from the question, a previous result or one of these. A slot is the recipe's
        name for a value the question gave, so each one is recorded here with its unit, and the
        router's own ``unsourced`` check (in the controller) marks any whose number the question
        does not actually contain.
        """
        out: dict[str, dict[str, Any]] = {}
        for slot in self.slots:
            if slot.name not in values or slot.name.endswith("_unit"):
                continue
            unit = values.get(f"{slot.name}_unit") or ""
            value = values[slot.name]
            if slot.kind == "numbers":
                # one source per element, so each matches the quantity the recipe builds from it
                for i, item in enumerate(value, start=1):
                    out[f"{slot.name}{i}"] = {"value": item, "unit": unit}
                continue
            out[slot.name] = {"value": value, "unit": unit}
        return out


# ------------------------------------------------------------------- builders
def num_text(value: float) -> str:
    """A number as a tool argument: 0 rather than 0.0, and no exponent for ordinary sizes."""
    return f"{value:.17g}" if abs(value) >= 1e16 else f"{value:g}"


def _quantity(values: Values, name: str, kind: str) -> dict[str, Any]:
    return {"value": values[name], "unit": values[f"{name}_unit"], "kind": kind}


def _one(tool: str, title: str, args: Callable[[Values], dict[str, Any]]) -> StepFn:
    def step(values: Values, _prior: list[dict[str, Any]]) -> tuple[str, dict[str, Any], str]:
        return tool, args(values), title
    return step


def placeholder(handle: str) -> str:
    return "{{" + handle + "}}"


def _say(text: str) -> AnswerFn:
    """An answer sentence that names the quantity and never its unit: the node renders the
    unit, and writing it in the sentence as well is what produced "35.3 m meters"."""
    def answer(values: Values, handles: list[str]) -> str:
        del values
        return text.replace("{h}", placeholder(handles[-1]))
    return answer


# -------------------------------------------------------------- unit_convert
def _convert_args(values: Values) -> dict[str, Any]:
    difference = values.get("kind") == "temperature_difference"
    return {"value": values["value"],
            "from_unit": _delta(values["from_unit"], difference),
            "to_unit": _delta(values["to_unit"], difference)}


def _delta(unit: str, difference: bool) -> str:
    """A span of degrees Celsius is 1.8 degrees Fahrenheit; a temperature of 1 degC is 33.8 degF.
    pint keeps the two apart with its delta_ units, and only the question knows which is meant."""
    return f"delta_{unit}" if difference and unit.strip().lower() in OFFSET_UNITS else unit


# ------------------------------------------------------------------- ode_ivp
def _as_function(expr: str, func: str, var: str) -> str:
    """``-2*y`` -> ``-2*y(x)``. The solver needs the applied form; a question never writes it."""
    return re.sub(rf"\b{re.escape(func)}\b(?!\s*\()", f"{func}({var})", expr)


def _dsolve_args(values: Values) -> dict[str, Any]:
    func, var = values["func"], values["var"]
    rhs = _as_function(values["dy_dx"], func, var)
    return {"equation": f"Derivative({func}({var}), {var}) - ({rhs})",
            "func": func, "var": var,
            "ics": [{"at": num_text(values["at"]), "value": num_text(values["y0"]),
                      "order": 0}]}


def _ode_evaluate(values: Values, prior: list[dict[str, Any]]) -> tuple[str, dict[str, Any], str] | None:
    """Only when the question asks for the solution at a point. The solution comes from the
    first step's result, so this costs no model call."""
    if values.get("evaluate_at") is None or not prior or prior[0].get("kind") != "expr":
        return None
    return ("numeric.evaluate",
            {"expr": prior[0]["value"],
             "values": {values["var"]: num_text(values["evaluate_at"])}},
            "the solution at the requested point")


def _ode_answer(values: Values, handles: list[str]) -> str:
    if values.get("evaluate_at") is not None and len(handles) > 1:
        return f"{values['func']}({values['evaluate_at']:g}) = {placeholder(handles[-1])}."
    return f"The solution is {values['func']}({values['var']}) = {placeholder(handles[0])}."


# ------------------------------------------------------ series_parallel_current
def _netlist(values: Values) -> dict[str, Any]:
    """One voltage source, a series chain and a parallel group, wired as the question says.

    Built here rather than by the model: a netlist is where a wrong guess is invisible, and
    these are the shapes this recipe claims to cover. ``circuit.dc`` then applies its own three
    independent checks (Kirchhoff, power balance, series/parallel reduction) to what this
    builds, so a netlist that does not match the question is still a netlist solved correctly
    -- which is why the slots are named after what the question says, not after nodes.
    """
    series = list(values.get("series") or [])
    parallel = list(values.get("parallel") or [])
    if len(series) + len(parallel) > MAX_RESISTORS:
        raise SlotError(f"this recipe handles up to {MAX_RESISTORS} resistors")
    if any(r <= 0 for r in series + parallel):
        raise SlotError("a resistance must be positive")
    elements = [{"name": "V1", "type": "V", "n1": "a", "n2": "0",
                 "value": {"value": values["voltage"], "unit": values["voltage_unit"],
                           "kind": "voltage"}}]
    outputs: list[str] = []
    # The series chain runs from the source's node towards the parallel group (or to ground).
    nodes = ["a"] + [f"m{i}" for i in range(1, len(series))] + ["p" if parallel else "0"]
    for i, r in enumerate(series, start=1):
        elements.append({"name": f"Rs{i}", "type": "R", "n1": nodes[i - 1], "n2": nodes[i],
                         "value": {"value": r, "unit": values["series_unit"],
                                   "kind": "resistance"}})
        outputs.append(f"I(Rs{i})")
    top = "p" if series and parallel else "a"
    for i, r in enumerate(parallel, start=1):
        elements.append({"name": f"Rp{i}", "type": "R", "n1": top, "n2": "0",
                         "value": {"value": r, "unit": values["parallel_unit"],
                                   "kind": "resistance"}})
        outputs.append(f"I(Rp{i})")
    return {"netlist": {"elements": elements, "ground": "0"}, "outputs": outputs}


def _current_answer(values: Values, handles: list[str]) -> str:
    """The sentence names what the listed currents are, and never a number.

    The source element's own current is left out on purpose: its sign is a wiring convention,
    and a leading "-0.06 A" reads as a wrong answer. A series resistor carries the whole
    current from the source, so when the question has one, that is the total.
    """
    where = ("through the circuit" if values.get("series") and not values.get("parallel")
             else "through each resistor" if not values.get("series")
             else "in each resistor (the series resistors carry the whole current "
                  "from the source)")
    return f"The current {where} is {placeholder(handles[0])}."


# ------------------------------------------------------------------ the library
RECIPES: dict[str, Recipe] = {}


def _add(recipe: Recipe) -> None:
    RECIPES[recipe.name] = recipe


_add(Recipe(
    name="unit_convert",
    about="convert one value from one unit to another, including degC and degF",
    slots=(Slot("value", "number", "the number to convert"),
           Slot("from_unit", "unit", "the unit it is given in"),
           Slot("to_unit", "unit", "the unit the answer must be in"),
           Slot("kind", "choice", "whether a temperature is a reading or a difference",
                required=False, choices=("absolute_temperature", "temperature_difference"))),
    steps=(_one("units.convert", "the conversion", _convert_args),),
    answer=_say("That is {h}."),
    examples=('"Convert 90 km/h to m/s" -> {"value":"90","from_unit":"km/h","to_unit":"m/s"}',
              '"What is 20 degC in degF?" -> {"value":"20","from_unit":"degC","to_unit":"degF",'
              '"kind":"absolute_temperature"}'),
))

_add(Recipe(
    name="definite_integral",
    about="the definite integral of one expression between two bounds",
    slots=(Slot("integrand", "expr", "the expression being integrated"),
           Slot("var", "var", "the variable of integration", default="x"),
           Slot("lower", "bound", "the lower bound"),
           Slot("upper", "bound", "the upper bound")),
    steps=(_one("sympy.integrate", "the integral",
                lambda v: {"expr": v["integrand"], "var": v["var"], "lower": v["lower"],
                           "upper": v["upper"]}),),
    answer=_say("The integral is {h}."),
    examples=('"Integrate x**2*exp(-x) from 0 to 1" -> {"integrand":"x**2*exp(-x)","var":"x",'
              '"lower":"0","upper":"1"}',
              '"Area under sin(t) from 0 to pi" -> {"integrand":"sin(t)","var":"t","lower":"0",'
              '"upper":"pi"}'),
))

_add(Recipe(
    name="solve_equation",
    about="solve one equation for one unknown",
    slots=(Slot("equation", "equation", "the equation, as lhs = rhs"),
           Slot("var", "var", "the unknown to solve for")),
    steps=(_one("sympy.solve", "the solutions",
                lambda v: {"equation": v["equation"], "var": v["var"]}),),
    answer=lambda v, h: f"{v['var']} = {placeholder(h[0])}.",
    examples=('"Solve x**2 - 5*x + 6 = 0" -> {"equation":"x**2 - 5*x + 6 = 0","var":"x"}',
              '"Find t where 3*t + 7 = 19" -> {"equation":"3*t + 7 = 19","var":"t"}'),
))

_add(Recipe(
    name="area_between_curves",
    about="the area enclosed between two curves (their crossings are the bounds unless given)",
    slots=(Slot("curve1", "expr", "the first curve"),
           Slot("curve2", "expr", "the second curve"),
           Slot("var", "var", "the variable", default="x"),
           Slot("lower", "bound", "the left bound, if the question gives one", required=False),
           Slot("upper", "bound", "the right bound, if the question gives one", required=False)),
    steps=(_one("calc.area_between", "the area",
                lambda v: {"curve1": v["curve1"], "curve2": v["curve2"], "var": v["var"],
                           **({"lower": v["lower"]} if v.get("lower") else {}),
                           **({"upper": v["upper"]} if v.get("upper") else {})}),),
    answer=_say("The area between the curves is {h}."),
    examples=('"Area between y = x**2 and y = 2*x" -> {"curve1":"x**2","curve2":"2*x","var":"x"}',
              '"Area between x**3 - 3*x and x from -2 to 2" -> {"curve1":"x**3 - 3*x",'
              '"curve2":"x","var":"x","lower":"-2","upper":"2"}'),
))

_add(Recipe(
    name="ode_ivp",
    about="solve a first-order initial value problem, and evaluate it at a point if asked",
    slots=(Slot("dy_dx", "derivative", "the derivative's right-hand side"),
           Slot("func", "var", "the unknown function's name", default="y"),
           Slot("var", "var", "the independent variable", default="x"),
           Slot("y0", "number", "the value of the function at the initial point"),
           Slot("at", "number", "the point where that value is given", default=0.0),
           Slot("evaluate_at", "number", "the point the answer is asked for", required=False)),
    steps=(_one("ode.dsolve", "the solution", _dsolve_args), _ode_evaluate),
    answer=_ode_answer,
    problem_type="ode",
    examples=('"Solve dy/dx = x*y with y(0) = 1" -> {"dy_dx":"x*y","func":"y","var":"x",'
              '"y0":"1","at":"0"}',
              '"y\' = -2*y, y(0) = 5; find y(3)" -> {"dy_dx":"-2*y","func":"y","var":"x",'
              '"y0":"5","at":"0","evaluate_at":"3"}'),
))

_add(Recipe(
    name="projectile_range",
    about="the horizontal range of a projectile launched over level ground",
    slots=(Slot("v0", "number", "the launch speed"),
           Slot("v0_unit", "unit", "the unit of the launch speed", default="m/s"),
           Slot("angle", "number", "the launch angle"),
           Slot("angle_unit", "unit", "the unit of the angle, deg or rad", default="deg"),
           Slot("g", "number", "gravity, if the question gives it", required=False),
           Slot("g_unit", "unit", "the unit of gravity", default="m/s**2"),
           Slot("to_unit", "unit", "the unit the range must be in", default="m")),
    steps=(_one("phys.projectile_range", "the range",
                lambda v: {"v0": _quantity(v, "v0", "speed"),
                           "angle": _quantity(v, "angle", "angle"),
                           **({"g": _quantity(v, "g", "acceleration")} if v.get("g") is not None else {}),
                           "to_unit": v["to_unit"]}),),
    answer=_say("The range is {h}."),
    problem_type="projectile",
    examples=('"A ball is thrown at 20 m/s at 30 degrees; how far does it land?" -> '
              '{"v0":"20","v0_unit":"m/s","angle":"30","angle_unit":"deg","to_unit":"m"}',
              '"Range of a shell fired at 400 m/s at 45 deg, in km" -> {"v0":"400",'
              '"v0_unit":"m/s","angle":"45","angle_unit":"deg","to_unit":"km"}'),
))

_add(Recipe(
    name="photon_energy",
    about="the energy of a photon of a given wavelength",
    slots=(Slot("wavelength", "number", "the wavelength"),
           Slot("wavelength_unit", "unit", "the unit of the wavelength", default="nm"),
           Slot("to_unit", "unit", "the unit the energy must be in", default="J")),
    steps=(_one("phys.photon_energy", "the photon energy",
                lambda v: {"wavelength": _quantity(v, "wavelength", "length"),
                           "to_unit": v["to_unit"]}),),
    answer=_say("The photon energy is {h}."),
    problem_type="general",
    examples=('"Energy of a 500 nm photon" -> {"wavelength":"500","wavelength_unit":"nm",'
              '"to_unit":"J"}',
              '"A photon of wavelength 0.1 nm, in eV" -> {"wavelength":"0.1",'
              '"wavelength_unit":"nm","to_unit":"eV"}'),
))

_add(Recipe(
    name="rc_discharge",
    about="the voltage left on a discharging RC circuit after a given time",
    slots=(Slot("v0", "number", "the starting voltage"),
           Slot("v0_unit", "unit", "the unit of the starting voltage", default="V"),
           Slot("resistance", "number", "the resistance"),
           Slot("resistance_unit", "unit", "the unit of the resistance", default="ohm"),
           Slot("capacitance", "number", "the capacitance"),
           Slot("capacitance_unit", "unit", "the unit of the capacitance", default="F"),
           Slot("t", "number", "the elapsed time"),
           Slot("t_unit", "unit", "the unit of the elapsed time", default="s"),
           Slot("to_unit", "unit", "the unit the voltage must be in", default="V")),
    steps=(_one("phys.rc_discharge", "the voltage",
                lambda v: {"v0": _quantity(v, "v0", "voltage"),
                           "resistance": _quantity(v, "resistance", "resistance"),
                           "capacitance": _quantity(v, "capacitance", "capacitance"),
                           "t": _quantity(v, "t", "time"), "to_unit": v["to_unit"]}),),
    answer=_say("The voltage after that time is {h}."),
    problem_type="rc_rl_transient",
    examples=('"A 10 V capacitor, 1 kohm and 100 uF, after 0.05 s" -> {"v0":"10","v0_unit":"V",'
              '"resistance":"1","resistance_unit":"kohm","capacitance":"100",'
              '"capacitance_unit":"uF","t":"0.05","t_unit":"s","to_unit":"V"}',
              '"5 V across 2 Mohm and 1 nF after 3 ms, in mV" -> {"v0":"5","v0_unit":"V",'
              '"resistance":"2","resistance_unit":"Mohm","capacitance":"1",'
              '"capacitance_unit":"nF","t":"3","t_unit":"ms","to_unit":"mV"}'),
))

_add(Recipe(
    name="series_parallel_current",
    about="the current in a resistor network of one series chain and one parallel group",
    slots=(Slot("voltage", "number", "the source voltage"),
           Slot("voltage_unit", "unit", "the unit of the source voltage", default="V"),
           Slot("series", "numbers", "the resistances in series, separated by commas",
                required=False),
           Slot("series_unit", "unit", "the unit of the series resistances", default="ohm"),
           Slot("parallel", "numbers", "the resistances in parallel with each other, "
                "separated by commas", required=False),
           Slot("parallel_unit", "unit", "the unit of the parallel resistances", default="ohm"),
           Slot("to_unit", "unit", "the unit the current must be in", default="A")),
    steps=(_one("circuit.dc", "the circuit", _netlist),),
    answer=_current_answer,
    require_any=(("series", "parallel"),),
    problem_type="dc_circuit",
    examples=('"12 V across 100 and 220 ohm in series; what current flows?" -> '
              '{"voltage":"12","voltage_unit":"V","series":"100, 220"}',
              '"A 12 V source drives 100 ohm in series with two 200 ohm in parallel; current '
              'from the source?" -> {"voltage":"12","voltage_unit":"V","series":"100",'
              '"parallel":"200, 200"}'),
))

# Router outcomes that are not recipes.
NONE = "none"
DATASET = "dataset_question"
MOLECULE = "molecule_question"
DELEGATED = {
    DATASET: "a question about an imported dataset",
    MOLECULE: "a question about a molecule or a drug-discovery candidate",
}
ROUTES: tuple[str, ...] = (*RECIPES, *DELEGATED, NONE)


def supported() -> str:
    """One line per recipe, for the out-of-scope message."""
    return "\n".join(f"- {r.name}: {r.about}" for r in RECIPES.values())


def prompt_block() -> str:
    """The recipe list and its examples, for the router's system prompt."""
    out = []
    for r in RECIPES.values():
        out.append(r.line())
        out += [f"    {ex}" for ex in r.examples]
    return "\n".join(out)


def out_of_scope(question: str) -> str:
    del question
    return ("I can only answer questions that fit one of the recipes I have, and this one does "
            "not fit any of them. These are the ones I have:\n" + supported())


__all__ = ["DATASET", "DELEGATED", "MOLECULE", "MissingSlots", "NONE", "Plan", "RECIPES",
           "ROUTES", "Recipe", "Slot", "SlotError", "num_text", "out_of_scope", "placeholder", "prompt_block", "supported"]
