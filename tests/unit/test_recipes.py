"""The recipe library, the router's contract, and the display rules.

Every test here is one of the failures the live checks hit on qwen2.5, qwen3 8B and 14B,
mistral-nemo and phi4, turned into something that cannot come back. The failures were all the
same shape: the schema could not hold what the question contained, so no reply could pass.
"""
from __future__ import annotations

import json
import re

import pytest

from sciai.controller import recipes as R
from sciai.graph.model import result_to_text, show_number
from sciai.llm.actions import ACTION_SCHEMA, ActionError, check_answer_template, check_question
from sciai.llm.roles import ROLES
from sciai.tools.units_tools import check_target_unit_fn

EXAMPLE = re.compile(r'->\s*(\{.*\})\s*$')


# ------------------------------------------------------------- the router's contract
def test_the_router_enum_is_the_whole_of_the_model_s_planning_freedom():
    from sciai.llm.actions import action_variant

    assert R.ROUTES == (*R.RECIPES, R.DATASET, R.MOLECULE, R.NONE)
    formalize = action_variant("formalize")
    assert formalize["properties"]["recipe"]["enum"] == list(R.ROUTES)
    # The router asks the model for the two things only it can decide, and nothing else: which
    # recipe, and the slot values read off the question. The statement is code's job now.
    assert formalize["required"] == ["action", "recipe", "slots"]
    # Slots are flat and scalar: a nested object is exactly what the models got wrong.
    slots = formalize["properties"]["slots"]
    assert slots["additionalProperties"] == {"type": ["string", "number", "boolean"]}
    assert ACTION_SCHEMA["anyOf"][0] == formalize


def test_the_nine_recipes_yeri_asked_for_exist():
    assert set(R.RECIPES) - set(R.CHEM_RECIPES) == {
        "unit_convert", "definite_integral", "solve_equation", "area_between_curves", "ode_ivp",
        "projectile_range", "photon_energy", "rc_discharge", "series_parallel_current"}


def test_the_five_chem_recipes_yeri_asked_for_exist():
    """The chem run of 2026-10-07: molecule questions went to physics recipes, so what a
    molecule question may become is a closed list too."""
    assert R.CHEM_RECIPES == ("chem_identity", "chem_descriptors", "chem_logp", "chem_druglike",
                              "chem_similarity")
    assert set(R.CHEM_RECIPES) <= set(R.RECIPES)
    for name in R.CHEM_RECIPES:
        assert R.RECIPES[name].problem_type == "chem"   # no modelling checklist to tick


def test_every_recipe_has_two_examples_that_actually_fill_it():
    """An example the recipe cannot accept teaches the model to write something invalid."""
    for recipe in R.RECIPES.values():
        assert len(recipe.examples) >= 2, recipe.name
        for example in recipe.examples:
            m = EXAMPLE.search(example)
            assert m, (recipe.name, example)
            plan = recipe.plan(json.loads(m.group(1)))
            built = [step(plan.values, [{"kind": "expr", "value": "exp(x)"}])
                     for step in plan.steps]
            assert any(b is not None for b in built), (recipe.name, example)


def test_every_recipe_names_a_problem_type_that_exists():
    from sciai.domains.physics.assumptions import DEFAULTS, NO_CHECKLIST

    for recipe in R.RECIPES.values():
        assert recipe.problem_type in (*DEFAULTS, *NO_CHECKLIST), recipe.name


def test_the_prompt_lists_every_recipe_and_stays_short():
    block = R.prompt_block()
    for recipe in R.RECIPES.values():
        assert f"{recipe.name}(" in block
    # Short enough to leave the context to the question: well under a thousand words.
    assert len(ROLES["formalizer"].system_prompt().split()) < 1200


def test_out_of_scope_says_so_and_lists_what_there_is():
    text = R.out_of_scope("write me a limerick")
    assert "does not fit any of them" in text
    for recipe in R.RECIPES.values():
        assert recipe.name in text


# ----------------------------------------------- expressions belong in expression slots
@pytest.mark.parametrize("recipe, slots", [
    # the four slots the live checks named, each of which used to be answered with
    # 'a quantity is {"value": number, "unit": "m/s"}' or "unknown kind"
    ("definite_integral", {"integrand": "x**2*exp(-x)", "lower": "0", "upper": "1"}),
    ("ode_ivp", {"dy_dx": "6*x**2 - 4*x", "y0": "1", "at": "0", "evaluate_at": "2"}),
    ("area_between_curves", {"curve1": "x**3 - 3*x", "curve2": "x"}),
    ("solve_equation", {"equation": "x**2 - 5*x + 6 = 0", "var": "x"}),
])
def test_an_expression_in_an_expression_slot_is_accepted(recipe, slots):
    plan = R.RECIPES[recipe].plan(slots)
    assert plan.values  # no QuantityError, no "unknown kind": the slot holds an expression


def test_an_expression_slot_rejects_a_quantity_and_says_what_it_takes():
    with pytest.raises(R.SlotError) as exc:
        R.RECIPES["definite_integral"].plan({"integrand": {"value": 2, "unit": "m"},
                                             "lower": "0", "upper": "1"})
    assert "not an expression I can read" in str(exc.value)


def test_a_derivative_slot_takes_y_or_y_of_x_and_the_label_in_front():
    for text in ("-2*y", "-2*y(x)", "y' = -2*y", "dy/dx = -2*y"):
        plan = R.RECIPES["ode_ivp"].plan({"dy_dx": text, "y0": "5", "at": "0"})
        tool, args, _ = plan.steps[0](plan.values, [])
        assert tool == "ode.dsolve"
        assert args["equation"] == "Derivative(y(x), x) - (-2*y(x))"


# -------------------------------------------------------- numeric strings are coerced
@pytest.mark.parametrize("raw, expected", [
    ("20", 20.0), (20, 20.0), (20.5, 20.5), ("20.5", 20.5), ("-3", -3.0), ("1e-6", 1e-6),
    ("1,000", 1000.0), ("3/4", 0.75), ("2*pi", pytest.approx(6.283185, rel=1e-5)),
    ("−20", -20.0),  # a unicode minus sign
])
def test_a_number_slot_coerces_rather_than_re_prompting(raw, expected):
    plan = R.RECIPES["unit_convert"].plan({"value": raw, "from_unit": "m", "to_unit": "km"})
    assert plan.values["value"] == expected


def test_a_number_with_its_unit_in_one_slot_fills_both():
    """"20 m/s" in a number slot is both halves of a quantity in one place, not a type error."""
    plan = R.RECIPES["projectile_range"].plan({"v0": "20 m/s", "angle": "30 deg"})
    assert plan.values["v0"] == 20.0 and plan.values["v0_unit"] == "m/s"
    assert plan.values["angle"] == 30.0 and plan.values["angle_unit"] == "deg"


def test_only_a_real_type_error_is_fed_back_to_the_model():
    with pytest.raises(R.SlotError) as exc:
        R.RECIPES["unit_convert"].plan({"value": "twenty", "from_unit": "m", "to_unit": "km"})
    assert "must be a plain number" in str(exc.value)
    with pytest.raises(R.SlotError):
        R.RECIPES["unit_convert"].plan({"value": "1", "from_unit": "bananas", "to_unit": "km"})


# ------------------------------------------------------------------- missing slots
def test_a_missing_required_slot_asks_the_user_not_the_model():
    with pytest.raises(R.MissingSlots) as exc:
        R.RECIPES["definite_integral"].plan({"integrand": "x**2"})
    question = exc.value.question()
    assert question == "I need the lower bound and the upper bound. What are they?"
    check_question(question)  # and it is a question ask_user is allowed to put


def test_a_default_is_never_asked_about():
    plan = R.RECIPES["projectile_range"].plan({"v0": "20", "angle": "30"})
    assert plan.values["v0_unit"] == "m/s" and plan.values["to_unit"] == "m"
    assert plan.supplied == {"v0", "angle"}  # the defaults are this code's, not the model's


def test_every_recipe_s_question_is_one_ask_user_may_put():
    for name, recipe in R.RECIPES.items():
        needed = tuple(s for s in recipe.slots if s.required and s.default is None)
        for i in range(1, len(needed) + 1):
            check_question(R.MissingSlots(name, needed[:i]).question())


@pytest.mark.parametrize("question", [
    "What formula should I use for the range?",
    "What is the value of g?",
    "How many metres are there in a mile?",
    "Which method would you like me to use?",
    "What is the conversion factor between mph and m/s?",
])
def test_ask_user_may_not_ask_for_what_the_system_holds(question):
    with pytest.raises(ActionError) as exc:
        check_question(question)
    assert "may not ask about that" in str(exc.value)


@pytest.mark.parametrize("question", [
    "What is the launch angle?", "Which wavelength did you mean?",
    "I need the resistance and the capacitance. What are they?",
])
def test_ask_user_may_ask_for_a_value_the_question_left_out(question):
    check_question(question)


# --------------------------------------------------- the requested unit is a slot
def test_the_requested_unit_is_a_required_slot_of_every_quantity_recipe():
    for name in ("projectile_range", "photon_energy", "rc_discharge", "unit_convert",
                 "series_parallel_current"):
        slot = R.RECIPES[name].slot("to_unit")
        assert slot is not None and slot.kind == "unit", name


@pytest.mark.parametrize("target, got, outcome", [
    ("m/s", "m/s", "pass"),
    ("m/s", "meter/second", "pass"),       # the same unit, spelled out
    ("m/s", "m/h", "fail"),                # mistral-nemo's 96560.64 m/h for a question in m/s
    ("J", "1 / m * J", "fail"),            # the photon answered "1 / m J"
    ("J", "J", "pass"),
    ("K", "degC", "fail"),                 # the right dimension is not the right unit
    ("m", "m", "pass"),
])
def test_a_result_in_another_unit_fails_the_check(target, got, outcome):
    out = check_target_unit_fn({"target_unit": target, "unit": got})["result"]
    assert out["outcome"] == outcome, out


def test_a_wrong_dimension_and_a_wrong_unit_are_told_apart():
    wrong_dim = check_target_unit_fn({"target_unit": "J", "unit": "J / m"})["result"]
    assert "want_dims" in wrong_dim and wrong_dim["outcome"] == "fail"
    wrong_unit = check_target_unit_fn({"target_unit": "m/s", "unit": "m/h"})["result"]
    assert "want_dims" not in wrong_unit
    assert "answers a different question" in wrong_unit["reason"]


def test_the_target_unit_check_is_required_and_must_pass_everywhere_it_applies():
    from sciai.tools.registry import all_tools, load_builtin_tools
    from sciai.verify.checks import PLANS

    load_builtin_tools()
    for spec in all_tools():
        if spec.kind != "solver" or "to_unit" not in spec.schema.get("properties", {}):
            continue
        plans = [p for p in PLANS.get(spec.name, []) if p.checker == "units.check_target"]
        assert plans, f"{spec.name} takes a target unit but nothing checks it"
        assert plans[0].required and plans[0].must_pass, spec.name


# -------------------------------------------------------------------- the guards
def test_an_area_that_is_not_positive_is_refused_by_the_solver():
    from sciai.tools.formula_tools import area_between_fn

    with pytest.raises(ValueError, match="not a positive area"):
        area_between_fn({"curve1": "x**2", "curve2": "2*x", "var": "x", "lower": "0", "upper": "0"})
    with pytest.raises(ValueError, match="fewer than two points"):
        area_between_fn({"curve1": "x", "curve2": "x", "var": "x"})


def test_the_area_check_refuses_a_negative_or_zero_area_and_recomputes_the_rest():
    from sciai.tools.formula_tools import area_between_fn, check_area_between_fn

    args = {"curve1": "x**3 - 3*x", "curve2": "x", "var": "x"}
    result = area_between_fn(args)["result"]
    assert result["value"] == pytest.approx(8.0, rel=1e-9)
    good = check_area_between_fn({**args, "area": result["value"],
                                  "crossings": result["crossings"]})["result"]
    assert good["outcome"] == "pass" and good["quadrature"] == pytest.approx(8.0, rel=1e-4)
    for bad in (-8.0, 0.0):
        out = check_area_between_fn({**args, "area": bad,
                                     "crossings": result["crossings"]})["result"]
        assert out["outcome"] == "fail", bad


def test_the_ode_residual_check_is_also_the_initial_condition_check():
    from sciai.verify.checks import PLANS

    residual = next(p for p in PLANS["ode.dsolve"] if p.method == "residual")
    assert residual.required and residual.must_pass

    from sciai.tools.ode_tools import check_residual_fn

    args = {"equation": "Derivative(y(x), x) - (-2*y(x))", "func": "y", "var": "x",
            "ics": [{"at": "0", "value": "5", "order": 0}]}
    assert check_residual_fn({**args, "solution": "5*exp(-2*x)"})["result"]["outcome"] == "pass"
    # solves the equation, but not this problem: y(0) = 3, not 5
    assert check_residual_fn({**args, "solution": "3*exp(-2*x)"})["result"]["outcome"] == "fail"


def test_a_projectile_range_is_checked_by_simulating_the_motion():
    from sciai.tools.formula_tools import check_projectile_range_fn, projectile_range_fn

    args = {"v0": {"value": 20, "unit": "m/s"}, "angle": {"value": 30, "unit": "deg"}}
    si = projectile_range_fn(args)["result"]["si_value"]
    good = check_projectile_range_fn({**args, "si_value": si})["result"]
    assert good["outcome"] == "pass" and good["method"] == "numeric trajectory"
    # sin(30 radians) instead of sin(30 degrees) would give this; the simulation catches it
    bad = check_projectile_range_fn({**args, "si_value": -12.4})["result"]
    assert bad["outcome"] == "fail"


# ------------------------------------------------------------------- the display
@pytest.mark.parametrize("value, shown", [
    (35.32400580358996, "35.32"), (3.9728917142978567e-19, "3.973e-19"), (8.0, "8"),
    (0.6766764161830635, "0.6767"), (0.0, "0"), (26.8224, "26.82"), (-12.4, "-12.4"),
])
def test_a_value_is_shown_to_four_significant_figures(value, shown):
    assert show_number(value) == shown


def test_a_unit_is_written_once():
    assert result_to_text({"kind": "quantity", "value": 35.324, "unit": "m"}) == "35.32 m"
    assert result_to_text({"kind": "number", "value": 26.8224, "units": "m/s"}) == "26.82 m/s"
    # units.convert shows the unit the question asked for, not pint's spelling of it
    from sciai.tools.units_tools import convert_fn

    out = convert_fn({"value": 90, "from_unit": "km/h", "to_unit": "m/s"})
    assert out["result"]["units"] == "m/s" and out["meta"]["pint_unit"] == "meter / second"


def test_no_answer_sentence_writes_a_unit_itself():
    """"35.3 m meters" came from a sentence naming the unit next to a node that renders it."""
    for recipe in R.RECIPES.values():
        slots = {s.name: 1.0 if s.kind in ("number", "numbers") else "x" for s in recipe.slots}
        sentence = recipe.answer({**slots, "series": [1.0], "parallel": []}, ["n2", "n3"])
        assert re.search(r"\{\{n\d+\}\}", sentence), recipe.name
        stripped = re.sub(r"\{\{n\d+\}\}", "", sentence)
        assert not re.search(r"\b(m|s|kg|V|A|J|K|ohm|m/s|meters?|volts?|joules?|amps?)\b",
                             stripped), (recipe.name, sentence)


# ----------------------------------------------------- the answer template's digits
def test_the_question_s_own_number_may_be_written_in_the_answer():
    """"y(2) = {{n5}}" was refused outright, and no wording would have passed."""
    assert check_answer_template("y(2) = {{n5}}", ["n5"],
                                 "Solve dy/dx = -2*y, y(0) = 5; find y(2).") == ["n5"]


def test_a_number_the_question_never_had_is_still_refused():
    with pytest.raises(ActionError) as exc:
        check_answer_template("The answer is 42: {{n1}}", ["n1"], "what is the integral?")
    assert "may not contain the number(s) 42" in str(exc.value)


def test_a_template_still_needs_a_placeholder():
    with pytest.raises(ActionError):
        check_answer_template("The answer is obvious.", ["n1"], "")
