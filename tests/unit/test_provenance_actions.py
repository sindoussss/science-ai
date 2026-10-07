import pytest

from sciai.controller.provenance import numbers_in, unsourced
from sciai.llm.actions import ActionError, check_answer_template, parse_action


def test_numbers_in():
    assert numbers_in("x**2 + 3.50*y - 1e3") == {"2", "3.5", "1000"}
    assert numbers_in({"a": ["x1", "2.0"]}) == {"2"}


def test_unsourced_allows_ancestor_and_structural_numbers():
    args = {"expr": "x**2 + 7.25", "values": {"x": "3.7"}}
    assert unsourced(args, ["f(x) = x**2 + 7.25"], 10) == ["3.7"]
    assert unsourced(args, ["7.25", {"result": "3.7"}], 10) == []
    assert unsourced({"expr": "x**2 + 11"}, [], 10) == ["11"]


def test_template_rules():
    assert check_answer_template("It is {{n3}}.", ["n3"]) == ["n3"]
    with pytest.raises(ActionError):
        check_answer_template("It is {{n3}} or 4.", ["n3"])
    with pytest.raises(ActionError):
        check_answer_template("It is {{n4}}.", ["n3"])
    with pytest.raises(ActionError):
        check_answer_template("No refs.", ["n3"])


def test_parse_action():
    a = parse_action('```json\n{"action":"ask_user","question":"which x?"}\n```')
    assert a.kind == "ask_user"
    with pytest.raises(ActionError):
        parse_action("not json")
    with pytest.raises(ActionError):
        parse_action('{"action":"call_tool","tool":"sympy.diff"}')
    with pytest.raises(ActionError):
        parse_action('{"action":"finish","answer_template":"x","answer_nodes":["bad handle"]}')


def test_quantity_provenance_needs_matching_si_value_and_dimension():
    from sciai.controller.provenance import unsourced

    givens = {"givens": {"v": {"value": 60, "unit": "mph"}, "t": {"value": 2, "unit": "s"}}}
    ok = {"expr": "v*t", "values": {"v": {"value": 96.56064, "unit": "km/h"}, "t": {"value": 2, "unit": "s"}}}
    assert unsourced(ok, [givens], 10) == []
    wrong_unit = {"expr": "v*t", "values": {"v": {"value": 60, "unit": "km/h"}, "t": {"value": 2, "unit": "s"}}}
    assert unsourced(wrong_unit, [givens], 10) == ["60 km/h"]
    wrong_dim = {"expr": "v", "values": {"v": {"value": 2, "unit": "m"}}}
    assert unsourced(wrong_dim, [givens], 10) == ["2 m"]
    # A quantity result of an ancestor sources the same value in other units.
    result = {"kind": "quantity", "value": 26.8224, "unit": "m/s", "si_value": 26.8224,
              "si_unit": "meter / second", "dims": {"[length]": 1, "[time]": -1}}
    assert unsourced({"expr": "v", "values": {"v": {"value": 60, "unit": "mph"}}}, [result], 10) == []
    # Plain numbers keep the Phase 1 rule.
    assert unsourced({"expr": "x*37"}, ["question with 37"], 10) == []
    assert unsourced({"expr": "x*37"}, ["question"], 10) == ["37"]


def test_si_value_of_a_given_sources_plain_si_numbers():
    from sciai.controller.provenance import unsourced

    givens = {"givens": {"R": {"value": 1, "unit": "kohm"}, "C": {"value": 1, "unit": "uF"},
                         "t": {"value": 2, "unit": "ms"}}}
    args = {"equation": "Derivative(v(t), t) = -v(t)/(1000*1e-6)", "t_span": ["0", "0.002"]}
    assert unsourced(args, [givens], 10) == []
    assert unsourced({"t_span": ["0", "0.003"]}, [givens], 10) == ["0.003"]


# ------------------------------------------------- structure slots (2026-10-07, chem run 6/8)
CAFFEINE = "Cn1cnc2c1c(=O)n(C)c(=O)n2C"
CAFFEINE_KEKULE = "CN1C=NC2=C1C(=O)N(C)C(=O)N2C"
ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"
CAFFEINE_QUESTION = f"What are the molecular weight and TPSA of caffeine, {CAFFEINE}?"


def plan_of(recipe: str, question: str, **slots):
    from sciai.controller.recipes import RECIPES

    return RECIPES[recipe].plan(slots, question)


def test_a_rewriting_of_the_questions_own_structure_is_sourced():
    """Both failures of the 2026-10-07 chem run were this question. The router chose
    chem_descriptors correctly; the model then wrote caffeine's Kekule form instead of copying
    the aromatic SMILES the question gave. It is the same molecule, so it is the question's own
    structure, and matching text refused a right answer."""
    assert plan_of("chem_descriptors", CAFFEINE_QUESTION, structure=CAFFEINE).unsourced == ()
    assert plan_of("chem_descriptors", CAFFEINE_QUESTION, structure=CAFFEINE_KEKULE).unsourced == ()


LIBRARY = ('{"members":[{"name":"caffeine","structure":"' + CAFFEINE + '"},'
           '{"name":"aspirin","structure":"' + ASPIRIN + '"}]}')
SEVERAL = f"Which of this library is nearest to paracetamol, CC(=O)Nc1ccc(O)cc1? Use {LIBRARY}"


@pytest.mark.parametrize("structure, why", [
    ("C", "methane is a substring of almost any question carrying a SMILES"),
    ("O", "so is water"),
    ("caffeine", "a name is not a structure, whatever the question calls the molecule"),
    ("CN1C=NC2=C1C(=O)N(C)C(=O)N2", "a caffeine one atom short is a different molecule"),
])
def test_a_structure_the_question_does_not_contain_is_refused(structure, why):
    """Where the model still chooses: a question carrying several structures needs it to say
    which one is the subject, so that choice is traced. The hole matching text left open was a
    one-atom SMILES passing as the molecule asked about, because its text was in the question."""
    assert plan_of("chem_similarity", SEVERAL, structure=structure,
                   library=LIBRARY).unsourced == ("structure",), why


@pytest.mark.parametrize("written, why", [
    (CAFFEINE, "copied correctly"),
    (CAFFEINE_KEKULE, "rewritten as the Kekule form, which qwen3:8b does from memory"),
    ("CN1C=NC2=C1C(=O)N(C)C(=O)N2", "a copy one atom short"),
    (ASPIRIN, "a different molecule entirely"),
    ("C", "a single atom"),
    ("caffeine", "the name instead of the structure"),
])
def test_a_lone_structure_is_read_out_of_the_question_not_copied_by_the_model(written, why):
    """qwen3:8b answered four molecule questions correctly three runs in a row and got caffeine
    wrong every time, because a structure was the one value still being transcribed by the model
    -- 26 characters of punctuation. The question carries exactly one structure here, so that is
    its answer to "which molecule", and code reads it."""
    plan = plan_of("chem_descriptors", CAFFEINE_QUESTION, structure=written)
    assert plan.unsourced == (), why
    assert plan.values["structure"] == CAFFEINE, why
    assert plan.from_question == (() if written == CAFFEINE else ("structure",)), why


def test_every_library_member_has_to_be_in_the_question():
    library = ('{"members":[{"name":"caffeine","structure":"' + CAFFEINE + '"},'
               '{"name":"aspirin","structure":"' + ASPIRIN + '"}]}')
    question = f"Which of this library is nearest {CAFFEINE}? Use {library}"
    assert plan_of("chem_similarity", question, structure=CAFFEINE, library=library).unsourced == ()

    smuggled = library.replace(ASPIRIN, "c1ccccc1O")   # a member nobody asked about
    assert plan_of("chem_similarity", question, structure=CAFFEINE,
                   library=smuggled).unsourced == ("library",)


def test_a_question_with_no_structure_in_it_sources_none():
    assert plan_of("chem_logp", "What is the logP of caffeine?",
                   structure=CAFFEINE).unsourced == ("structure",)


def test_reading_structures_out_of_a_question_is_quiet_and_exact():
    """The candidates are runs of SMILES-legal characters, so prose and a JSON library are both
    split correctly -- and an English word is never handed to the toolkit, which would print a
    parse error to the user's console for every word of every question."""
    from sciai.domains.chem import standardize

    assert standardize.keys_in_text(CAFFEINE_QUESTION) == \
        frozenset({standardize.inchikey_of(CAFFEINE)})
    # written in prose, in brackets, with the sentence's question mark against it
    assert standardize.in_text(ASPIRIN, f"How many atoms has aspirin ({ASPIRIN})?")
    assert standardize.inchikey_of("molecular") is None
    assert standardize.inchikey_of("not a structure at all") is None


@pytest.mark.parametrize("written", [
    f"{CAFFEINE}?",            # the question ends in a question mark, right against the SMILES
    f"{CAFFEINE}.",
    f'"{CAFFEINE}"',
    f"  {CAFFEINE}  ",
    f"({CAFFEINE})",           # a question writes a structure in brackets, in prose
    f"({CAFFEINE})?",
    f"{CAFFEINE_KEKULE}?",
])
def test_a_structure_copied_with_the_sentence_around_it_is_still_the_question(written):
    """A model copying a structure out of prose copies what is next to it. None of this
    punctuation can begin or end a SMILES, and the charset guard has to stay wide enough for
    InChI (which does use "?" and ","), so it is trimmed from the slot instead."""
    assert plan_of("chem_descriptors", CAFFEINE_QUESTION, structure=written).unsourced == ()


def test_an_inchi_keeps_its_own_punctuation():
    """An InChI layer may legitimately end in "?" for undefined stereo, so nothing is trimmed."""
    from sciai.controller.recipes import _trim_structure

    inchi = "InChI=1S/C10H14N2/c1-12-7-3-5-10(12)9-4-2-6-11-8-9/h2,4,6,8,10H,3,5,7H2,1H3/t10?"
    assert _trim_structure(inchi) == inchi
