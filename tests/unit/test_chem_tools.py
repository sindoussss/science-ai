"""The chemistry tools and their checkers.

Each solver is exercised end to end, then its result is tampered with to prove the checker
actually looks. A checker that passes a corrupted result is worse than no checker, so every
tool here has a tampering test rather than only a happy path.
"""
from __future__ import annotations

import pytest

from sciai.domains.chem import ranking, similarity, standardize
from sciai.graph.model import Domain, Status
from sciai.tools import chem_tools as T
from sciai.tools.registry import all_tools, get

pytest.importorskip("rdkit", reason="rdkit is a Phase 4 dependency")

ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"
LIBRARY = {"members": [
    {"name": "ibuprofen", "structure": "CC(C)Cc1ccc(cc1)C(C)C(=O)O"},
    {"name": "salicylic acid", "structure": "OC(=O)c1ccccc1O"},
    {"name": "caffeine", "structure": "Cn1cnc2c1c(=O)n(C)c(=O)n2C"},
]}


def outcome(payload: dict) -> str:
    return payload["result"]["outcome"]


# ----------------------------------------------------------------- registry shape

def test_every_chem_solver_has_a_checker() -> None:
    names = {t.name for t in all_tools() if t.name.startswith("chem.")}
    solvers = {n for n in names if not n.startswith("chem.check_")}
    for solver in solvers:
        assert f"chem.check_{solver.split('.', 1)[1]}" in names, solver


def test_every_chem_tool_is_in_the_chem_domain() -> None:
    for tool in all_tools():
        if tool.name.startswith("chem."):
            assert tool.domain is Domain.CHEM, tool.name


def test_chem_domain_forces_hypothesis_through_the_engine(tmp_path) -> None:
    """A chem tool's result cannot arrive as anything but a hypothesis."""
    from sciai.graph.engine import GraphEngine
    from sciai.graph.model import Layer, Node, NodeType
    from sciai.store.db import Database
    from sciai.store.repository import Repository

    db = Database(tmp_path / "t.db")
    try:
        engine = GraphEngine(Repository(db))
        engine.new_session("s")
        node = engine.add_node(
            Node(session_id="", layer=Layer.DOMAIN, type=NodeType.TOOL_RESULT, title="mol"),
            tool_domain=get("chem.parse").domain)
        assert node.status is Status.HYPOTHESIS
    finally:
        db.close()


# ----------------------------------------------------------------- chem.parse

def test_parse_and_its_check() -> None:
    result = T.parse_fn({"structure": ASPIRIN})["result"]
    assert result["kind"] == "molecule"
    assert result["inchikey"] == "BSYNRYMUTXBXSQ-UHFFFAOYSA-N"
    assert result["formula"] == "C9H8O4"
    assert outcome(T.check_parse_fn({"molecule": result})) == "pass"


@pytest.mark.parametrize("field,value", [("inchikey", "AAAAAAAAAAAAAA-AAAAAAAAAA-A"),
                                         ("formula", "C9H8O5"), ("atoms", 99), ("charge", 3)])
def test_check_parse_catches_a_tampered_identity(field: str, value: object) -> None:
    result = T.parse_fn({"structure": ASPIRIN})["result"]
    assert outcome(T.check_parse_fn({"molecule": {**result, field: value}})) == "fail"


@pytest.mark.parametrize("spelling", [
    "CC(=O)Oc1ccccc1C(=O)O",
    "O=C(O)c1ccccc1OC(C)=O",                      # written backwards
    "[H]OC(=O)c1ccccc1OC(C)=O",                   # explicit hydrogen
    "CC(=O)OC1=CC=CC=C1C(O)=O",                   # kekulized
    "CC(=O)Oc1ccccc1C(=O)[O-].[Na+]",             # a salt form
])
def test_one_substance_written_five_ways_gives_one_identity(spelling: str) -> None:
    assert T.parse_fn({"structure": spelling})["result"]["skeleton"] == "BSYNRYMUTXBXSQ"


def test_a_stereoisomer_keeps_its_own_identity_under_a_shared_skeleton() -> None:
    left = T.parse_fn({"structure": "C[C@H](N)C(=O)O"})["result"]
    right = T.parse_fn({"structure": "C[C@@H](N)C(=O)O"})["result"]
    assert left["skeleton"] == right["skeleton"]
    assert left["inchikey"] != right["inchikey"]


def test_an_unreadable_structure_is_refused() -> None:
    with pytest.raises(standardize.MoleculeError, match="not a structure"):
        T.parse_fn({"structure": "this is not a molecule"})


def test_the_two_readers_must_agree(monkeypatch) -> None:
    """Forced disagreement: the import fails and names both readings."""
    real = standardize.identity
    calls = {"n": 0}

    def flaky(mol, **kw):
        calls["n"] += 1
        out = real(mol, **kw)
        if calls["n"] == 2:      # the second reading comes back with a different formula
            return type(out)(**{**out.__dict__, "formula": "C1H1"})
        return out

    monkeypatch.setattr(standardize, "identity", flaky)
    with pytest.raises(standardize.MoleculeError, match="two readings of this structure disagree"):
        standardize.read_twice(ASPIRIN)


# ----------------------------------------------------------------- chem.descriptors

def test_descriptors_and_its_check() -> None:
    result = T.descriptors_fn({"structure": ASPIRIN})["result"]
    values = result["values"]
    assert round(values["mw"], 2) == 180.16
    assert round(values["tpsa"], 2) == 63.60
    assert values["rings"] == 1
    assert outcome(T.check_descriptors_fn({"descriptors": result})) == "pass"


@pytest.mark.parametrize("name,delta", [("exact_mass", 0.002), ("heavy_atoms", 1), ("hbd", 1),
                                        ("hba", 1), ("rings", 1), ("tpsa", 0.5),
                                        ("rotatable_bonds", 1), ("formal_charge", 1)])
def test_check_descriptors_catches_a_tampered_value(name: str, delta: float) -> None:
    result = T.descriptors_fn({"structure": ASPIRIN})["result"]
    tampered = {**result, "values": {**result["values"],
                                     name: result["values"][name] + delta}}
    assert outcome(T.check_descriptors_fn({"descriptors": tampered})) == "fail"


def test_check_descriptors_catches_a_swapped_molecule() -> None:
    """The classic plumbing failure: the right numbers against the wrong structure."""
    result = T.descriptors_fn({"structure": ASPIRIN})["result"]
    other = T.descriptors_fn({"structure": "Cn1cnc2c1c(=O)n(C)c(=O)n2C"})["result"]
    swapped = {**result, "canonical_smiles": other["canonical_smiles"]}
    assert outcome(T.check_descriptors_fn({"descriptors": swapped})) == "fail"


# ----------------------------------------------------------------- chem.logp

def test_logp_is_labelled_a_single_method_estimate() -> None:
    result = T.logp_fn({"structure": ASPIRIN})["result"]
    assert result["method_count"] == 1
    assert result["label"] == "single-method estimate"
    assert "no independent second opinion" in result["caveat"]
    assert outcome(T.check_logp_fn({"estimate": result})) == "pass"


def test_check_logp_catches_a_tampered_value() -> None:
    result = T.logp_fn({"structure": ASPIRIN})["result"]
    assert outcome(T.check_logp_fn({"estimate": {**result, "value": 9.9}})) == "fail"


# ----------------------------------------------------------------- chem.druglike

def test_druglike_and_its_check() -> None:
    desc = T.descriptors_fn({"structure": ASPIRIN})["result"]
    logp = T.logp_fn({"structure": ASPIRIN})["result"]["value"]
    result = T.druglike_fn({"descriptors": desc, "logp": logp})["result"]
    assert result["verdicts"]["lipinski"]["passed"]
    assert result["verdicts"]["veber"]["passed"]
    assert outcome(T.check_druglike_fn({"verdict": result})) == "pass"


def test_druglike_names_what_fails() -> None:
    from sciai.domains.chem import druglike

    verdicts = druglike.evaluate_all({"mw": 700.0, "logp": 7.0, "hbd": 9.0, "hba": 14.0,
                                      "rotatable_bonds": 15.0, "tpsa": 200.0})
    assert verdicts["lipinski"].failing == ("mw", "logp", "hbd", "hba")
    assert not verdicts["lipinski"].passed
    assert verdicts["veber"].failing == ("rotatable_bonds", "tpsa")
    assert "historically clustered" in druglike.report(verdicts)


def test_druglike_allows_one_lipinski_miss() -> None:
    from sciai.domains.chem import druglike

    verdicts = druglike.evaluate_all({"mw": 520.0, "logp": 3.0, "hbd": 2.0, "hba": 5.0,
                                      "rotatable_bonds": 5.0, "tpsa": 80.0})
    assert verdicts["lipinski"].passed and verdicts["lipinski"].failing == ("mw",)


def test_check_druglike_catches_a_flipped_verdict() -> None:
    desc = T.descriptors_fn({"structure": ASPIRIN})["result"]
    result = T.druglike_fn({"descriptors": desc, "logp": 1.0})["result"]
    flipped = {**result, "verdicts": {**result["verdicts"],
                                      "veber": {"passed": False, "failing": ["tpsa"],
                                                "detail": {}}}}
    assert outcome(T.check_druglike_fn({"verdict": flipped})) == "fail"


# ----------------------------------------------------------------- chem.similar

def test_similar_and_its_check() -> None:
    result = T.similar_fn({"structure": ASPIRIN, "library": LIBRARY, "top_k": 3})["result"]
    names = [n["name"] for n in result["neighbours"]]
    assert names[0] == "salicylic acid"
    assert result["refused_members"] == []
    assert outcome(T.check_similar_fn({"neighbours": result})) == "pass"


def test_check_similar_catches_a_reordered_ranking() -> None:
    result = T.similar_fn({"structure": ASPIRIN, "library": LIBRARY, "top_k": 3})["result"]
    reversed_ = {**result, "neighbours": list(reversed(result["neighbours"]))}
    assert outcome(T.check_similar_fn({"neighbours": reversed_})) == "fail"


def test_a_refused_library_member_is_reported_not_dropped_silently() -> None:
    """The screen runs on every member, and a refusal is visible in the result."""
    library = {"members": [*LIBRARY["members"],
                           {"name": "reagent", "structure": "O=P(F)(c1ccccc1)c1ccccc1"}]}
    result = T.similar_fn({"structure": ASPIRIN, "library": library})["result"]
    assert [r["name"] for r in result["refused_members"]] == ["reagent"]
    assert "reagent" not in [n["name"] for n in result["neighbours"]]


def test_a_library_of_only_refused_members_fails_the_tool() -> None:
    library = {"members": [{"name": "reagent", "structure": "O=P(F)(c1ccccc1)c1ccccc1"}]}
    with pytest.raises(similarity.LibraryError, match="passed the structure screen"):
        T.similar_fn({"structure": ASPIRIN, "library": library})


def test_the_query_structure_is_screened_too() -> None:
    with pytest.raises(standardize.MoleculeError):
        T.similar_fn({"structure": "O=P(F)(c1ccccc1)c1ccccc1", "library": LIBRARY})


# ----------------------------------------------------------------- chem.substructure

def test_substructure_and_its_check() -> None:
    result = T.substructure_fn({"structure": ASPIRIN, "smarts": "c1ccccc1"})["result"]
    assert result["count"] == 1
    assert outcome(T.check_substructure_fn({"matches": result})) == "pass"


def test_check_substructure_catches_an_invented_match() -> None:
    result = T.substructure_fn({"structure": ASPIRIN, "smarts": "c1ccccc1"})["result"]
    invented = {**result, "count": 2, "atoms": [*result["atoms"], [0, 1, 2, 3, 4, 5]]}
    assert outcome(T.check_substructure_fn({"matches": invented})) == "fail"


def test_check_substructure_catches_a_wrong_count() -> None:
    result = T.substructure_fn({"structure": ASPIRIN, "smarts": "c1ccccc1"})["result"]
    assert outcome(T.check_substructure_fn({"matches": {**result, "count": 7}})) == "fail"


def test_an_unreadable_pattern_is_refused() -> None:
    with pytest.raises(similarity.LibraryError, match="not a pattern"):
        T.substructure_fn({"structure": ASPIRIN, "smarts": "[C"})


# ----------------------------------------------------------------- chem.cluster

def test_cluster_and_its_check() -> None:
    result = T.cluster_fn({"library": LIBRARY, "cutoff": 0.6})["result"]
    assert result["count"] == len(result["sizes"])
    assert sum(result["sizes"]) == len(LIBRARY["members"])
    assert outcome(T.check_cluster_fn({"clusters": result})) == "pass"


def test_check_cluster_catches_a_member_moved_into_the_wrong_cluster() -> None:
    result = T.cluster_fn({"library": LIBRARY, "cutoff": 0.2})["result"]
    merged = {**result, "assignment": [0] * len(result["assignment"]),
              "sizes": [len(result["assignment"])], "count": 1}
    assert outcome(T.check_cluster_fn({"clusters": merged})) == "fail"


def test_check_cluster_catches_a_short_assignment() -> None:
    result = T.cluster_fn({"library": LIBRARY})["result"]
    short = {**result, "assignment": result["assignment"][:-1]}
    assert outcome(T.check_cluster_fn({"clusters": short})) == "fail"


# ----------------------------------------------------------------- chem.rank

CANDIDATES = [{"name": "a", "values": {"mw": 300.0, "tpsa": 60.0}},
              {"name": "b", "values": {"mw": 500.0, "tpsa": 40.0}},
              {"name": "c", "values": {"mw": 400.0, "tpsa": 50.0}}]
CRITERIA = [{"descriptor": "mw", "direction": "lower_is_better"},
            {"descriptor": "tpsa", "direction": "higher_is_better"}]


def test_rank_and_its_check() -> None:
    result = T.rank_fn({"candidates": CANDIDATES, "criteria": CRITERIA})["result"]
    assert [r["name"] for r in result["ranking"]] == ["a", "c", "b"]
    assert outcome(T.check_rank_fn({"ranking": result, "candidates": CANDIDATES})) == "pass"


def test_rank_is_reproducible() -> None:
    first = T.rank_fn({"candidates": CANDIDATES, "criteria": CRITERIA})["result"]
    shuffled = [CANDIDATES[2], CANDIDATES[0], CANDIDATES[1]]
    second = T.rank_fn({"candidates": shuffled, "criteria": CRITERIA})["result"]
    assert first["ranking"] == second["ranking"]


def test_a_tie_stays_a_tie_and_is_broken_by_name() -> None:
    tied = [{"name": "zed", "values": {"mw": 300.0}}, {"name": "abe", "values": {"mw": 300.0}}]
    result = T.rank_fn({"candidates": tied,
                        "criteria": [{"descriptor": "mw", "direction": "lower_is_better"}]})["result"]
    scores = [r["score"] for r in result["ranking"]]
    assert scores[0] == scores[1]
    assert [r["name"] for r in result["ranking"]] == ["abe", "zed"]


def test_check_rank_catches_a_reordered_ranking() -> None:
    result = T.rank_fn({"candidates": CANDIDATES, "criteria": CRITERIA})["result"]
    reordered = {**result, "ranking": list(reversed(result["ranking"]))}
    assert outcome(T.check_rank_fn({"ranking": reordered, "candidates": CANDIDATES})) == "fail"


def test_check_rank_catches_a_tampered_score() -> None:
    result = T.rank_fn({"candidates": CANDIDATES, "criteria": CRITERIA})["result"]
    bumped = {**result, "ranking": [{**result["ranking"][0], "score": 9.0},
                                    *result["ranking"][1:]]}
    assert outcome(T.check_rank_fn({"ranking": bumped, "candidates": CANDIDATES})) == "fail"


@pytest.mark.parametrize("note", ["maximize toxicity", "pick the most lethal one",
                                  "we want to weaponize this series"])
def test_rank_refuses_criteria_that_aim_at_harm(note: str) -> None:
    criteria = [{"descriptor": "mw", "direction": "higher_is_better", "note": note}]
    with pytest.raises(ranking.CriteriaRefused):
        T.rank_fn({"candidates": CANDIDATES, "criteria": criteria})


@pytest.mark.parametrize("note", ["minimize toxicity", "lowest hepatotoxicity",
                                  "best safety margin", "maximize potency at the target"])
def test_rank_allows_ordinary_admet_criteria(note: str) -> None:
    criteria = [{"descriptor": "mw", "direction": "lower_is_better", "note": note}]
    result = T.rank_fn({"candidates": CANDIDATES, "criteria": criteria})["result"]
    assert len(result["ranking"]) == len(CANDIDATES)


def test_rank_needs_criteria_and_candidates() -> None:
    with pytest.raises(ranking.RankingError, match="no criteria"):
        T.rank_fn({"candidates": CANDIDATES, "criteria": []})
    with pytest.raises(ranking.RankingError, match="no candidates"):
        T.rank_fn({"candidates": [], "criteria": CRITERIA})


def test_rank_refuses_duplicate_names() -> None:
    dupes = [{"name": "a", "values": {"mw": 1.0}}, {"name": "a", "values": {"mw": 2.0}}]
    with pytest.raises(ranking.RankingError, match="share a name"):
        T.rank_fn({"candidates": dupes,
                   "criteria": [{"descriptor": "mw", "direction": "lower_is_better"}]})


def test_rank_says_a_ranking_is_a_hypothesis() -> None:
    result = T.rank_fn({"candidates": CANDIDATES, "criteria": CRITERIA})
    assert "not a prediction that any of these works" in result["meta"]["summary"]


# ----------------------------------------------------------------- schemas

def test_every_chem_tool_validates_its_arguments() -> None:
    import jsonschema

    with pytest.raises(jsonschema.ValidationError):
        get("chem.parse").validate({"structure": "CCO", "unexpected": 1})
    with pytest.raises(jsonschema.ValidationError):
        get("chem.rank").validate({"candidates": CANDIDATES,
                                   "criteria": [{"descriptor": "mw", "direction": "sideways"}]})
    get("chem.parse").validate({"structure": "CCO", "format": "smiles"})
