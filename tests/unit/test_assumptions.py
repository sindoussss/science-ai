from sciai.domains.physics.assumptions import DEFAULTS, PROBLEM_TYPES, checklist, normalize_type


def test_every_problem_type_has_a_checklist():
    assert set(PROBLEM_TYPES) == {"projectile", "kinematics", "dynamics", "energy", "thermo", "dc_circuit",
                                  "rc_rl_transient", "ode", "general"}
    for t in PROBLEM_TYPES:
        items = checklist(t)
        assert items and all(i["source"] == "default" for i in items)
        assert [i["text"] for i in items] == list(DEFAULTS[t])


def test_unknown_type_gets_general_and_math_gets_none():
    assert normalize_type("Fluid-Dynamics") == "general"
    assert normalize_type("DC circuit") == "dc_circuit"
    assert normalize_type("math") is None and normalize_type(None) is None and normalize_type(" ") is None
    assert checklist("math") == []
    assert [i["text"] for i in checklist("nonsense")] == list(DEFAULTS["general"])


def test_model_assumptions_follow_defaults_without_duplicates():
    items = checklist("projectile", ["No air resistance.", "flat ground", "  flat   ground ", "", "x" * 300])
    texts = [i["text"] for i in items]
    assert texts[:3] == list(DEFAULTS["projectile"])
    assert texts[3] == "flat ground" and items[3]["source"] == "model"
    assert len(texts) == 5 and len(texts[4]) == 120
    assert checklist(None, ["ideal wires"]) == [{"text": "ideal wires", "source": "model"}]
