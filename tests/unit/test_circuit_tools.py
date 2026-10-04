import pytest

from sciai.tools.registry import get
from sciai.tools.sandbox import InlineRunner
from sciai.verify.checks import plans_for

R = InlineRunner()


def el(name, kind, n1, n2, value, unit=None):
    return {"name": name, "type": kind, "n1": n1, "n2": n2,
            "value": {"value": value, "unit": unit} if unit else value}


DIVIDER = {"elements": [el("V1", "V", "1", "0", 10, "V"), el("R1", "R", "1", "2", 100, "ohm"),
                        el("R2", "R", "2", "0", 150, "ohm")], "ground": "0"}
LADDER = {"elements": [el("V1", "V", "1", "0", 12), el("R1", "R", "1", "2", 1000), el("R2", "R", "2", "0", 2000),
                       el("R3", "R", "2", "3", 500), el("R4", "R", "3", "0", 1500), el("R5", "R", "3", "4", 300)]}
BRIDGE = {"elements": [el("V1", "V", "1", "0", 10), el("R1", "R", "1", "2", 100), el("R2", "R", "1", "3", 200),
                       el("R3", "R", "2", "3", 300), el("R4", "R", "2", "0", 400), el("R5", "R", "3", "0", 500)]}
CURRENT = {"elements": [el("I1", "I", "0", "1", 0.01, "A"), el("R1", "R", "1", "0", 1, "kohm"),
                        el("R2", "R", "1", "0", 1, "kohm")]}


def run(tool, **args):
    out = R.run(tool, args)
    assert out.ok, out.error
    return out.value["result"]


def checks(netlist, result):
    return {p.method: run(p.checker, **a)["outcome"] for p, a in plans_for("circuit.dc", {"netlist": netlist}, result)}


def test_divider():
    r = run("circuit.dc", netlist=DIVIDER, outputs=["V(2)", "I(R1)", "P(V1)", "V(1,2)"])
    vals = [v["value"] for v in r["value"]]
    assert vals == pytest.approx([6.0, 0.04, -0.4, 4.0])
    assert [v["unit"] for v in r["value"]] == ["V", "A", "W", "V"]
    assert checks(DIVIDER, r) == {"kirchhoff": "pass", "power_balance": "pass", "series_parallel": "pass"}


def test_ladder_with_dangling_resistor():
    r = run("circuit.dc", netlist=LADDER)
    assert checks(LADDER, r) == {"kirchhoff": "pass", "power_balance": "pass", "series_parallel": "pass"}
    assert r["solution"]["currents"]["R5"] == pytest.approx(0.0, abs=1e-15)


def test_bridge_reduction_not_applicable():
    r = run("circuit.dc", netlist=BRIDGE)
    assert checks(BRIDGE, r) == {"kirchhoff": "pass", "power_balance": "pass", "series_parallel": "inconclusive"}


def test_current_source():
    r = run("circuit.dc", netlist=CURRENT, outputs=["V(1)"])
    assert r["value"][0]["value"] == pytest.approx(5.0)
    assert set(checks(CURRENT, r).values()) == {"pass"}


@pytest.mark.parametrize("tamper", ["voltage", "resistor_current", "source_current"])
def test_checks_catch_tampered_solution(tamper):
    r = run("circuit.dc", netlist=DIVIDER)
    sol = r["solution"]
    if tamper == "voltage":
        sol["node_voltages"]["2"] *= 1.05
    elif tamper == "resistor_current":
        sol["currents"]["R2"] *= 1.05
    else:
        sol["currents"]["V1"] *= 1.05
    assert "fail" in checks(DIVIDER, r).values()


def test_netlist_errors():
    def bad(netlist):
        return not R.run("circuit.dc", {"netlist": netlist}).ok

    assert bad({"elements": [el("R1", "R", "1", "0", 100), el("R1", "R", "1", "0", 100)]})  # duplicate name
    assert bad({"elements": [el("V1", "V", "1", "0", 5), el("R1", "R", "1", "1", 100)]})  # self loop
    assert bad({"elements": [el("V1", "V", "1", "0", 5), el("R1", "R", "1", "0", 0)]})  # zero resistance
    assert bad({"elements": [el("V1", "V", "1", "0", 5), el("R1", "R", "2", "3", 100)]})  # floating node
    assert bad({"elements": [el("V1", "V", "1", "0", 5), el("V2", "V", "1", "0", 6)]})  # source loop
    assert bad({"elements": [el("V1", "V", "1", "0", 5, "A"), el("R1", "R", "1", "0", 100)]})  # wrong unit
    assert bad({"elements": [el("V1", "V", "1", "0", 5), el("R1", "R", "1", "0", 100)], "ground": "9"})
    assert not R.run("circuit.dc", {"netlist": DIVIDER, "outputs": ["I(R9)"]}).ok


def test_canonical_netlist_ignores_order_and_units():
    canon = get("circuit.dc").canonical
    flipped = {"elements": [el("R2", "R", "2", "0", 0.15, "kohm"), el("R1", "R", "1", "2", 100),
                            el("V1", "V", "1", "0", 10000, "mV")], "ground": "0"}
    assert canon({"netlist": DIVIDER}) == canon({"netlist": flipped})
