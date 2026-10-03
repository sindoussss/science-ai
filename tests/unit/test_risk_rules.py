import pytest

from sciai.config import RiskConfig
from sciai.graph.engine import GraphEngine
from sciai.graph.model import Layer, Node, NodeType
from sciai.store.db import Database
from sciai.store.repository import Repository
from sciai.tools.registry import get
from sciai.verify.risk_rules import assess


@pytest.fixture
def engine():
    db = Database(":memory:")
    e = GraphEngine(Repository(db))
    e.new_session("s")
    yield e
    db.close()


def node(tool="sympy.diff", args=None, result=None, **kw):
    return Node(session_id="", layer=Layer.REASONING, type=NodeType.TOOL_RESULT, title="t", tool_name=tool,
                tool_inputs=args or {"expr": "x", "var": "x"},
                result=result or {"kind": "expr", "value": "1", "srepr": "Integer(1)"}, **kw)


def test_nothing_fires_for_a_quiet_step(engine):
    n = engine.add_node(node())
    assert not assess(engine, n, get("sympy.diff"), RiskConfig()).required


def test_stakes(engine):
    n = engine.add_node(node())
    engine.add_node(node(), [n.id])
    r = assess(engine, n, get("sympy.diff"), RiskConfig(stakes_threshold=2))
    assert not r.required
    engine.add_node(node(), [n.id])
    assert "stakes" in assess(engine, n, get("sympy.diff"), RiskConfig(stakes_threshold=2)).rules()
    assert "stakes" in assess(engine, engine.add_node(node()), None, RiskConfig(), user_facing=True).rules()


def test_surprise_sanity_and_contradiction(engine):
    n = engine.add_node(node(result={"kind": "number", "value": float("inf")}))
    assert "surprise" in assess(engine, n, None, RiskConfig()).rules()
    a = engine.add_node(node(fingerprint="fp", result={"kind": "expr", "value": "2", "srepr": "Integer(2)"}))
    b = engine.add_node(node(fingerprint="fp", result={"kind": "expr", "value": "3", "srepr": "Integer(3)"}))
    assert "surprise" in assess(engine, b, None, RiskConfig()).rules()
    assert a.id != b.id


def test_unsourced_numbers_flag(engine):
    n = engine.add_node(node(flags=["unsourced_numbers: 3.7"]))
    assert "surprise" in assess(engine, n, None, RiskConfig()).rules()


def test_confidence_and_step_type(engine):
    n = engine.add_node(node(tool="numeric.quad", confidence=0.5,
                             args={"expr": "x", "var": "x", "lower": "0", "upper": "1"}))
    rules = assess(engine, n, get("numeric.quad"), RiskConfig()).rules()
    assert "confidence" in rules and "step_type" in rules
    long_algebra = {"expr": "(" * 7 + "x" + ")" * 7}
    assert get("sympy.simplify").requires_check(long_algebra)
    assert not get("sympy.simplify").requires_check({"expr": "x+1"})
