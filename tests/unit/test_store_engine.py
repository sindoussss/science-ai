import sqlite3

import pytest

from sciai.graph.engine import GraphEngine, GraphRuleError
from sciai.graph.model import Domain, EdgeKind, Evidence, Layer, Node, NodeType, Status
from sciai.store.db import Database
from sciai.store.repository import Repository


@pytest.fixture
def engine(tmp_path):
    db = Database(tmp_path / "t.db")
    e = GraphEngine(Repository(db))
    e.new_session("s")
    yield e
    db.close()


def tnode(title="n", **kw):
    return Node(session_id="", layer=Layer.REASONING, type=kw.pop("type", NodeType.TOOL_RESULT), title=title, **kw)


def verify(engine, node):
    engine.repo.add_evidence(Evidence(node_id=node.id, method="t", tool_name="t", inputs={}, outcome="pass", detail={}))
    engine.set_status(node.id, Status.VERIFIED, "test")


def test_handles_and_versions(engine):
    a = engine.add_node(tnode("a"))
    b = engine.add_node(tnode("b"), [a.id])
    b2 = engine.add_node(tnode("b2"), [a.id], replaces=b.id)
    assert engine.handle(a.id) == "n1" and engine.handle(b2.id) == "n3"
    assert b2.lineage_id == b.lineage_id and b2.version == 2
    assert engine.resolve(b.id).superseded_by == b2.id


def test_cascade_invalidates_transitive_dependents(engine):
    a = engine.add_node(tnode("a"))
    b = engine.add_node(tnode("b"), [a.id])
    c = engine.add_node(tnode("c"), [b.id])
    report = engine.set_status(a.id, Status.FAILED, "bad")
    assert set(report.invalidated) == {b.id, c.id}
    assert engine.resolve(c.id).status == Status.INVALIDATED
    with pytest.raises(GraphRuleError):
        engine.add_node(tnode("d"), [b.id])


def test_cross_session_cascade_leaves_notice(engine):
    a = engine.add_node(tnode("a"))
    verify(engine, a)
    first = engine.session_id
    engine.new_session("other")
    engine.link_existing(a.id)
    final = engine.add_node(tnode("answer", type=NodeType.FINAL), [a.id])
    other = engine.session_id
    engine.open_session(first)
    report = engine.set_status(a.id, Status.FAILED, "recheck failed")
    assert other in report.affected_sessions
    assert report.affected_sessions[other] == [(final.id, "answer")]
    notices = engine.repo.notices(other)
    assert len(notices) == 1 and "answer" in notices[0]["text"]


def test_locked_semantics(engine):
    a = engine.add_node(tnode("a"))
    b = engine.add_node(tnode("b"), [a.id])
    verify(engine, b)
    engine.lock(b.id)
    engine.demote(b.id, automatic=True)
    assert engine.resolve(b.id).status == Status.VERIFIED and engine.resolve(b.id).needs_recheck
    with pytest.raises(GraphRuleError):
        engine.delete(b.id)
    report = engine.set_status(a.id, Status.FAILED, "bad")
    assert b.id in report.locked_warnings
    node = engine.resolve(b.id)
    assert node.status == Status.INVALIDATED and node.warning
    assert engine.prune_invalidated() == 0  # locked node is never auto-deleted
    engine.demote(b.id, automatic=False)  # user demotion of an invalidated node is a no-op


def test_verified_requires_passing_evidence(engine):
    a = engine.add_node(tnode("a"))
    with pytest.raises(GraphRuleError):
        engine.set_status(a.id, Status.VERIFIED, "no evidence")


def test_domain_comes_from_tool_registry_only(engine):
    a = engine.add_node(tnode("a", domain=Domain.CHEM))  # caller-supplied domain is ignored
    assert a.domain == Domain.GENERAL
    m = engine.add_node(tnode("m"), tool_domain=Domain.MATH)
    child = engine.add_node(tnode("c"), [m.id])
    assert child.domain == Domain.MATH


def test_chem_never_verified_engine_and_database(engine):
    c = engine.add_node(tnode("mol"), tool_domain=Domain.CHEM)
    assert c.status == Status.HYPOTHESIS
    engine.repo.add_evidence(Evidence(node_id=c.id, method="t", tool_name="t", inputs={}, outcome="pass", detail={}))
    with pytest.raises(GraphRuleError):
        engine.set_status(c.id, Status.VERIFIED, "x")
    db = engine.repo.db
    with pytest.raises(sqlite3.IntegrityError):
        with db.tx() as conn:
            conn.execute("UPDATE nodes SET status='verified' WHERE id=?", (c.id,))
    m = engine.add_node(tnode("m"), tool_domain=Domain.MATH)
    verify(engine, m)
    with pytest.raises(sqlite3.IntegrityError):
        with db.tx() as conn:
            conn.execute("UPDATE nodes SET domain='chem' WHERE id=?", (m.id,))
    with pytest.raises(sqlite3.IntegrityError):
        with db.tx() as conn:
            conn.execute(
                "INSERT INTO nodes (id, lineage_id, version, session_id, layer, type, domain, title, status, "
                "created_at, updated_at) VALUES ('x','x',1,?,'reasoning','claim','chem','t','verified',0,0)",
                (engine.session_id,))


def test_delete_invalidates_dependents(engine):
    a = engine.add_node(tnode("a"))
    b = engine.add_node(tnode("b"), [a.id])
    engine.delete(a.id)
    assert engine.resolve(b.id).status == Status.INVALIDATED
    assert engine.repo.get_node(a.id) is None


def test_reload_session_rebuilds_graph(engine):
    a = engine.add_node(tnode("a"))
    b = engine.add_node(tnode("b"), [a.id])
    sid = engine.session_id
    engine.new_session("x")
    engine.open_session(sid)
    assert engine.g.has_edge(b.id, a.id)
    assert engine.g.edges[b.id, a.id]["kind"] == EdgeKind.DEPENDS_ON
