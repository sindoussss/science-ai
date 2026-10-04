"""Display helpers: Unicode math text and the canvas layout grid (no Qt needed)."""
from __future__ import annotations

from sciai.graph.model import EdgeKind, Layer, Node, NodeType
from sciai.ui.canvas.layout import COL_GAP, NODE_H, NODE_W, PILL_H, ROW_GAP, layered_positions
from sciai.ui.mathtext import answer_text, prose_text, value_text


def _n(title: str, typ: NodeType = NodeType.TOOL_RESULT, **kw) -> Node:
    return Node(session_id="s", layer=Layer.REASONING, type=typ, title=title, **kw)


def test_value_text():
    assert value_text("-pi**2") == "−π²"
    assert value_text("x**2*cos(x) + 2*x*sin(x)") == "x²·cos(x) + 2x·sin(x)"
    assert value_text("2*pi") == "2π"
    assert value_text("x**(1/3)") == "x^(1/3)"
    assert value_text("sqrt(2)") == "√2"
    assert value_text("not an expr ((") == "not an expr (("


def test_answer_text_renders_template_from_node_results():
    n6 = _n("f'(pi)", result={"kind": "expr", "value": "-pi**2"})
    final = _n("Answer", NodeType.FINAL, content="f'(pi) = -pi**2.",
               tool_inputs={"template": "f'(pi) = {{n6}}.", "answer_nodes": [n6.id]})
    assert answer_text(final, {"n6": n6}.get) == "f'(π) = −π²."


def test_answer_text_falls_back_when_a_value_is_missing():
    final = _n("Answer", NodeType.FINAL, content="f'(pi) = -pi**2.",
               tool_inputs={"template": "f'(pi) = {{n6}}.", "answer_nodes": []})
    assert answer_text(final, lambda h: None) == prose_text("f'(pi) = -pi**2.") == "f'(π) = −π²."


def test_layout_grid_and_check_pills_do_not_overlap():
    a, b, c = _n("a"), _n("b"), _n("c")
    checks = [_n(f"check: m{i}", NodeType.CHECK) for i in range(3)]
    nodes = {n.id: n for n in (a, b, c, *checks)}
    edges = [(b.id, a.id, EdgeKind.DEPENDS_ON), (c.id, a.id, EdgeKind.DEPENDS_ON)]
    edges += [(k.id, b.id, EdgeKind.CHECKS) for k in checks]
    pos = layered_positions(nodes, edges)
    assert pos[b.id][0] - pos[a.id][0] == NODE_W + COL_GAP
    # b carries three pills, so the next row starts below them plus the row gap
    last_pill_bottom = max(pos[k.id][1] for k in checks) + PILL_H
    assert pos[c.id][1] == last_pill_bottom + ROW_GAP
    tops = sorted(pos[k.id][1] for k in checks)
    assert tops[0] >= pos[b.id][1] + NODE_H
    assert all(t2 - t1 >= PILL_H for t1, t2 in zip(tops, tops[1:]))
