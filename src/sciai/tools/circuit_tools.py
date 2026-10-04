"""DC circuits: modified nodal analysis, with three independent checks.

Netlist: ``{"elements": [{"name": "R1", "type": "R", "n1": "1", "n2": "0", "value": {"value": 100,
"unit": "ohm"}}, ...], "ground": "0"}``. Types: R (resistor), V (voltage source, n1 is +),
I (current source pushing current from n1 through itself to n2). A plain number is in SI.

Sign convention everywhere: ``I(X)`` is the current through X from n1 to n2, and
``P(X) = (V(n1) - V(n2)) * I(X)`` is the power X absorbs (negative when it delivers).

Checks:
- kirchhoff: KCL at every node, plus every element's own law (V = IR, source values),
  rebuilt from the netlist rather than from the solver's matrix. Current and voltage can
  satisfy KCL and KVL with a wrong resistor current, so the element laws are part of it.
- power_balance: power the sources deliver equals sum(V^2 / R) over the resistors. The
  resistor side uses the element law, so it is not just KCL and KVL restated.
- series_parallel: for a single-source network that reduces by series and parallel
  steps, the source current and every resistor current from the reduction. Anything
  else (a bridge, several sources) records "not applicable" rather than a pass.
"""
from __future__ import annotations

import re
from typing import Any

import numpy as np

from sciai.domains.physics import quantities as Q
from sciai.graph.model import Domain
from sciai.tools.phys_tools import VALUE
from sciai.tools.registry import ToolSpec, register, schema

NODE = {"type": "string", "pattern": r"^[A-Za-z0-9_]{1,16}$"}
NAME = {"type": "string", "pattern": r"^[A-Za-z][A-Za-z0-9_]{0,15}$"}
ELEMENT = {"type": "object", "properties": {"name": NAME, "type": {"enum": ["R", "V", "I"]}, "n1": NODE,
                                            "n2": NODE, "value": VALUE},
           "required": ["name", "type", "n1", "n2", "value"], "additionalProperties": False}
NETLIST = {"type": "object", "properties": {"elements": {"type": "array", "items": ELEMENT, "minItems": 2,
                                                         "maxItems": 60},
                                            "ground": NODE},
           "required": ["elements"], "additionalProperties": False}
OUTPUT_RE = re.compile(r"^(V)\(([A-Za-z0-9_]{1,16})(?:,\s*([A-Za-z0-9_]{1,16}))?\)$|^([IP])\(([A-Za-z][A-Za-z0-9_]{0,15})\)$")
OUTPUTS = {"type": "array", "items": {"type": "string", "maxLength": 40}, "maxItems": 12}
KIND_OF = {"R": "resistance", "V": "voltage", "I": "current"}
SI_UNIT = {"R": "ohm", "V": "V", "I": "A"}
REL_TOL = 1e-9


def _check(outcome: str, **detail: Any) -> dict[str, Any]:
    return {"result": {"kind": "check", "outcome": outcome, **detail}, "confidence": None, "meta": {}}


def _value(el: dict[str, Any]) -> float:
    v = el["value"]
    kind = KIND_OF[el["type"]]
    if isinstance(v, dict):
        q, _, _ = Q.from_object({**v, "kind": v.get("kind", kind)})
        return float(Q.to_si(q).magnitude)
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise ValueError(f"value of {el['name']} must be a number (SI) or a quantity")
    return float(v)


def parse_netlist(netlist: dict[str, Any]) -> tuple[list[dict[str, Any]], str]:
    ground = netlist.get("ground", "0")
    elements, names = [], set()
    for el in netlist["elements"]:
        if el["name"] in names:
            raise ValueError(f"element name {el['name']} is used twice")
        names.add(el["name"])
        if el["n1"] == el["n2"]:
            raise ValueError(f"{el['name']} connects node {el['n1']} to itself")
        value = _value(el)
        if el["type"] == "R" and not value > 0:
            raise ValueError(f"{el['name']} must have a positive resistance (use a 0 V source for a wire)")
        elements.append({**el, "si": value})
    nodes = {e["n1"] for e in elements} | {e["n2"] for e in elements}
    if ground not in nodes:
        raise ValueError(f"ground node {ground!r} is not connected to anything")
    return elements, ground


def solve_mna(elements: list[dict[str, Any]], ground: str) -> dict[str, Any]:
    nodes = sorted(({e["n1"] for e in elements} | {e["n2"] for e in elements}) - {ground})
    idx = {n: i for i, n in enumerate(nodes)}
    vs = [e for e in elements if e["type"] == "V"]
    n, m = len(nodes), len(vs)
    A = np.zeros((n + m, n + m))
    z = np.zeros(n + m)

    def at(node: str) -> int | None:
        return idx.get(node)

    for e in elements:
        a, b = at(e["n1"]), at(e["n2"])
        if e["type"] == "R":
            g = 1.0 / e["si"]
            for i, j, s in ((a, a, g), (b, b, g), (a, b, -g), (b, a, -g)):
                if i is not None and j is not None:
                    A[i, j] += s
        elif e["type"] == "I":
            if a is not None:
                z[a] -= e["si"]
            if b is not None:
                z[b] += e["si"]
    for k, e in enumerate(vs):
        a, b = at(e["n1"]), at(e["n2"])
        if a is not None:
            A[a, n + k] += 1
            A[n + k, a] += 1
        if b is not None:
            A[b, n + k] -= 1
            A[n + k, b] -= 1
        z[n + k] = e["si"]
    cond = np.linalg.cond(A) if A.size else 1.0
    if not np.isfinite(cond) or cond > 1e14:
        raise ValueError("the circuit has no unique solution (a floating node or a loop of voltage sources)")
    x = np.linalg.solve(A, z)
    volts = {ground: 0.0, **{node: float(x[i]) for node, i in idx.items()}}
    currents = {}
    for e in elements:
        if e["type"] == "R":
            currents[e["name"]] = (volts[e["n1"]] - volts[e["n2"]]) / e["si"]
        elif e["type"] == "I":
            currents[e["name"]] = e["si"]
    for k, e in enumerate(vs):
        currents[e["name"]] = float(x[n + k])
    return {"node_voltages": volts, "currents": currents, "condition_number": float(cond)}


def _output(label: str, elements: list[dict[str, Any]], sol: dict[str, Any]) -> dict[str, Any]:
    m = OUTPUT_RE.match(label.replace(" ", ""))
    if m is None:
        raise ValueError(f"output {label!r} must look like V(2), V(2,3), I(R1) or P(R1)")
    volts, cur = sol["node_voltages"], sol["currents"]
    u = Q.ureg()
    if m.group(1):
        a, b = m.group(2), m.group(3)
        for node in (a, b):
            if node is not None and node not in volts:
                raise ValueError(f"unknown node {node!r}")
        v = volts[a] - (volts[b] if b is not None else 0.0)
        return Q.result(u.Quantity(v, "V"), "V", "voltage")
    kind, name = m.group(4), m.group(5)
    el = next((e for e in elements if e["name"] == name), None)
    if el is None:
        raise ValueError(f"unknown element {name!r}")
    if kind == "I":
        return Q.result(u.Quantity(cur[name], "A"), "A", "current")
    p = (volts[el["n1"]] - volts[el["n2"]]) * cur[name]
    return Q.result(u.Quantity(p, "W"), "W", "power")


def dc_fn(args: dict[str, Any]) -> dict[str, Any]:
    elements, ground = parse_netlist(args["netlist"])
    sol = solve_mna(elements, ground)
    labels = list(args.get("outputs") or [f"V({n})" for n in sorted(sol["node_voltages"]) if n != ground])
    items = [_output(lbl, elements, sol) for lbl in labels]
    confidence = 1.0 if sol["condition_number"] < 1e10 else 0.5
    return {"result": {"kind": "list", "value": items, "labels": labels, "solution": sol},
            "confidence": confidence, "meta": {"algorithm": "modified nodal analysis"}}


def canonical_netlist(netlist: dict[str, Any]) -> dict[str, Any]:
    elements = []
    for el in sorted(netlist["elements"], key=lambda e: e["name"]):
        v = el["value"]
        # A plain number is already SI, so 100 and {"value": 0.1, "unit": "kohm"} fingerprint alike.
        obj = v if isinstance(v, dict) else {"value": v, "unit": SI_UNIT[el["type"]]}
        value = Q.canonical_object({**obj, "kind": obj.get("kind", KIND_OF[el["type"]])})
        elements.append({"name": el["name"], "type": el["type"], "n1": el["n1"], "n2": el["n2"], "value": value})
    return {"elements": elements, "ground": netlist.get("ground", "0")}


def describe_netlist(netlist: dict[str, Any]) -> str:
    parts = []
    for el in netlist["elements"]:
        v = el["value"]
        shown = f"{v['value']:g} {v['unit']}" if isinstance(v, dict) else f"{v:g} {SI_UNIT[el['type']]}"
        parts.append(f"{el['name']} {el['n1']}-{el['n2']} {shown}")
    return "; ".join(parts)


register(ToolSpec(
    name="circuit.dc", domain=Domain.PHYSICS, kind="solver",
    description='DC circuit by nodal analysis: netlist={"elements": [{"name": "V1", "type": "V", "n1": "1", '
                '"n2": "0", "value": {"value": 10, "unit": "V"}}, {"name": "R1", "type": "R", "n1": "1", '
                '"n2": "2", "value": {"value": 100, "unit": "ohm"}}, ...], "ground": "0"}; '
                'outputs=["V(2)", "I(R1)", "P(R1)"]',
    schema=schema({"netlist": NETLIST, "outputs": OUTPUTS}, ["netlist"]),
    fn=dc_fn, always_check=True, entity_arg="netlist",
    canonical=lambda a: {"netlist": canonical_netlist(a["netlist"]), "outputs": list(a.get("outputs") or [])},
))


# ------------------------------------------------------------------ checkers
def _scale(values: list[float]) -> float:
    return max([abs(v) for v in values] + [1e-300])


def check_kirchhoff_fn(args: dict[str, Any]) -> dict[str, Any]:
    elements, ground = parse_netlist(args["netlist"])
    volts, cur = args["solution"]["node_voltages"], args["solution"]["currents"]
    problems = []
    if abs(volts.get(ground, 0.0)) > 0:
        problems.append("ground is not at 0 V")
    i_scale = _scale(list(cur.values()))
    v_scale = _scale(list(volts.values()))
    for node in sorted(set(volts) - {ground}):
        leaving = sum(cur[e["name"]] for e in elements if e["n1"] == node) - \
            sum(cur[e["name"]] for e in elements if e["n2"] == node)
        if abs(leaving) > REL_TOL * i_scale:
            problems.append(f"KCL at node {node}: net current {leaving:.3g} A")
    for e in elements:
        v = volts[e["n1"]] - volts[e["n2"]]
        if e["type"] == "R" and abs(v - cur[e["name"]] * e["si"]) > REL_TOL * v_scale:
            problems.append(f"{e['name']}: V = {v:.6g} V but I*R = {cur[e['name']] * e['si']:.6g} V")
        elif e["type"] == "V" and abs(v - e["si"]) > REL_TOL * v_scale:
            problems.append(f"{e['name']}: {v:.6g} V across a {e['si']:g} V source")
        elif e["type"] == "I" and abs(cur[e["name"]] - e["si"]) > REL_TOL * i_scale:
            problems.append(f"{e['name']}: {cur[e['name']]:.6g} A through a {e['si']:g} A source")
    return _check("fail" if problems else "pass", reason="; ".join(problems[:5]) or "KCL and element laws hold")


def check_power_fn(args: dict[str, Any]) -> dict[str, Any]:
    elements, _ = parse_netlist(args["netlist"])
    volts, cur = args["solution"]["node_voltages"], args["solution"]["currents"]
    delivered = sum(-(volts[e["n1"]] - volts[e["n2"]]) * cur[e["name"]] for e in elements if e["type"] != "R")
    dissipated = sum((volts[e["n1"]] - volts[e["n2"]]) ** 2 / e["si"] for e in elements if e["type"] == "R")
    ok = abs(delivered - dissipated) <= REL_TOL * _scale([delivered, dissipated])
    return _check("pass" if ok else "fail", sources_deliver_W=delivered, resistors_dissipate_W=dissipated)


def _reduce(elements: list[dict[str, Any]], a: str, b: str):
    """Series/parallel reduction of the resistors between source terminals a and b.
    Returns (tree, zero_current_resistors) or None when the network doesn't reduce."""
    edges = [{"u": e["n1"], "v": e["n2"], "R": e["si"], "tree": ("leaf", e["name"], e["si"])}
             for e in elements if e["type"] == "R"]
    zero: list[str] = []

    def leaves(tree) -> list[str]:
        return [tree[1]] if tree[0] == "leaf" else leaves(tree[1]) + leaves(tree[2])

    changed = True
    while changed:
        changed = False
        degree: dict[str, list[int]] = {}
        for i, e in enumerate(edges):
            degree.setdefault(e["u"], []).append(i)
            degree.setdefault(e["v"], []).append(i)
        for node, ids in degree.items():  # dangling resistors carry no current
            if node not in (a, b) and len(ids) == 1:
                zero += leaves(edges[ids[0]]["tree"])
                edges.pop(ids[0])
                changed = True
                break
        if changed:
            continue
        for i in range(len(edges)):  # parallel
            for j in range(i + 1, len(edges)):
                e, f = edges[i], edges[j]
                if {e["u"], e["v"]} == {f["u"], f["v"]}:
                    edges[i] = {"u": e["u"], "v": e["v"], "R": e["R"] * f["R"] / (e["R"] + f["R"]),
                                "tree": ("par", e["tree"], f["tree"], e["R"], f["R"])}
                    edges.pop(j)
                    changed = True
                    break
            if changed:
                break
        if changed:
            continue
        for node, ids in degree.items():  # series
            if node not in (a, b) and len(ids) == 2:
                e, f = edges[ids[0]], edges[ids[1]]
                ends = [x for x in (e["u"], e["v"], f["u"], f["v"]) if x != node]
                if len(ends) != 2:
                    continue
                if ends[0] == ends[1]:  # a closed loop with no source in it carries no current
                    zero += leaves(e["tree"]) + leaves(f["tree"])
                    for k in sorted(ids, reverse=True):
                        edges.pop(k)
                    changed = True
                    break
                new = {"u": ends[0], "v": ends[1], "R": e["R"] + f["R"],
                       "tree": ("ser", e["tree"], f["tree"], e["R"], f["R"])}
                for k in sorted(ids, reverse=True):
                    edges.pop(k)
                edges.append(new)
                changed = True
                break
    if len(edges) == 1 and {edges[0]["u"], edges[0]["v"]} == {a, b}:
        return edges[0], zero
    return None


def _push(tree, voltage: float, current: float, out: dict[str, float]) -> None:
    kind = tree[0]
    if kind == "leaf":
        out[tree[1]] = current
    elif kind == "ser":
        _push(tree[1], current * tree[3], current, out)
        _push(tree[2], current * tree[4], current, out)
    else:
        _push(tree[1], voltage, voltage / tree[3], out)
        _push(tree[2], voltage, voltage / tree[4], out)


def check_reduction_fn(args: dict[str, Any]) -> dict[str, Any]:
    elements, _ = parse_netlist(args["netlist"])
    volts, cur = args["solution"]["node_voltages"], args["solution"]["currents"]
    sources = [e for e in elements if e["type"] != "R"]
    if len(sources) != 1:
        return _check("inconclusive", reason=f"not applicable: {len(sources)} sources (needs exactly one)")
    src = sources[0]
    reduced = _reduce(elements, src["n1"], src["n2"])
    if reduced is None:
        return _check("inconclusive", reason="not applicable: the network does not reduce by series and "
                                             "parallel steps (e.g. a bridge)")
    edge, zero = reduced
    if src["type"] == "V":
        v_total, i_total = abs(src["si"]), abs(src["si"]) / edge["R"]
        claimed_source = abs(cur[src["name"]])
        expected_source = i_total
    else:
        i_total, v_total = abs(src["si"]), abs(src["si"]) * edge["R"]
        claimed_source = abs(volts[src["n1"]] - volts[src["n2"]])
        expected_source = v_total
    branch: dict[str, float] = {}
    _push(edge["tree"], v_total, i_total, branch)
    branch.update({name: 0.0 for name in zero})
    problems = []
    scale = _scale([expected_source, claimed_source])
    if abs(claimed_source - expected_source) > REL_TOL * scale:
        problems.append(f"source {src['name']}: reduction gives {expected_source:.6g}, solution {claimed_source:.6g}")
    i_scale = _scale(list(branch.values()) + [abs(c) for c in cur.values()])
    for name, i in sorted(branch.items()):
        if abs(abs(cur[name]) - i) > REL_TOL * i_scale:
            problems.append(f"{name}: reduction gives {i:.6g} A, solution {abs(cur[name]):.6g} A")
    return _check("fail" if problems else "pass", equivalent_resistance=edge["R"],
                  reason="; ".join(problems[:5]) or "reduction agrees")


_CHECK_SCHEMA = schema({"netlist": NETLIST, "solution": {"type": "object"}}, ["netlist", "solution"])
for _name, _fn, _desc in (("circuit.check_kirchhoff", check_kirchhoff_fn, "KCL and element laws from the netlist"),
                          ("circuit.check_power", check_power_fn, "source power against resistor V^2/R"),
                          ("circuit.check_reduction", check_reduction_fn, "series/parallel reduction")):
    register(ToolSpec(name=_name, domain=Domain.PHYSICS, kind="checker", description=_desc,
                      schema=_CHECK_SCHEMA, fn=_fn))
