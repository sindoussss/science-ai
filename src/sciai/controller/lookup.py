"""Knowledge-store lookups. Pure CPU; runs before any model call."""
from __future__ import annotations

from typing import Any

from sciai.graph.fingerprint import question_fingerprint, tool_fingerprint
from sciai.graph.model import Domain, Node, NodeType, Status
from sciai.store.repository import Repository
from sciai.tools.sandbox import Runner


def verified_answer_for_question(repo: Repository, question: str) -> Node | None:
    finals = [n for n in repo.verified_finals_for_problem(question_fingerprint(question))
              if not repo.unsourced_root(n.id)]
    return finals[0] if finals else None


def answer_for_question(repo: Repository, question: str) -> Node | None:
    """A stored answer to this same question, or None. No model call, no tool run.

    Normally that means a verified final. A chemistry answer is never verified -- that is the
    rule of its phase -- so asking the same molecule question twice used to cost a full routing
    call and a full recomputation. The test for those is chemistry's own, the one its tool-level
    reuse already uses: every result the answer rests on passed its checks and none failed. The
    answer comes back exactly as it was stored, which is to say still a hypothesis.
    """
    hit = verified_answer_for_question(repo, question)
    if hit is not None:
        return hit
    for final in repo.finals_for_problem(question_fingerprint(question), [Status.HYPOTHESIS]):
        if final.domain != Domain.CHEM or repo.unsourced_root(final.id):
            continue
        rests_on = _results_behind(repo, final.id)
        if rests_on and all(_checked(repo, n) for n in rests_on):
            return final
    return None


def _results_behind(repo: Repository, node_id: str) -> list[Node]:
    """Every tool result the answer rests on, however deep.

    The answer also depends on the problem node and on its assumptions, which are not results
    and have nothing to check; walking to the results is what asks the right question.
    """
    seen: set[str] = set()
    found: list[Node] = []
    queue = list(repo.dependencies(node_id))
    while queue:
        nid = queue.pop()
        if nid in seen:
            continue
        seen.add(nid)
        node = repo.get_node(nid)
        if node is None:
            continue
        if node.type is NodeType.TOOL_RESULT:
            found.append(node)
        if node.type is not NodeType.PROBLEM:
            queue += repo.dependencies(nid)
    return found


def _checked(repo: Repository, node: Node) -> bool:
    """Verified, or a chemistry hypothesis whose checks all passed."""
    if node.status == Status.VERIFIED:
        return True
    if node.status != Status.HYPOTHESIS or node.domain != Domain.CHEM:
        return False
    outcomes = {ev.outcome for ev in repo.evidence_for(node.id)}
    return "pass" in outcomes and "fail" not in outcomes


def canonical_fingerprint(runner: Runner, tool: str, args: dict[str, Any]) -> tuple[str | None, str | None]:
    """(fingerprint, error). Canonicalization parses model text, so it runs in the sandbox."""
    out = runner.canonicalize(tool, args)
    if not out.ok:
        return None, out.error
    return tool_fingerprint(tool, out.value["canonical"]), None


def reusable_by_fingerprint(repo: Repository, fp: str) -> Node | None:
    """An earlier result this call may reuse.

    Nothing descended from a problem whose slots were not in the question is ever reused: the
    chem run of 2026-10-07 stored such answers and the repeat question then came back with one.

    Normally that means a verified node. A chemistry node is never verified, so for those the
    test is the one its phase was given instead: every check that ran passed, and none failed.
    What that buys is reuse, not belief, and the node comes back still a hypothesis.
    """
    hits = [n for n in repo.find_by_fingerprint(fp, [Status.VERIFIED])
            if not repo.unsourced_root(n.id)]
    if hits:
        return hits[0]
    for node in repo.find_by_fingerprint(fp, [Status.HYPOTHESIS]):
        if node.domain != Domain.CHEM or repo.unsourced_root(node.id):
            continue
        outcomes = {ev.outcome for ev in repo.evidence_for(node.id)}
        if "pass" in outcomes and "fail" not in outcomes:
            return node
    return None


def hint_by_fingerprint(repo: Repository, fp: str) -> Node | None:
    """An unverified earlier result: surfaced as a hint, never reused silently."""
    hits = [n for n in repo.find_by_fingerprint(fp, [Status.PROPOSED, Status.HYPOTHESIS])
            if not repo.unsourced_root(n.id)]
    return hits[0] if hits else None
