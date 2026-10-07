"""Knowledge-store lookups. Pure CPU; runs before any model call."""
from __future__ import annotations

from typing import Any

from sciai.graph.fingerprint import question_fingerprint, tool_fingerprint
from sciai.graph.model import Domain, Node, Status
from sciai.store.repository import Repository
from sciai.tools.sandbox import Runner


def verified_answer_for_question(repo: Repository, question: str) -> Node | None:
    finals = [n for n in repo.verified_finals_for_problem(question_fingerprint(question))
              if not repo.unsourced_root(n.id)]
    return finals[0] if finals else None


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
