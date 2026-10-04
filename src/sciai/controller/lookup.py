"""Knowledge-store lookups. Pure CPU; runs before any model call."""
from __future__ import annotations

from typing import Any

from sciai.graph.fingerprint import question_fingerprint, tool_fingerprint
from sciai.graph.model import Node, Status
from sciai.store.repository import Repository
from sciai.tools.sandbox import Runner


def verified_answer_for_question(repo: Repository, question: str) -> Node | None:
    finals = repo.verified_finals_for_problem(question_fingerprint(question))
    return finals[0] if finals else None


def canonical_fingerprint(runner: Runner, tool: str, args: dict[str, Any]) -> tuple[str | None, str | None]:
    """(fingerprint, error). Canonicalization parses model text, so it runs in the sandbox."""
    out = runner.canonicalize(tool, args)
    if not out.ok:
        return None, out.error
    return tool_fingerprint(tool, out.value["canonical"]), None


def verified_by_fingerprint(repo: Repository, fp: str) -> Node | None:
    hits = repo.find_by_fingerprint(fp, [Status.VERIFIED])
    return hits[0] if hits else None


def hint_by_fingerprint(repo: Repository, fp: str) -> Node | None:
    """An unverified earlier result: surfaced as a hint, never reused silently."""
    hits = repo.find_by_fingerprint(fp, [Status.PROPOSED, Status.HYPOTHESIS])
    return hits[0] if hits else None
