"""Canonical hashes used to find reusable results in the knowledge store."""
from __future__ import annotations

import hashlib
import json
import re
from typing import Any


def fingerprint(kind: str, payload: Any) -> str:
    blob = json.dumps({"kind": kind, "payload": payload}, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


_WS = re.compile(r"\s+")


def normalize_question(text: str) -> str:
    """Cheap, deterministic normalization for exact-match reuse of a question."""
    t = _WS.sub(" ", text.strip().lower())
    t = t.replace("**", "^")
    return t.rstrip(" .?!")


def question_fingerprint(text: str) -> str:
    return fingerprint("question", normalize_question(text))


def tool_fingerprint(tool_name: str, canonical_args: dict[str, Any]) -> str:
    return fingerprint("tool", {"tool": tool_name, "args": canonical_args})
