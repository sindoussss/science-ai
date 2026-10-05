"""Computational chemistry: structure identity, descriptors, screening and ranked hypotheses.

Scope is computational only. Nothing here plans a reaction, writes a procedure or gives a dose:
no such tool is registered, so the controller has nothing to call (see ``decline``), and every
structure is screened before it enters the graph (see ``restricted``).

Every chem node stays a hypothesis. The store, the graph engine and the verifier each enforce
that independently; this package never tries to set another status.
"""
from __future__ import annotations

__all__: list[str] = []
