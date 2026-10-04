"""Shared UI fixtures: a window that has run the injected-fault task."""
from __future__ import annotations

import pytest

pytest.importorskip("PyQt6")

from tests.ui.metrics import fault_runner, fault_script, run_fault_window  # noqa: E402


@pytest.fixture
def fault_window(make_rt, runner):
    wins = []

    def open_(size):
        rt = make_rt(fault_script(), run=fault_runner(runner))
        win = run_fault_window(rt, size)
        wins.append(win)
        return win

    yield open_
    for w in wins:
        w.close()
