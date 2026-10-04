from __future__ import annotations

import os
import re

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from sciai.config import Config  # noqa: E402
from sciai.runtime import build_runtime  # noqa: E402
from sciai.tools.sandbox import SandboxRunner  # noqa: E402


class SharedRunner:
    """Session-wide sandbox that tests cannot close."""

    def __init__(self, inner: SandboxRunner) -> None:
        self.inner = inner
        self.calls: list[str] = []

    def run(self, tool, args, timeout=None):
        self.calls.append(tool)
        return self.inner.run(tool, args, timeout)

    def canonicalize(self, tool, args):
        return self.inner.canonicalize(tool, args)

    def close(self):
        pass


@pytest.fixture(scope="session")
def sandbox():
    s = SandboxRunner(timeout=30)
    yield s
    s.close()


@pytest.fixture
def runner(sandbox):
    return SharedRunner(sandbox)


@pytest.fixture
def cfg():
    c = Config()
    c.controller.auto_confirm_root = True
    return c


@pytest.fixture
def make_rt(cfg, tmp_path, runner):
    made = []

    def make(llm, db_path=None, run=None):
        rt = build_runtime(cfg, db_path=db_path or tmp_path / "kb.db", llm=llm, runner=run or runner)
        made.append(rt)
        return rt

    yield make
    for rt in made:
        rt.close()


def result_of(prompt: str, handle: str) -> str:
    for line in prompt.splitlines():
        if line.startswith(handle + " "):
            return line.split(" = ", 1)[1].split(" <- ")[0].strip()
    raise AssertionError(f"{handle} not in prompt:\n{prompt}")


def numbers(text: str) -> set[str]:
    return set(re.findall(r"\d+(?:\.\d+)?", text))
