"""Render the review screenshots with widget.grab() (real widgets, offscreen).

    QT_QPA_PLATFORM=offscreen python -m tests.ui.screenshots docs/screenshots [reference.png]

For each size: (a) chat with the table and answer cards, (b) a run in progress with the
tools popover open, (c) Graph tab with a pin popover, (d) node Code tab, (e) node Review tab.
With a reference image, also writes a side-by-side of the reference and (a) at 1200x720.
"""
from __future__ import annotations

import os
import sys
import tempfile
import threading
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

SIZES = [(1200, 720), (1440, 900)]
# The app window inside the reference screenshot (it sits on a decorative backdrop), measured on the
# 2000px-wide preview of the 2048px original: (left, top, right, bottom).
REF_WINDOW_2000 = (82, 82, 1918, 1146)


def app_window_box(ref) -> tuple[int, int, int, int]:  # noqa: ANN001 - a PIL image
    k = ref.width / 2000
    return tuple(round(v * k) for v in REF_WINDOW_2000)  # type: ignore[return-value]
PAUSE_AT_CALL = 5  # tool call to hold the run at for (b)


def main() -> None:
    from sciai.config import Config
    from sciai.graph.model import NodeType
    from sciai.runtime import build_runtime
    from sciai.tools.sandbox import SandboxRunner
    from sciai.ui.controller_thread import RootConfirmer
    from sciai.ui.main_window import MainWindow
    from sciai.ui.theme.theme import Theme
    from tests.acceptance.test_phase1 import QUESTION
    from tests.ui.metrics import (
        TABLE_HEAD,
        failed_diff,
        fault_runner,
        fault_script,
        measure,
        open_pin_editor,
        pump,
        setup_app,
        table_row,
    )

    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    reference = Path(sys.argv[2]) if len(sys.argv) > 2 else None
    app = setup_app()
    cfg = Config()
    cfg.controller.auto_confirm_root = True
    sandbox = SandboxRunner(timeout=30)
    rows = []
    for w, h in SIZES:
        tag = f"{w}x{h}"
        runner = fault_runner(sandbox)
        gate, paused = threading.Event(), threading.Event()
        calls = {"n": 0}

        def hold(_tool, _args, gate=gate, paused=paused, calls=calls) -> None:  # noqa: ANN001
            calls["n"] += 1
            if calls["n"] == PAUSE_AT_CALL:
                paused.set()
                gate.wait(30)

        runner.on_call = hold
        rt = build_runtime(cfg, db_path=Path(tempfile.mkdtemp()) / "kb.db", llm=fault_script(), runner=runner)
        win = MainWindow(rt, Theme.load("light"), RootConfirmer())
        win.resize(w, h)
        win.show()
        pump(app, lambda win=win: win.session_id is not None)
        win.chat.input.setPlainText(QUESTION)
        win.chat._submit()

        # (b) running: Live, the strip ticking, the tools popover open
        pump(app, lambda paused=paused: paused.is_set())
        pump(app, lambda: True)
        win.chat.toggle_tools()
        pump(app, lambda: True)
        win.grab().save(str(out / f"b-running-tools-{tag}.png"))
        win.chat.toggle_tools()
        gate.set()
        pump(app, lambda win=win: not win.executor.is_busy
             and any(n.type == NodeType.FINAL for n in win.graph.nodes.values()))
        pump(app, lambda: True)

        # (a) the chat with the table card and the final answer card
        bar = win.chat.scroll.verticalScrollBar()
        bar.setValue(bar.maximum())
        pump(app, lambda: True)
        win.grab().save(str(out / f"a-chat-{tag}.png"))

        # metrics on the same window (failed derivative selected, as in the test)
        bad = failed_diff(win)
        win._select_and_show(bad.id)
        pump(app, lambda: True)
        m = measure(win)
        rows.append(table_row(m))
        if m.clipped:
            print("clipped:", m.clipped)

        # (c) Graph tab with a pin popover open
        win.workspace.show_graph()
        open_pin_editor(win)
        win.graph.pin_editor.text.setPlainText("Recheck this derivative")
        pump(app, lambda: True)
        win.grab().save(str(out / f"c-graph-pin-{tag}.png"))
        win.graph.pin_editor.hide()

        # (d) node Code tab, (e) node Review tab
        win.workspace.show_node_tab()
        win.node_panel.tabs.setCurrentIndex(0)
        pump(app, lambda: True)
        win.grab().save(str(out / f"d-node-code-{tag}.png"))
        win.node_panel.tabs.setCurrentIndex(4)
        pump(app, lambda: True)
        win.grab().save(str(out / f"e-node-review-{tag}.png"))
        win.close()
        rt.close()
    sandbox.close()
    print(TABLE_HEAD + "\n" + "\n".join(rows))

    if reference is not None:
        from PIL import Image

        ours = Image.open(out / "a-chat-1200x720.png").convert("RGB")
        ref = Image.open(reference).convert("RGB")
        ref = ref.crop(app_window_box(ref))  # the app window only, without the backdrop around it
        ref = ref.resize((round(ref.width * ours.height / ref.height), ours.height), Image.LANCZOS)
        gap = 24
        side = Image.new("RGB", (ref.width + gap + ours.width, ours.height), (255, 255, 255))
        side.paste(ref, (0, 0))
        side.paste(ours, (ref.width + gap, 0))
        side.save(out / "side-by-side-reference-vs-chat.png")


if __name__ == "__main__":
    main()
