# UI layout metrics (chat-centered layout)

Measured by `tests/ui/test_layout_metrics.py` (offscreen, injected-fault scenario, failed derivative selected).
The test reads real geometry from the laid-out widgets; nothing is estimated. Clipping is checked in five
states: Graph tab, node Code tab, node Review tab, the tools popover and the pin popover.

| window | chat | sidebar | workspace | chat widest | graph scale | node title px | nodes fully in view | composer parts | clipped text |
|---|---|---|---|---|---|---|---|---|---|
| 1200x720 | 505 | 220 | 420 | yes | 0.941 | 12.24 | 6 | 6/6 | 0 |
| 1440x900 | 745 | 220 | 420 | yes | 0.941 | 12.24 | 6 | 6/6 | 0 |

- **chat / sidebar / workspace**: widths in px. The chat must be at least 440 and the widest region; the sidebar is 220;
  the workspace defaults to 420 and can be dragged between 340 and 640 (Ctrl+G collapses it). Window resizes go to the chat.
- **graph scale**: never below 0.92 and never above 1.3. This graph fits the 420px workspace at 0.941, so all six
  step nodes are in view. In a narrower workspace it shows at 1.0 centered on the focus node; Fit (Ctrl+0) may go
  lower until the next layout.
- **node title px**: 13 × scale, which must be at least 12.
- **composer parts**: status strip, "+", tools, mic, send, and the "Ask anything" input, all visible.
- **clipped text**: visible labels, buttons, inputs and tab bars whose text doesn't fit or that a parent cuts off,
  plus table-card columns narrower than their cells. Intentional elision (the long-expression column, elided labels)
  is excluded, and so is content scrolled out of view.

The tests also assert the exact tab labels (`Graph`, `n2 · f'(x)`, and `Code`, `Execution Log`, `Messages`,
`Environment`, `Review`), that Ctrl+G collapses and restores the workspace, that dragging is clamped to 340–640, and
that "View graph" opens the Graph tab with the answer selected and unrelated nodes dimmed.

Screenshots (`widget.grab()`, rendered by `python -m tests.ui.screenshots docs/screenshots <reference.png>`):

| state | 1200x720 | 1440x900 |
|---|---|---|
| (a) chat with table and answer cards | [a](screenshots/a-chat-1200x720.png) | [a](screenshots/a-chat-1440x900.png) |
| (b) run in progress, tools popover open | [b](screenshots/b-running-tools-1200x720.png) | [b](screenshots/b-running-tools-1440x900.png) |
| (c) Graph tab with a pin popover | [c](screenshots/c-graph-pin-1200x720.png) | [c](screenshots/c-graph-pin-1440x900.png) |
| (d) node Code tab | [d](screenshots/d-node-code-1200x720.png) | [d](screenshots/d-node-code-1440x900.png) |
| (e) node Review tab | [e](screenshots/e-node-review-1200x720.png) | [e](screenshots/e-node-review-1440x900.png) |

Side by side with the reference: [side-by-side-reference-vs-chat.png](screenshots/side-by-side-reference-vs-chat.png).
