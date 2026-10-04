# UI layout metrics (chat-centered layout, polish pass)

Measured by `tests/ui/test_layout_metrics.py` and `tests/ui/test_polish.py` (offscreen, injected-fault scenario,
failed derivative selected). The tests read real geometry from the laid-out widgets and colors from grabbed pixels;
nothing is estimated. Clipping is checked in five states: Graph tab, node Code tab, node Review tab, the tools
popover and the pin popover.

| window | regions sidebar / chat / workspace | chat pane | workspace card | tab gaps (min) | graph scale | node title px | nodes fully in view | composer parts | clipped text |
|---|---|---|---|---|---|---|---|---|---|
| 1200x720 | 204 / 516 / 480 (17 / 43 / 40%) | 490 | 455 | 18 | 1.020 | 13.26 | 6 | 6/6 | 0 |
| 1440x900 | 240 / 624 / 576 (17 / 43 / 40%) | 598 | 551 | 18 | 1.238 | 16.09 | 6 | 6/6 | 0 |

- **regions**: they sum to the window width. Sidebar = its card, 17% clamped to 200–240. Workspace = from the divider
  line to the right edge, 40% clamped to 440–640. Chat = everything between (its gutters included); the chat pane
  itself never goes below 440. Proportions hold at every width (tested at 1200, 1280, 1440, 1680, 1920). A dragged
  divider keeps its share of the window on resize; Ctrl+G collapses the workspace.
- **tab gaps**: px between adjacent node sub-tab labels (Code, Execution Log, Messages, Environment, Review), 13.5px
  labels, 16px side padding. 18px at every default width; at the narrowest drag they go to 12 and the side padding
  to 12, and no label elides.
- **graph scale**: never below 0.92 and never above 1.3. **node title px** = 13 × scale.
- **composer parts**: status strip, "+", tools, mic, send, and the "Ask anything" input, all visible.
- **clipped text**: visible text that doesn't fit or that a parent cuts off, plus table columns narrower than they
  may go. Designed elision (long expressions, elided labels) is excluded, and so is content scrolled out of view.

## Polish checks (tests/ui/test_polish.py)

| # | item | what is asserted |
|---|---|---|
| 1 | session title | 15px Medium, one line; a 120-character title ends in "…" and fits; the thread viewport and its first message start ≥ 16px below the title |
| 2 | scrollbars | every scroll area (not the canvas) has native bars off and overlay bars: 8px wide, hidden at rest, shown on hover and while scrolling (then hidden after 0.9s), no arrow buttons, handle pixel #D8D6CF |
| 3 | proportions | the region rules above at 5 widths; all five tab labels in full with 18px gaps; ≥ 12px at the narrowest drag |
| 4 | one column | table, answer card, plot, paragraphs and composer share the left edge (pad = round(4.5% × chat width)); cards and composer have equal widths; still true at the narrowest chat |
| 5 | chips | the five status colors exactly as specified, pale (HSV saturation < 0.15), radius 6, 12px Medium, 2px/8px padding, dashed border only for invalidated; painted fill checked by pixel; "Stakes" uses the proposed colors |
| 6 | composer | radius 20, border #E4E2DB, shadow 0 2px 8px α 13 (0.05), strip #F4F3EF and card fill by pixel, strip text 12.5px wide, 20px icons in 32px hit areas, 36px terracotta send, 16px padding to glyphs, input and send |
| 7 | rhythm | 24px between the summary paragraph, table card and answer card; table header 36, rows 40, radius 12; warm-white row fill by pixel |
| 8 | colors | panel and window not #FFFFFF; no neutral token is pure gray or cooler than warm (whites only as text on accent buttons) |

## Screenshots

`widget.grab()`, rendered by `python -m tests.ui.screenshots docs/screenshots <reference.png>`:

| state | 1200x720 | 1440x900 |
|---|---|---|
| (a) chat with table and answer cards | [a](screenshots/a-chat-1200x720.png) | [a](screenshots/a-chat-1440x900.png) |
| (b) run in progress, tools popover open | [b](screenshots/b-running-tools-1200x720.png) | [b](screenshots/b-running-tools-1440x900.png) |
| (c) Graph tab with a pin popover | [c](screenshots/c-graph-pin-1200x720.png) | [c](screenshots/c-graph-pin-1440x900.png) |
| (d) node Code tab | [d](screenshots/d-node-code-1200x720.png) | [d](screenshots/d-node-code-1440x900.png) |
| (e) node Review tab | [e](screenshots/e-node-review-1200x720.png) | [e](screenshots/e-node-review-1440x900.png) |

Comparisons with the reference:
- [side-by-side-reference-vs-chat.png](screenshots/side-by-side-reference-vs-chat.png): the reference's app window
  (backdrop cropped) scaled to 720 high, next to (a) at 1200x720.
- [left](screenshots/logical-1440-left-reference-vs-ours.png) and
  [right](screenshots/logical-1440-right-reference-vs-ours.png) halves at logical scale: the reference is a ~1.28x
  capture of a ~1440px window, so it is resized to 1440 wide and compared with our 1440x900 grab.

## Remaining differences from the reference

- **Type scale.** The reference's logical text is ~15–17px (wordmark ~28); ours follows the specified sizes
  (UI 14, chat 15, wordmark 24), so our text is smaller at the same window size.
- **Fractional sizes.** This font engine rasterizes 12.5 and 13.5px text at 13 and 14px. Widths are kept exact with a
  matching horizontal stretch, so glyphs are about half a pixel taller than specified.
- **Table rows.** 40px rows and a 36px header as specified; the reference's are ~30px.
- **Composer width.** Equal edges with the text column, as specified; the reference's composer is wider than its text.
- **Workspace share.** 40% as specified; the reference's is ~42%, so its cards are ~25px wider at 1440.
- **Download script** stays the one filled blue button (an earlier spec); the reference has no blue button.
- **Content.** The reference shows a notebook and a scatter plot; ours shows the reasoning graph, the legend and the
  knowledge base, which the reference doesn't have.
