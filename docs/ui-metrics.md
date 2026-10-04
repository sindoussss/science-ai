# UI layout metrics

Measured by `tests/ui/test_layout_metrics.py` (offscreen, injected-fault scenario, failed derivative selected).
The test reads real geometry from the laid-out widgets; nothing is estimated.

| window | scale | node title px | canvas % | chat % | sidebar | inspector | nodes fully in view | clipped text |
|---|---|---|---|---|---|---|---|---|
| 1200x720 | 1.000 | 13.00 | 71.1 | 27.9 | 220 | 340 | 2 | 0 |
| 1440x900 | 1.000 | 13.00 | 71.2 | 28.0 | 220 | 340 | 5 | 0 |

- **scale**: the canvas view transform, which must be at least 0.92. This graph does not fit at the floor at either size, so it shows at 1.0 centered on the focus node; Fit (Ctrl+0) shows everything.
- **node title px**: 13 × scale, which must be at least 12.
- **canvas % / chat %**: the graph's and the chat's share of the center column's height. The canvas must be at least 70% and the chat at most 30% (28% by default).
- **clipped text**: visible labels, buttons, inputs and tab bars whose text doesn't fit, or that a parent cuts off. Intentional elision is excluded, and so is content scrolled out of a scroll area. `test_clipping_detector_catches_real_clipping` proves the check flags a squeezed button.

Screenshots (`widget.grab()`): [1200x720](screenshots/injected-fault-1200x720.png), [1440x900](screenshots/injected-fault-1440x900.png).
