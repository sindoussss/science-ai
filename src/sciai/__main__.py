"""Entry point: python -m sciai"""
from __future__ import annotations

import os
import sys


def main() -> int:
    from PyQt6.QtWidgets import QApplication

    from sciai.config import load
    from sciai.runtime import build_runtime
    from sciai.ui.controller_thread import RootConfirmer
    from sciai.ui.main_window import MainWindow
    from sciai.ui.theme.theme import Theme

    app = QApplication(sys.argv)
    app.setApplicationName("Science AI")
    theme = Theme.load(os.environ.get("SCIAI_THEME", "light"))  # light is the default
    app.setStyleSheet(theme.qss())
    cfg = load()
    confirmer = RootConfirmer()
    rt = build_runtime(cfg, confirm_root=confirmer)
    win = MainWindow(rt, theme, confirmer)
    win.show()
    try:
        return app.exec()
    finally:
        rt.close()


if __name__ == "__main__":
    raise SystemExit(main())
