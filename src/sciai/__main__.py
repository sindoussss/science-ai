"""Entry point: python -m sciai [--model TAG] [--think | --no-think]"""
from __future__ import annotations

import argparse
import os
import sys


def parse_args(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    """Our options, plus whatever is left for Qt (e.g. -platform)."""
    p = argparse.ArgumentParser(prog="sciai", description="Science AI: local research workspace")
    p.add_argument("--model", metavar="TAG", help="Ollama model tag to use (overrides [model] name in the config)")
    p.add_argument("--think", action=argparse.BooleanOptionalAction, default=None,
                   help="turn the model's reasoning mode on or off (overrides [model] think)")
    return p.parse_known_args(argv)


def apply_overrides(cfg, args: argparse.Namespace) -> None:  # noqa: ANN001 - sciai.config.Config
    if args.model:
        cfg.model.name = args.model
    if args.think is not None:
        cfg.model.think = args.think


def main() -> int:
    args, qt_args = parse_args(sys.argv[1:])

    from PyQt6.QtWidgets import QApplication

    from sciai.config import load
    from sciai.runtime import build_runtime
    from sciai.ui.controller_thread import RootConfirmer
    from sciai.ui.main_window import MainWindow
    from sciai.ui.theme.theme import Theme, load_bundled_fonts

    app = QApplication([sys.argv[0], *qt_args])
    app.setApplicationName("Science AI")
    app.setStyle("Fusion")  # no native bevels; everything else comes from the QSS
    load_bundled_fonts()
    theme = Theme.load(os.environ.get("SCIAI_THEME", "light"))  # light is the default
    app.setStyleSheet(theme.qss())
    cfg = load()
    apply_overrides(cfg, args)
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
