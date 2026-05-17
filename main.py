# Copyright (C) 2026 b1nku
# SPDX-License-Identifier: GPL-3.0-or-later

import argparse
from ui.app import RouterApp


def main() -> None:
    parser = argparse.ArgumentParser(description="Prompt router TUI")
    parser.add_argument(
        "--mode",
        choices=["a", "b", "c"],
        default="b",
        metavar="MODE",
        help="Display mode: a = no energy info, b = energy with colour (default), c = energy without colour",
    )
    args = parser.parse_args()
    RouterApp(display_mode=args.mode).run()


if __name__ == "__main__":
    main()
