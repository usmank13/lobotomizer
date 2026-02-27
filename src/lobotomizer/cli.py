"""CLI entry point — placeholder."""
from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(prog="lobotomize", description="Composable model compression for PyTorch")
    parser.add_argument("--version", action="version", version="lobotomizer 0.1.0")
    _args = parser.parse_args()
    parser.print_help()


if __name__ == "__main__":
    main()
