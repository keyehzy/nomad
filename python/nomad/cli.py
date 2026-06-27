"""NOMAD command-line entry point.

The standalone qoal parser is deliberately not included.  The CLI offers a
sanity-check demo and file-based normalization for Python snippets that assign
an expression to a variable named ``expr``.
"""

from __future__ import annotations

import argparse
import runpy
from pathlib import Path
from typing import Any

from . import a, adag, indices, latex, normal_order, sum_, tensor, text


def _demo_expr() -> Any:
    p, q = indices("p q")
    h = tensor("h", [p, q], hermitian=True)
    return sum_(p, q, h[p, q] * adag(p) * a(q))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="nomad", description="NOMAD algebra compiler frontend")
    sub = parser.add_subparsers(dest="cmd")

    demo = sub.add_parser("demo", help="print a normalized one-body Hamiltonian demo")
    demo.add_argument("--emit", choices=["text", "latex"], default="text")

    norm = sub.add_parser("normalize", help="normalize a Python DSL file defining variable 'expr'")
    norm.add_argument("file", type=Path)
    norm.add_argument("--emit", choices=["text", "latex"], default="text")

    args = parser.parse_args(argv)
    if args.cmd == "demo":
        expr = normal_order(_demo_expr())
    elif args.cmd == "normalize":
        ns = runpy.run_path(str(args.file))
        if "expr" not in ns:
            raise SystemExit("Input file must define variable 'expr'")
        expr = normal_order(ns["expr"])
    else:
        parser.print_help()
        return 1

    print(latex(expr) if args.emit == "latex" else text(expr))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
