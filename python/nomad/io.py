"""Small serialization/export helpers for NOMAD."""

from __future__ import annotations

import json
from typing import Any

from .core import Expr, as_expr, latex, openfermion, text


def dumps_text(expr: Any) -> str:
    return text(expr)


def dumps_latex(expr: Any) -> str:
    return latex(expr)


def dumps_openfermion(expr: Any) -> str:
    return openfermion(expr)


def dumps_json(expr: Any) -> str:
    e: Expr = as_expr(expr).simplify()
    terms = []
    for t in e.terms:
        terms.append(
            {
                "coeff": [t.coeff.numerator, t.coeff.denominator],
                "summed": list(t.summed),
                "deltas": [[str(d.left), str(d.right)] for d in t.deltas],
                "tensors": [
                    {"symbol": tf.symbol.name, "ports": [str(p) for p in tf.ports]}
                    for tf in t.tensors
                ],
                "ops": [{"kind": op.kind, "mode": str(op.mode)} for op in t.ops],
            }
        )
    return json.dumps({"terms": terms}, indent=2, sort_keys=True)
