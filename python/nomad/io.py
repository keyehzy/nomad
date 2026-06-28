"""Small serialization/export helpers for NOMAD."""

from __future__ import annotations

import json
from typing import Any

from .core import (
    Domain,
    Expr,
    _is_default_spin_orbital_domain,
    _tensor_port_text,
    as_expr,
    latex,
    openfermion,
    text,
)


def dumps_text(expr: Any) -> str:
    return text(expr)


def dumps_latex(expr: Any) -> str:
    return latex(expr)


def dumps_openfermion(expr: Any) -> str:
    return openfermion(expr)


def _domain_payload(value: Domain) -> dict[str, Any]:
    payload: dict[str, Any] = {"name": value.name}
    if value.values is not None:
        payload["values"] = list(value.values)
    elif value.size is not None:
        payload["size"] = value.size
        if value.start:
            payload["start"] = value.start
    return payload


def dumps_json(expr: Any) -> str:
    e: Expr = as_expr(expr).simplify()
    terms = []
    for t in e.terms:
        item: dict[str, Any] = {
            "coeff": [t.coeff.numerator, t.coeff.denominator],
            "summed": [str(i) for i in t.summed],
            "deltas": [[str(d.left), str(d.right)] for d in t.deltas],
            "tensors": [
                {"symbol": tf.symbol.name, "ports": [_tensor_port_text(p) for p in tf.ports]}
                for tf in t.tensors
            ],
            "ops": [{"kind": op.kind, "mode": str(op.mode)} for op in t.ops],
        }
        summed_domains: dict[str, Any] = {}
        for i in t.summed:
            domain_value = i.domain
            if isinstance(domain_value, Domain) and not _is_default_spin_orbital_domain(
                domain_value
            ):
                summed_domains[str(i)] = _domain_payload(domain_value)
        if summed_domains:
            item["summed_domains"] = summed_domains
        terms.append(item)
    return json.dumps({"terms": terms}, indent=2, sort_keys=True)
