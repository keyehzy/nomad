import json

from nomad import *
from nomad.io import dumps_json


def test_json_export_serializes_summed_indices_by_display_name():
    p, q = indices("p q")
    h = tensor("h", [p, q], hermitian=True)
    expr = sum_(p, q, h[p, q] * adag(p) * a(q))
    payload = json.loads(dumps_json(expr))
    (term,) = payload["terms"]
    # Bound dummies serialize by their canonical display name, consistently
    # across the summed list, the tensor ports, and the operator modes.
    assert term["summed"] == ["_0", "_1"]
    assert term["tensors"] == [{"symbol": "h", "ports": ["_0", "_1"]}]
    assert term["ops"] == [
        {"kind": "create", "mode": "_0"},
        {"kind": "destroy", "mode": "_1"},
    ]


def test_latex_export_for_one_body_sum():
    p, q = indices("p q")
    h = tensor("h", [p, q], hermitian=True)
    expr = sum_(p, q, h[p, q] * adag(p) * a(q))
    assert latex(expr) == r"\sum_{_0,_1} h_{_0,_1} a^\dagger_{_0} a_{_1}"


def test_openfermion_source_export():
    src = openfermion(2 * adag(0) * a(1))
    assert "from openfermion import FermionOperator" in src
    assert "FermionOperator('0^ 1', 2)" in src


def test_json_export_includes_typed_summed_domains():
    occ = domain("occ", size=2)
    i = index("i", occ)
    payload = json.loads(dumps_json(sum_(i, adag(i))))
    (term,) = payload["terms"]
    assert term["summed"] == ["_0"]
    assert term["summed_domains"] == {"_0": {"name": "occ", "size": 2}}
