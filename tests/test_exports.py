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


def test_latex_export_annotates_typed_domain_sum():
    occ = domain("occ", size=2)
    i = index("i", occ)
    # A non-default domain annotates the summation symbol with ``\in <name>``,
    # mirroring the ``Σ_i∈occ`` form that text() renders.
    assert latex(sum_(i, adag(i) * a(i))) == r"\sum_{_0\in occ} a^\dagger_{_0} a_{_0}"


def test_json_export_summed_domains_carry_values_and_offset_forms():
    # The explicit values form and the offset size/start form each serialize
    # their distinguishing finite data, not just a bare size.
    vi = index("i", domain("occ", values=[0, 2]))
    payload = json.loads(dumps_json(sum_(vi, adag(vi))))
    assert payload["terms"][0]["summed_domains"] == {"_0": {"name": "occ", "values": [0, 2]}}

    si = index("a", domain("virt", size=2, start=2))
    payload = json.loads(dumps_json(sum_(si, adag(si))))
    assert payload["terms"][0]["summed_domains"] == {
        "_0": {"name": "virt", "size": 2, "start": 2}
    }
