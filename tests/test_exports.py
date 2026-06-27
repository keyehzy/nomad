from nomad import *


def test_latex_export_for_one_body_sum():
    p, q = indices("p q")
    h = tensor("h", [p, q], hermitian=True)
    expr = sum_(p, q, h[p, q] * adag(p) * a(q))
    assert latex(expr) == r"\sum_{_0,_1} h_{_0,_1} a^\dagger_{_0} a_{_1}"


def test_openfermion_source_export():
    src = openfermion(2 * adag(0) * a(1))
    assert "from openfermion import FermionOperator" in src
    assert "FermionOperator('0^ 1', 2)" in src
