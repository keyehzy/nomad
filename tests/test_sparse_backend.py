import numpy as np

from nomad import *


def test_apply_ops_to_det_sign():
    ops = [Op("create", Orbital(2)), Op("destroy", Orbital(0))]
    assert apply_ops_to_det(0b0011, ops) == (0b0110, -1)


def test_number_operator_sparse_dense_matrix():
    H = 2 * adag(0) * a(0) + 3 * adag(1) * a(1)
    op = compile(H, target="sparse", n_orbitals=2, sector={"N": 1})
    assert op.basis == (0b01, 0b10)
    assert op.to_dense() == [[2.0, 0.0], [0.0, 3.0]]
    assert op.matvec([5.0, 7.0]) == [10.0, 21.0]


def test_hopping_sparse_matrix():
    H = adag(1) * a(0) + adag(0) * a(1)
    op = compile(H, target="sparse", n_orbitals=2, sector={"N": 1})
    assert op.to_dense() == [[0.0, 1.0], [1.0, 0.0]]


def test_tensor_sum_expansion_matches_dense_one_body_matrix():
    p, q = indices("p q")
    h = tensor("h", [p, q])
    H = sum_(p, q, h[p, q] * adag(p) * a(q))
    values = np.array([[1.0, 2.0], [3.0, 4.0]])
    op = compile(H, target="sparse", n_orbitals=2, sector={"N": 1}, tensor_values={"h": values})
    assert op.to_dense() == [[1.0, 2.0], [3.0, 4.0]]


def test_matrix_free_matvec_agrees_with_materialized_dense_matrix():
    H = 0.5 * adag(0) * a(0) + adag(1) * a(0) + adag(0) * a(1) + 1.5 * adag(1) * a(1)
    op = compile(H, target="sparse", n_orbitals=2, sector={"N": 1})
    x = np.array([0.25, -0.75])
    dense = np.array(op.to_dense())
    assert np.allclose(op.matvec(x), dense @ x)


def test_typed_domain_expansion_uses_domain_specific_ranges():
    occ = domain("occ", size=2)
    virt = domain("virt", size=2, start=2)
    i = index("i", occ)
    av = index("a", virt)

    expanded = expand_sums(sum_(i, av, adag(av) * a(i)), n_orbitals=4)
    pairs = sorted((term.ops[0].mode.value, term.ops[1].mode.value) for term in expanded.terms)
    assert pairs == [(2, 0), (2, 1), (3, 0), (3, 1)]


def test_typed_domain_tensor_values_use_domain_local_axes():
    occ = domain("occ", size=2)
    virt = domain("virt", size=2, start=2)
    i = index("i", occ)
    av = index("a", virt)
    t = tensor("t", [av, i])

    expanded = expand_sums(
        sum_(av, i, t[av, i] * adag(av) * a(i)),
        n_orbitals=4,
        tensor_values={"t": [[10, 11], [20, 21]]},
    )
    coeffs = {
        (term.ops[0].mode.value, term.ops[1].mode.value): int(term.coeff)
        for term in expanded.terms
    }
    assert coeffs == {(2, 0): 10, (2, 1): 11, (3, 0): 20, (3, 1): 21}


def test_string_domain_can_be_made_finite_with_size_override():
    i, j = indices("i j", domain="occ")
    expanded = expand_sums(sum_(i, j, adag(i) * a(j)), domain_sizes={"occ": 2})

    pairs = sorted((term.ops[0].mode.value, term.ops[1].mode.value) for term in expanded.terms)
    assert pairs == [(0, 0), (0, 1), (1, 0), (1, 1)]
