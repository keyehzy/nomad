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
