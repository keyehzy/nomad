import numpy as np
import pytest

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
    expanded = expand_sums(sum_(i, j, adag(i) * a(j)), domains={"occ": 2})

    pairs = sorted((term.ops[0].mode.value, term.ops[1].mode.value) for term in expanded.terms)
    assert pairs == [(0, 0), (0, 1), (1, 0), (1, 1)]


def test_equivalent_domain_forms_compile_to_same_matrix():
    # size/start and an equivalent explicit values list are the same domain, so
    # they must compile to identical operators.
    i_size = index("i", domain("occ", size=2))
    i_vals = index("i", domain("occ", values=[0, 1]))
    m_size = np.array(compile(sum_(i_size, adag(i_size) * a(i_size)), n_orbitals=2).to_dense())
    m_vals = np.array(compile(sum_(i_vals, adag(i_vals) * a(i_vals)), n_orbitals=2).to_dense())
    assert np.allclose(m_size, m_vals)


def test_typed_domain_compiles_to_same_matrix_as_explicit_orbitals():
    occ = domain("occ", size=2)
    i = index("i", occ)
    typed = compile(sum_(i, adag(i) * a(i)), n_orbitals=4)
    explicit = compile(adag(0) * a(0) + adag(1) * a(1), n_orbitals=4)
    assert np.allclose(np.array(typed.to_dense()), np.array(explicit.to_dense()))


def test_overlapping_domains_resolve_delta_at_finite_expansion():
    occ = domain("occ", values=[0, 1, 2])
    act = domain("act", values=[1, 2, 3])
    i = index("i", occ)
    av = index("a", act)

    expanded = expand_sums(sum_(i, av, delta(i, av) * adag(i) * a(av)), n_orbitals=4)
    pairs = sorted((term.ops[0].mode.value, term.ops[1].mode.value) for term in expanded.terms)
    # The retained delta survives canonicalization and only the coincident
    # orbitals in the overlap {1, 2} contribute once expanded.
    assert pairs == [(1, 1), (2, 2)]


def test_domains_override_accepts_domain_size_and_values_specs():
    # The single ``domains=`` knob resolves string domains from a Domain object,
    # an int size, or an iterable of orbital labels.
    i = index("i", "band")
    j = index("j", "shell")
    k = index("k", "core")

    expanded = expand_sums(
        sum_(i, j, k, adag(i) * a(j) * adag(k)),
        n_orbitals=6,
        domains={
            "band": domain("band", size=2, start=4),
            "shell": [0, 3],
            "core": 1,
        },
    )
    triples = sorted(
        (term.ops[0].mode.value, term.ops[1].mode.value, term.ops[2].mode.value)
        for term in expanded.terms
    )
    assert triples == [(4, 0, 0), (4, 3, 0), (5, 0, 0), (5, 3, 0)]


def test_override_for_already_finite_domain_is_rejected():
    virt = domain("virt", size=2, start=2)
    av = index("a", virt)

    # A name-keyed override must not silently shadow the concrete start=2 domain.
    with pytest.raises(ValueError, match="already finite"):
        expand_sums(sum_(av, adag(av)), n_orbitals=4, domains={"virt": 2})


def test_domain_orbital_outside_n_orbitals_is_rejected():
    band = index("i", "band")
    with pytest.raises(ValueError, match="outside"):
        expand_sums(sum_(band, adag(band)), n_orbitals=2, domains={"band": [0, 5]})
