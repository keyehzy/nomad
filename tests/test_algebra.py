import pytest

from nomad import *


def test_car_identity_normal_orders_to_delta():
    i, j = indices("i j")
    expr = a(i) * adag(j) + adag(j) * a(i)
    out = normal_order(expr)
    assert text(out) == "δ(i,j)"
    assert latex(out) == r"\delta_{i,j}"


def test_annihilator_anticommutator_cancels():
    i, j = indices("i j")
    assert not normal_order(a(i) * a(j) + a(j) * a(i)).terms


def test_creator_anticommutator_cancels():
    i, j = indices("i j")
    assert not normal_order(adag(i) * adag(j) + adag(j) * adag(i)).terms


def test_delta_elimination_reduces_summation_domain():
    p, q = indices("p q")
    h = tensor("h", [p, q])
    expr = sum_(p, q, delta(p, q) * h[p, q] * adag(p) * a(q))
    out = normal_order(expr)
    assert text(out) == "Σ__0 h[_0,_0] a†(_0) a(_0)"


def test_dummy_index_canonicalization_equates_alpha_renamed_sums():
    p, q = indices("p q")
    i, j = indices("i j")
    h = tensor("h", 2)
    e1 = sum_(p, q, h[p, q] * adag(p) * a(q))
    e2 = sum_(i, j, h[i, j] * adag(i) * a(j))
    assert e1.simplify() == e2.simplify()


def test_multiplication_freshens_independent_bound_indices():
    p, q = indices("p q")
    expr = sum_(p, adag(p)) * sum_(q, a(q))
    assert text(expr) == "Σ__0,_1 a†(_0) a(_1)"
    assert len(expr.terms[0].summed) == 2


def test_independent_bound_indices_expand_as_cartesian_product():
    p, q = indices("p q")
    expr = sum_(p, adag(p)) * sum_(q, a(q))
    out = expand_sums(expr, n_orbitals=2)
    assert text(out) == "a†(0) a(0) + a†(0) a(1) + a†(1) a(0) + a†(1) a(1)"


def test_reserved_dummy_namespace_rejects_free_indices():
    # `_<digits>` is reserved for canonical bound dummies (see _rename_bound_dummies),
    # so a free index may not use it and collide with a rendered dummy.
    for reserved in ("_0", "_1", "_42"):
        with pytest.raises(ValueError):
            Index(reserved)
        with pytest.raises(ValueError):
            index(reserved)
    # Names that only resemble the reserved namespace stay valid.
    assert index("_p").name == "_p"
    assert index("p0").name == "p0"
    assert index("_").name == "_"
    # Canonical dummies still render in the reserved namespace.
    p = index("p")
    assert text(sum_(p, adag(p))) == "Σ__0 a†(_0)"


def test_tensor_antisymmetric_pair_canonicalization():
    p, q = indices("p q")
    g = tensor("g", [p, q], antisymmetric_pairs=[(p, q)])
    assert not (g[p, q] + g[q, p]).terms


def test_particle_number_pruning():
    p = index("p")
    expr = adag(p) * a(p) + adag(p) + a(p)
    pruned = prune_by_charge(expr, delta_n=0)
    assert text(pruned) == "a†(p) a(p)"


def test_index_and_indices_constructors():
    p = index("p")
    assert isinstance(p, Index)

    pair = indices("p q")
    assert isinstance(pair, tuple)
    assert [m.name for m in pair] == ["p", "q"]

    # indices() always returns a tuple, even for a single name
    (only,) = indices("p")
    assert only.name == "p"

    # index() rejects multiple names
    with pytest.raises(ValueError):
        index("p q")
