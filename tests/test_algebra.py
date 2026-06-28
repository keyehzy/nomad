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


def test_same_kind_operator_run_canonicalization_is_order_invariant():
    # A same-kind operator run (a† a†) over >=2 bound dummies must canonicalize
    # independently of the order the indices are listed in sum_ (Σ_pq == Σ_qp),
    # and the canonical form must be a fixed point of simplify(). Regression for
    # a bug where _rename_bound_dummies renumbered dummies *after* the
    # sign-bearing operator/tensor sorts, so the result depended on the
    # (irrelevant) binding order. See _canonicalize_term.
    p, q, r, s = indices("p q r s")
    g = tensor("g", [p, q, r, s])
    h1 = sum_(p, q, r, s, g[p, q, r, s] * adag(p) * adag(q) * a(s) * a(r))
    h2 = sum_(q, p, s, r, g[p, q, r, s] * adag(p) * adag(q) * a(s) * a(r))
    assert h1 == h2
    assert h1.simplify() == h1
    assert h2.simplify() == h2


def test_alpha_equivalent_terms_merge_in_a_single_simplify():
    # Two alpha-equivalent operator words must collapse into one term with the
    # coefficient doubled, not survive as two un-merged terms. Regression for the
    # non-idempotent canonicalization above leaking into term de-duplication.
    p, q, r, s = indices("p q r s")
    g = tensor("g", [p, q, r, s])
    word = g[p, q, r, s] * adag(p) * adag(q) * a(s) * a(r)
    relabeled = g[q, p, r, s] * adag(q) * adag(p) * a(s) * a(r)  # == word via p<->q
    merged = sum_(p, q, r, s, word + relabeled)
    single = sum_(p, q, r, s, word)
    assert len(merged.terms) == 1
    assert merged.terms[0].coeff == 2 * single.terms[0].coeff


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


def test_free_indices_differing_only_in_metadata_are_distinct():
    # Index metadata participates in structural identity: two free indices that
    # share a display name but differ in metadata are distinct modes. They still
    # render by name alone, so text() aliases them -- structure does not.
    p = index("p")
    p_up = spin_index("p", spin_z2=1)
    assert p != p_up
    expr = (adag(p_up) * a(p)).simplify()
    create_mode, destroy_mode = expr.terms[0].ops[0].mode, expr.terms[0].ops[1].mode
    assert create_mode != destroy_mode
    assert text(expr) == "a†(p) a(p)"  # display aliases; identity does not


def test_sum_capture_requires_matching_index_metadata():
    # sum_ binds by structural identity, so a binder only captures body
    # occurrences whose metadata matches. A bare binder does not capture a
    # spin-typed body index, leaving it free (and the unused dummy is dropped).
    p = index("p")
    p_up = spin_index("p", spin_z2=1)
    assert text(sum_(p, adag(p_up))) == "a†(p)"
    # Matching metadata captures and renders as a canonical dummy.
    assert text(sum_(p_up, adag(p_up))) == "Σ__0 a†(_0)"


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


def test_typed_index_domain_constructors_and_rendering():
    occ = domain("occ", size=2)
    virt = domain("virt", size=2, start=2)
    i, j = indices("i j", domain=occ)
    av = index("a", virt)
    p = index("p")

    assert isinstance(occ, Domain)
    assert i.domain == occ
    assert av.domain == virt
    assert p.domain.name == "spin_orbital"
    assert i != index("i", virt)

    rendered = text(sum_(i, av, adag(av) * a(i)))
    assert "∈occ" in rendered
    assert "∈virt" in rendered
    assert text(sum_(i, j, delta(i, j) * adag(i) * a(j))) == "Σ__0∈occ a†(_0) a(_0)"


def test_cross_domain_delta_with_disjoint_finite_domains_is_zero():
    occ = domain("occ", values=[0, 1])
    virt = domain("virt", values=[2, 3])
    i = index("i", occ)
    av = index("a", virt)

    assert not sum_(i, av, delta(i, av) * adag(i) * a(av)).terms


def test_cross_domain_delta_with_overlapping_finite_domains_is_retained():
    occ = domain("occ", values=[0, 1, 2])
    act = domain("act", values=[1, 2, 3])
    i = index("i", occ)
    av = index("a", act)

    (term,) = sum_(i, av, delta(i, av) * adag(i) * a(av)).terms
    # Domains may overlap, so the delta cannot be consumed symbolically: both
    # dummies remain bound and the constraint is kept for finite expansion.
    assert len(term.deltas) == 1
    assert len(term.summed) == 2


def test_normal_order_drops_delta_between_disjoint_domains():
    occ = domain("occ", size=2)
    virt = domain("virt", size=2, start=2)
    i = index("i", occ)
    av = index("a", virt)

    # a_i a†_a normal-orders to δ_ia - a†_a a_i; with disjoint occ/virt domains
    # the contraction term vanishes, leaving only the swapped product.
    out = normal_order(sum_(i, av, a(i) * adag(av)))
    (term,) = out.terms
    assert not term.deltas
    assert term.coeff == -1


def test_normal_order_keeps_contraction_for_default_domains():
    p, q = indices("p q")
    out = normal_order(sum_(p, q, a(p) * adag(q)))
    # Default spin-orbital indices may coincide, so the δ contraction survives
    # as a constant term alongside the swapped product.
    assert any(not term.ops for term in out.terms)
