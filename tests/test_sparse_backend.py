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


def test_finite_typed_domain_expands_without_n_orbitals():
    # Headline: n_orbitals is optional once every summed index carries a finite
    # domain, since each sum already knows its own range. Both the size/start
    # form and the explicit values form expand on their own.
    i = index("i", domain("occ", size=2))  # globals {0, 1}
    assert text(expand_sums(sum_(i, adag(i) * a(i)))) == "a†(0) a(0) + a†(1) a(1)"

    j = index("j", domain("act", values=[1, 3]))
    assert text(expand_sums(sum_(j, adag(j)))) == "a†(1) + a†(3)"


def test_empty_domain_expands_to_zero():
    # A size-0 domain has no expansion points, so the Cartesian product is empty
    # and every term carrying that index drops out.
    empty = index("z", domain("none", size=0))
    assert not expand_sums(sum_(empty, adag(empty) * a(empty))).terms


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


def test_typed_domain_mapping_tensor_values_use_domain_local_keys():
    occ = domain("occ", size=2)
    virt = domain("virt", size=2, start=2)
    i = index("i", occ)
    av = index("a", virt)
    t = tensor("t", [av, i])

    # A Mapping is keyed by domain-local coordinates, exactly like the array
    # and callable forms -- not by the global orbital labels (2, 3).
    expanded = expand_sums(
        sum_(av, i, t[av, i] * adag(av) * a(i)),
        n_orbitals=4,
        tensor_values={"t": {(0, 0): 10, (0, 1): 11, (1, 0): 20, (1, 1): 21}},
    )
    coeffs = {
        (term.ops[0].mode.value, term.ops[1].mode.value): int(term.coeff)
        for term in expanded.terms
    }
    assert coeffs == {(2, 0): 10, (2, 1): 11, (3, 0): 20, (3, 1): 21}


def test_typed_domain_mapping_tensor_values_reject_global_keys():
    virt = domain("virt", size=2, start=2)
    av = index("a", virt)
    t = tensor("t", [av])

    # Keying by the global labels (2, 3) instead of the local (0, 1) is a user
    # mistake that must surface, not be silently papered over by a fallback.
    with pytest.raises(KeyError):
        expand_sums(
            sum_(av, t[av] * adag(av)),
            n_orbitals=4,
            tensor_values={"t": {(2,): 1, (3,): 2}},
        )


def test_delta_pinned_typed_index_keeps_domain_local_tensor_axis():
    # Regression: when a delta pins a typed dummy to a concrete in-domain
    # orbital, _canonicalize_delta_constraints substituted a bare Orbital that
    # dropped the domain-local provenance, so the tensor was evaluated at the
    # *global* axis (here 3) instead of the local one (1).  With a compact
    # local-sized array that raised IndexError; with a larger array it silently
    # returned the wrong coefficient.
    virt = domain("virt", size=2, start=2)  # globals {2, 3}, local {0, 1}
    av = index("a", virt)
    t = tensor("t", [av])

    expanded = expand_sums(
        sum_(av, delta(av, 3) * t[av] * adag(av)),
        n_orbitals=4,
        tensor_values={"t": [100, 200]},  # local 0 -> 100, local 1 -> 200
    )
    (term,) = expanded.terms
    assert term.ops[0].mode.value == 3  # operator keeps the global label
    assert int(term.coeff) == 200  # tensor addressed by local axis 1, not global 3


def test_normal_order_contraction_pin_keeps_domain_local_tensor_axis():
    # The same pin arises from an ordinary contraction: a(i) a†(2) normal-orders
    # to δ_{i,2} - a†(2) a(i), and δ_{i,2} pins i to global orbital 2 (= local 0
    # in virt).  The surviving t[i] contraction term must evaluate at local 0.
    virt = domain("virt", size=2, start=2)
    i = index("i", virt)
    t = tensor("t", [i])

    lowered = expand_sums(
        normal_order(sum_(i, t[i] * a(i) * adag(2))),
        n_orbitals=4,
        tensor_values={"t": [100, 200]},  # local 0 -> 100
    )
    constant = next(term for term in lowered.terms if not term.ops)
    assert int(constant.coeff) == 100


def test_shared_constant_pins_each_domain_to_its_own_local_axis():
    # One global label can pin dummies living in different offset domains; each
    # must resolve its tensor against its *own* local coordinate, which a single
    # shared Orbital could not express.  Global 2 is local 1 in occ={0,2} but
    # local 0 in virt={2,3}.
    occ = domain("occ", values=[0, 2])
    virt = domain("virt", size=2, start=2)
    i = index("i", occ)
    av = index("a", virt)
    u = tensor("u", [i])
    w = tensor("w", [av])

    expanded = expand_sums(
        sum_(i, av, delta(i, 2) * delta(av, 2) * u[i] * w[av]),
        n_orbitals=4,
        tensor_values={"u": [10, 20], "w": [30, 40]},
    )
    (term,) = expanded.terms
    assert int(term.coeff) == 20 * 30  # u@local1 * w@local0


def test_symbolic_typed_tensor_keeps_local_axis_across_offset_domains():
    # Regression: a tensor left symbolic through expand_sums is addressed by its
    # domain-local axis, so two offset domains that share a global orbital label
    # at *different* local axes must stay distinct, not merge by global label.
    # occ-local 1 and virt-local 0 both sit at global orbital 2; the symbolic
    # port t[1] (occ) and t[0] (virt) must not collapse into a bogus 2*t[2].
    occ = domain("occ", values=[0, 2])
    virt = domain("virt", size=2, start=2)
    i = index("i", occ)
    av = index("a", virt)
    t = tensor("t", [i])

    e = expand_sums(sum_(i, t[i] * adag(i)) + sum_(av, t[av] * adag(av)), n_orbitals=4)
    # Tensor ports render by local axis; operators keep the global label.
    assert text(e) == "t[0] a†(0) + t[0] a†(2) + t[1] a†(2) + t[1] a†(3)"

    # Operators still merge by global label and numeric evaluation is unchanged:
    # global orbital 2 sums both local contributions (local 0 -> 10, local 1 -> 20).
    ev = expand_sums(
        sum_(i, t[i] * adag(i)) + sum_(av, t[av] * adag(av)),
        n_orbitals=4,
        tensor_values={"t": [10, 20]},
    )
    assert text(ev) == "10*a†(0) + 30*a†(2) + 20*a†(3)"


def test_antisymmetric_typed_tensor_survives_same_global_distinct_local_axes():
    # Regression: antisymmetric-pair canonicalization compares ports by local
    # tensor axis, not global label. Two ports on the same global orbital 2 but
    # different local axes (occ-local 1, virt-local 0) are off-diagonal, so the
    # factor must NOT vanish -- pre-fix the global labels compared equal and the
    # whole term was wrongly dropped to zero.
    occ = domain("occ", values=[0, 2])
    virt = domain("virt", size=2, start=2)
    i = index("i", occ)
    av = index("a", virt)
    g = tensor("g", [i, av], antisymmetric_pairs=[(0, 1)])

    out = sum_(i, av, delta(i, 2) * delta(av, 2) * g[i, av])
    (term,) = out.terms
    assert [p.value for p in term.tensors[0].ports] == [2, 2]  # both global orbital 2
    # Evaluated against an antisymmetric local-axis array, it gives the
    # off-diagonal entry A[occ-local 1][virt-local 0] = A[1][0] = -7, not zero.
    val = expand_sums(out, n_orbitals=4, tensor_values={"g": [[0, 7], [-7, 0]]})
    assert text(val) == "-7"


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


def test_concrete_domain_orbital_outside_n_orbitals_is_rejected():
    # The n_orbitals upper-bound guard also covers a domain's own finite labels,
    # not just override-supplied ones: virt = {3, 4} overflows n_orbitals=4.
    hi = index("a", domain("virt", size=2, start=3))
    with pytest.raises(ValueError, match="outside"):
        expand_sums(sum_(hi, adag(hi)), n_orbitals=4)


def test_string_domain_without_finite_data_or_override_is_rejected():
    # A string-only domain is not yet finite; expanding it needs either a
    # domains= override or the n_orbitals fallback (which only applies to the
    # default spin_orbital basis). With neither, expansion must refuse.
    s = index("s", "band")
    with pytest.raises(ValueError, match="no finite size or values"):
        expand_sums(sum_(s, adag(s)))
