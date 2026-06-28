import json

import numpy as np
import pytest

from nomad import *
from nomad.io import dumps_json

# ---------------------------------------------------------------------------
# Charge construction and validation
# ---------------------------------------------------------------------------


def test_charge_helper_builds_per_orbital_vector():
    c = charge([0, 1, 2, 3])
    assert isinstance(c, Charge)
    assert c.values == (0, 1, 2, 3)
    assert c.modulus is None


def test_charge_carries_optional_modulus():
    c = charge([0, 1, 2, 3], modulus=4)
    assert c.values == (0, 1, 2, 3)
    assert c.modulus == 4


def test_charge_modulus_must_be_a_positive_integer():
    with pytest.raises(ValueError, match="positive integer"):
        Charge((1, 2), modulus=0)
    with pytest.raises(ValueError, match="positive integer"):
        charge([1, 2], modulus=-3)
    with pytest.raises(ValueError, match="must be an integer"):
        charge([1, 2], modulus=2.5)


def test_charge_values_must_be_an_iterable_of_integers():
    # A bare string is a classic accidental iterable; it must be rejected rather
    # than silently expanded into character "charges".
    with pytest.raises(TypeError, match="iterable of integer"):
        Charge("abc")


# ---------------------------------------------------------------------------
# basis_sector
# ---------------------------------------------------------------------------


def test_basis_sector_filters_particle_number():
    assert basis_sector(4, target={"N": 2}) == (3, 5, 6, 9, 10, 12)


def test_basis_sector_defaults_to_alternating_spin_for_sz2():
    # With no explicit Sz2 charge vector the legacy alternating (+1, -1) labels
    # apply, so the Sz2 = 0 sector of two orbitals is the empty and doubly
    # occupied determinants.
    assert basis_sector(2, target={"Sz2": 0}) == (0, 3)


def test_basis_sector_honours_explicit_spin_z2_vector():
    # An explicit spin vector overrides the alternating default.
    full = basis_sector(2, target={"Sz2": 2}, spin_z2=[1, 1])
    assert full == (3,)  # both orbitals spin-up: only the doubly occupied det


def test_basis_sector_combines_multiple_charges():
    charges = {
        "N": [1, 1, 1, 1],
        "Sz2": [1, -1, 1, -1],
        "K": charge([0, 1, 2, 3], modulus=4),
    }
    assert basis_sector(n_orbitals=4, charges=charges, target={"N": 2, "Sz2": 0, "K": 1}) == (3, 12)


def test_basis_sector_supports_modular_parity_charge():
    # P is a Z2 parity charge: orbitals {2, 3} are odd, {0, 1} even.  The P = 0
    # sector keeps determinants occupying an even number of odd orbitals.
    out = basis_sector(4, charges={"P": charge([0, 0, 1, 1], modulus=2)}, target={"P": 0})
    assert out == (0, 1, 2, 3, 12, 13, 14, 15)


def test_basis_sector_empty_target_returns_full_basis():
    assert basis_sector(2) == (0, 1, 2, 3)


def test_basis_sector_requires_charge_table_for_custom_target():
    with pytest.raises(ValueError, match="No per-orbital charge values"):
        basis_sector(4, target={"K": 1})


def test_basis_sector_rejects_charge_vector_of_wrong_length():
    with pytest.raises(ValueError, match="expected n_orbitals=4"):
        basis_sector(4, charges={"K": [0, 1, 2]}, target={"K": 0})


def test_basis_sector_accepts_mapping_charge_specification():
    # The dict form {"values": ..., "modulus": ...} is equivalent to charge().
    spec = basis_sector(4, charges={"K": {"values": [0, 1, 2, 3], "modulus": 4}}, target={"K": 1})
    assert spec == basis_sector(4, charges={"K": charge([0, 1, 2, 3], modulus=4)}, target={"K": 1})


def test_basis_sector_accepts_tuple_charge_specification():
    # The (values, modulus) pair form is equivalent to charge(values, modulus=...).
    spec = basis_sector(4, charges={"K": ([0, 1, 2, 3], 4)}, target={"K": 1})
    assert spec == basis_sector(4, charges={"K": charge([0, 1, 2, 3], modulus=4)}, target={"K": 1})


# ---------------------------------------------------------------------------
# determinant_charges
# ---------------------------------------------------------------------------


def test_determinant_charges_defaults_to_particle_count():
    assert determinant_charges(0b101, n_orbitals=4) == {"N": 2}


def test_determinant_charges_uses_supplied_tables():
    charges = {"Sz2": [1, -1, 1, -1], "K": charge([0, 1, 2, 3], modulus=4)}
    # det 12 = 0b1100 occupies orbitals 2 and 3.
    assert determinant_charges(0b1100, n_orbitals=4, charges=charges) == {"K": 1, "N": 2, "Sz2": 0}


def test_determinant_charges_rejects_out_of_range_determinant():
    with pytest.raises(ValueError, match="does not fit in n_orbitals=2"):
        determinant_charges(99, n_orbitals=2)


# ---------------------------------------------------------------------------
# operator / term / expression charge deltas
# ---------------------------------------------------------------------------


def test_operator_charge_delta_uses_orbital_table_and_particle_number():
    op = Op("destroy", Orbital(2))
    delta = operator_charge_delta(op, charges={"K": charge([0, 1, 2, 3], modulus=4)})
    # Annihilating orbital 2 removes momentum 2 (≡ -2 ≡ 2 mod 4) and one particle.
    assert delta == {"K": 2, "N": -1}


def test_term_charge_delta_reports_structural_cancellation_for_symbolic_index():
    # a†(p) a(p) carries the same symbolic index on both ends, so even an unknown
    # per-orbital charge K provably cancels and is reported as zero.
    p = index("p")
    term = (adag(p) * a(p)).terms[0]
    assert term_charge_delta(term, charges={"K": [0, 1, 2, 3]}) == {"K": 0, "N": 0}


def test_term_charge_delta_omits_genuinely_unknown_charges():
    # Distinct symbolic indices leave K unknown; it is omitted so pruning never
    # discards the term, while N (always known) is still reported.
    p, q = indices("p q")
    term = (adag(p) * a(q)).terms[0]
    assert term_charge_delta(term, charges={"K": [0, 1, 2, 3]}) == {"N": 0}


def test_term_charge_delta_counts_particle_number_for_creation_word():
    assert term_charge_delta((adag(0) * adag(1)).terms[0]) == {"N": 2}


def test_charge_delta_dispatches_over_op_term_and_expr():
    assert charge_delta(Op("create", Orbital(0)), charges={"K": [0, 1, 2, 3]}) == {"K": 0, "N": 1}
    assert charge_delta((adag(0) * a(1)).terms[0]) == {"N": 0}
    # A whole expression collapses to a single common delta when its terms agree.
    assert charge_delta(adag(0) * a(1) + adag(2) * a(3)) == {"N": 0}


def test_charge_delta_rejects_expression_with_inconsistent_deltas():
    with pytest.raises(ValueError, match="different charge deltas"):
        charge_delta(adag(0) * a(1) + adag(0) * adag(1))


def test_op_and_term_charge_delta_methods_match_module_functions():
    op = adag(0).terms[0].ops[0]
    assert op.charge_delta(charges={"K": [0, 1, 2, 3]}) == operator_charge_delta(
        op, charges={"K": [0, 1, 2, 3]}
    )
    term = (adag(0) * a(1)).terms[0]
    assert term.charge_delta() == term_charge_delta(term)


# ---------------------------------------------------------------------------
# prune_by_charge (generalized)
# ---------------------------------------------------------------------------


def test_prune_by_charge_legacy_particle_number_default():
    p = index("p")
    expr = adag(p) * a(p) + adag(p) + a(p)
    assert text(prune_by_charge(expr)) == "a†(p) a(p)"


def test_prune_by_charge_target_requires_conservation_and_keeps_unknowns():
    # target={"N": 0} prunes the number-changing pairing term but retains both the
    # number-conserving concrete term and the symbolic term whose charge is unknown.
    p, q = indices("p q")
    expr = adag(0) * adag(1) + adag(0) * a(1) + adag(p) * a(q)
    assert text(prune_by_charge(expr, target={"N": 0})) == "a†(0) a(1) + a†(p) a(q)"


def test_prune_by_charge_supports_nonzero_required_delta():
    p, q = indices("p q")
    expr = adag(0) * adag(1) + adag(0) * a(1) + adag(p) * a(q)
    # delta={"N": 2} keeps only the term that creates two particles; delta_n=None
    # disables the implicit ΔN = 0 default that would otherwise contradict it.
    assert text(prune_by_charge(expr, delta={"N": 2}, delta_n=None)) == "a†(0) a†(1)"


def test_prune_by_charge_delta_n_none_disables_default_and_keeps_all():
    p, q = indices("p q")
    expr = adag(0) * adag(1) + adag(0) * a(1) + adag(p) * a(q)
    assert len(prune_by_charge(expr, delta_n=None).terms) == 3


def test_prune_by_charge_conserves_modular_charge():
    # Hopping 0 -> 2 changes momentum by 2 (mod 4) and is dropped; the on-site
    # term conserves K.  conserve=[...] is shorthand for "require zero delta".
    expr = adag(2) * a(0) + adag(0) * a(0)
    kept = prune_by_charge(
        expr, conserve=["K"], delta_n=None, charges={"K": charge([0, 1, 2, 3], modulus=4)}
    )
    assert text(kept) == "a†(0) a(0)"


# ---------------------------------------------------------------------------
# compile with charges, sectors, and explicit bases
# ---------------------------------------------------------------------------


def _ring_hopping(n):
    return sum(adag(i) * a((i + 1) % n) + adag((i + 1) % n) * a(i) for i in range(n))


def test_compile_sector_matrix_equals_full_basis_restriction():
    # The headline guarantee: a sector-restricted operator is exactly the full
    # operator restricted to the rows/columns of that sector's basis.
    n = 4
    H = _ring_hopping(n)
    full = compile(H, n_orbitals=n)
    sec = compile(H, n_orbitals=n, sector={"N": 2})
    full_basis = list(full.basis)
    rows = [full_basis.index(d) for d in sec.basis]
    sub = np.array(full.to_dense())[np.ix_(rows, rows)]
    assert np.allclose(sub, np.array(sec.to_dense()))


def test_compile_accepts_precomputed_basis_matching_the_sector():
    n = 4
    H = _ring_hopping(n)
    sec = compile(H, n_orbitals=n, sector={"N": 2})
    explicit = compile(H, n_orbitals=n, basis=list(sec.basis))
    assert explicit.basis == sec.basis
    assert np.allclose(np.array(explicit.to_dense()), np.array(sec.to_dense()))


def test_compile_conserves_modular_momentum_sector():
    n = 4
    charges = {"N": [1, 1, 1, 1], "Sz2": [1, -1, 1, -1], "K": charge([0, 1, 2, 3], modulus=4)}
    H = adag(0) * a(0) + adag(2) * a(2)
    op = compile(H, n_orbitals=n, sector={"N": 2, "K": 1}, charges=charges)
    assert op.basis == (3, 12)
    assert op.to_dense() == [[1.0, 0.0], [0.0, 1.0]]


def test_compile_prunes_charge_violating_terms_from_the_hamiltonian():
    # A K-violating hopping (0 -> 2) added to the conserving Hamiltonian must be
    # dropped during compilation rather than corrupt the sector matrix.
    n = 4
    charges = {"N": [1, 1, 1, 1], "K": charge([0, 1, 2, 3], modulus=4)}
    conserving = adag(0) * a(0) + adag(2) * a(2)
    polluted = conserving + adag(2) * a(0)
    sector = {"N": 2, "K": 1}
    clean = compile(conserving, n_orbitals=n, sector=sector, charges=charges)
    pruned = compile(polluted, n_orbitals=n, sector=sector, charges=charges)
    assert np.allclose(np.array(pruned.to_dense()), np.array(clean.to_dense()))


def test_compile_validates_explicit_basis_against_n_orbitals():
    with pytest.raises(ValueError, match="does not fit in n_orbitals=2"):
        compile(adag(0) * a(0), n_orbitals=2, basis=[0, 99])


# ---------------------------------------------------------------------------
# generate_basis backward compatibility
# ---------------------------------------------------------------------------


def test_generate_basis_particle_number_sector_unchanged():
    assert generate_basis(4, {"N": 2}) == (3, 5, 6, 9, 10, 12)


def test_generate_basis_accepts_legacy_sz_alias():
    assert generate_basis(2, {"Sz": 0}) == (0, 3)


def test_generate_basis_accepts_general_charge_tables():
    # The new charges= keyword routes through the same machinery as basis_sector.
    out = generate_basis(4, {"K": 1}, charges={"K": charge([0, 1, 2, 3], modulus=4)})
    assert out == basis_sector(4, charges={"K": charge([0, 1, 2, 3], modulus=4)}, target={"K": 1})


# ---------------------------------------------------------------------------
# Index charge metadata and JSON export
# ---------------------------------------------------------------------------


def test_index_charges_alias_legacy_spin_and_momentum_fields():
    pi = index("p", spin_z2=1, momentum=2)
    assert pi.charges == (("K", 2), ("Sz2", 1))
    # The flexible charges= mapping reaches the identical canonical index.
    assert pi == index("p", charges={"Sz2": 1, "K": 2})


def test_index_charges_accepts_iterable_of_pairs():
    # charges= also accepts an iterable of (name, value) pairs, equivalent to the
    # mapping form.
    assert index("p", charges=[("K", 2), ("Sz2", 1)]) == index("p", charges={"Sz2": 1, "K": 2})


def test_index_rejects_conflicting_legacy_and_charge_metadata():
    with pytest.raises(ValueError, match="conflicts"):
        index("p", spin_z2=1, charges={"Sz2": -1})


def test_index_charge_metadata_participates_in_identity():
    # Two free indices that share a display name but differ in charge metadata are
    # distinct modes, mirroring the existing spin_z2 behaviour.
    p = index("p")
    p_k = index("p", charges={"K": 1})
    assert p != p_k
    expr = (adag(p_k) * a(p)).simplify()
    create_mode, destroy_mode = expr.terms[0].ops[0].mode, expr.terms[0].ops[1].mode
    assert create_mode != destroy_mode


def test_json_export_serializes_summed_index_charges():
    pj = index("p", charges={"K": 2})
    payload = json.loads(dumps_json(sum_(pj, adag(pj))))
    (term,) = payload["terms"]
    assert term["summed_charges"] == {"_0": {"K": 2}}


def test_json_export_omits_summed_charges_when_absent():
    q = index("q")
    (term,) = json.loads(dumps_json(sum_(q, adag(q))))["terms"]
    assert "summed_charges" not in term
