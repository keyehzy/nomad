# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).


## [Unreleased]

### Added
- Typed index domains via ``Domain``/``domain()`` and ``index``/``indices``
  ``domain=...`` arguments.  Finite expansion now uses domain-specific ranges
  or values, supports a per-call ``domains`` override (mapping a domain name to
  a ``Domain``, an ``int`` size, or an iterable of orbital labels) for
  not-yet-finite (e.g. string-only) domains, preserves domain
  metadata under hygienic dummy renaming, and evaluates typed tensor values
  using domain-local axes while operators use global orbital labels.  Domain
  identity is canonical: domains sharing a name and the same ordered global
  labels compare equal whether written as ``size``/``start`` or as an explicit
  ``values`` list, so equivalent forms merge (and their δ unifies) rather than
  being kept distinct.  Supplying an override for an already-finite domain is
  rejected so a concrete domain's ``start``/``values`` are never silently
  shadowed by a name collision.
- Cross-domain delta handling now avoids unsafe symbolic contraction; deltas
  across incompatible finite domains can collapse to zero, while potentially
  overlapping domains are retained until concrete finite expansion.

## [0.3.0] - 2026-06-28

### Changed
- **BREAKING:** Index display names of the form `_<digits>` are now reserved for
  canonical bound dummies. Constructing a free index with such a name raises
  `ValueError` — whether directly (`Index("_0")`, `index("_0")`) or through any
  path that coerces a string into a free index (e.g. `adag("_0")`,
  `delta("_0", q)`, or a string entry in `Term.summed`). A free index can
  therefore no longer visually collide with a rendered bound dummy.
- Index metadata (`spin_z2`, `momentum`) now participates in an index's
  structural identity. Two free indices that share a display name but differ in
  metadata (e.g. `index("p")` vs `spin_index("p", spin_z2=1)`) are now distinct
  modes — previously they compared equal. As a consequence, `sum_` binds by
  structural identity, so a binder only captures body occurrences whose metadata
  matches. Note such indices still *render* by name alone, so they can look
  identical in `text`/`latex`/`dumps_json` output.

### Fixed
- `sum_` bound indices are now hygienic. Each bound summation index carries a
  hidden, globally-unique identity separate from its human-readable display
  name, so capture-avoidance and alpha-renaming are structural rather than
  name-based. Multiplying two independently-bound sums that reuse the same
  display label (e.g. `sum_(p, adag(p)) * sum_(p, a(p))`) no longer collides
  them into a single index; they correctly expand as a Cartesian product.
- Canonicalization now iterates to a fixed point, making `simplify()` idempotent
  and independent of the order indices are listed in `sum_` for terms with
  same-kind operator runs (e.g. `a†_p a†_q`) over bound dummies. Previously the
  canonical dummy renaming ran *after* the sign-bearing operator/tensor sorts,
  so equality and term de-duplication could depend on that (irrelevant) order.

## [0.2.0] - 2026-06-27

### Added
- `index(name) -> Index` for constructing a single symbolic index.
- `py.typed` marker so downstream consumers receive the package's type
  information (PEP 561).
- Development tooling: Ruff (lint + format), mypy in strict mode, and a
  GitHub Actions CI workflow (lint/type-check plus pytest on Python 3.10–3.14).

### Changed
- **BREAKING:** `indices()` now always returns `tuple[Index, ...]`. Previously a
  single name returned a bare `Index`; that convenience moved to the new
  `index()`. Migrate single-name calls from `p = indices("p")` to
  `p = index("p")` (or `(p,) = indices("p")`).

### Fixed
- `__version__` is read from installed package metadata, giving a single source
  of truth and resolving a mismatch (`0.1.0-v1` in code vs `0.1.0` in packaging).

## [0.1.0] - 2026-06-27

Initial release — a validated implementation of NOMAD as a multi-level compiler
prototype for fermionic second-quantized algebra.

### Added
- Python embedded DSL: `indices`, `tensor`, `adag`/`create`, `a`/`destroy`, `sum_`.
- NCIR-like term IR: weighted sums of tensor/operator/delta graph terms.
- Fermionic CAR normal ordering: `a_i a†_j -> δ_ij - a†_j a_i`.
- Delta elimination and symbolic equality constraints with summation-domain
  reduction.
- Dummy-index canonicalization with deterministic `_0`, `_1`, ... names.
- Same-kind fermion canonicalization and exact sign accounting.
- Tensor antisymmetric-pair canonicalization.
- U(1) particle-number pruning.
- Finite-basis sum expansion and tensor evaluation.
- Matrix-free determinant-space sparse backend over bitstring Slater determinants.
- LaTeX and OpenFermion-source exports.

[0.3.0]: https://github.com/keyehzy/nomad/releases/tag/v0.3.0
[0.2.0]: https://github.com/keyehzy/nomad/releases/tag/v0.2.0
[0.1.0]: https://github.com/keyehzy/nomad/releases/tag/v0.1.0
