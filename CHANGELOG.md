# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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

[0.2.0]: https://github.com/keyehzy/nomad/releases/tag/v0.2.0
[0.1.0]: https://github.com/keyehzy/nomad/releases/tag/v0.1.0
