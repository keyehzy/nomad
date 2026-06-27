# NOMAD — Normal-Ordered Matrix-free Algebra Device

This repository contains a validated implementation of NOMAD as a multi-level
compiler prototype for fermionic second-quantized algebra.

Scope implemented here:

- Python embedded DSL: `index/indices`, `tensor`, `adag/create`, `a/destroy`, `sum_`.
- NCIR-like term IR: weighted sums of tensor/operator/delta graph terms.
- Fermionic CAR normal ordering: `a_i a†_j -> δ_ij - a†_j a_i`.
- Delta elimination and symbolic equality constraints with summation-domain reduction.
- Dummy-index canonicalization with deterministic `_0`, `_1`, ... names.
- Same-kind fermion canonicalization and exact sign accounting.
- Tensor antisymmetric-pair canonicalization.
- U(1) particle-number pruning.
- Finite-basis sum expansion and tensor evaluation.
- Matrix-free determinant-space sparse backend over bitstring Slater determinants.
- LaTeX and OpenFermion-source exports.

NOMAD intentionally does **not** include a standalone `qoal` parser, bosonic CCR,
full point-group irreps, HFB/Bogoliubov approximation passes, QIR, or native
hardware routing. The Python DSL is the source frontend.

## Install/use from source

```bash
python -m pip install -e ".[test]"
python -m pytest
nomad demo --emit latex
```

This is a pure-Python package with no compiled extensions; installing requires
only a Python toolchain. The `[test]` extra pulls in `pytest` and `numpy` for the
test suite.

## Example

```python
from nomad import *

p, q, r, s = indices("p q r s")
h = tensor("h", [p, q], hermitian=True)
g = tensor("g", [p, q, r, s], antisymmetric_pairs=[(p, q), (r, s)])

H1 = sum_(p, q, h[p, q] * adag(p) * a(q))
H2 = 0.25 * sum_(p, q, r, s, g[p, q, r, s] * adag(p) * adag(q) * a(s) * a(r))
H = H1 + H2

print(latex(normal_order(H)))
```

Finite sparse compilation example:

```python
from nomad import *

H = 2 * adag(0) * a(0) + 3 * adag(1) * a(1)
op = compile(H, target="sparse", n_orbitals=2, sector={"N": 1})
print(op.basis)       # (1, 2)
print(op.to_dense())  # [[2.0, 0.0], [0.0, 3.0]]
```

## Architecture

```text
Python DSL
  ↓
Typed symbolic expression objects
  ↓
NCIR/Wick term sum: coeff × tensors × op-word × deltas × constraints
  ↓
normal ordering → delta elimination → dummy-index canonicalization
  ↓
LaTeX / OpenFermion-source / matrix-free determinant backend
```

The reference runtime in `python/nomad/` is a small, deterministic,
dependency-light implementation of the core term algebra and determinant kernels.
