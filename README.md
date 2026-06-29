# NOMAD — Normal-Ordered Matrix-free Algebra Device

NOMAD is a multi-level compiler prototype for fermionic second-quantized
algebra, with a Python DSL as its source frontend.

See [CHANGELOG.md](CHANGELOG.md) for the implemented feature set and
[TODO.md](TODO.md) for planned and out-of-scope work.

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

Fixed-particle determinant bases can be generated lazily from combinations,
without scanning the full ``2**n`` bitstring space:

```python
basis = determinant_basis(n_orbitals=40, N=20)
basis = determinant_basis(spin_up_orbs=20, spin_down_orbs=20, N_up=10, N_down=10)
```

General additive-charge sectors are supported through ``basis_sector`` and the
same ``charges=...`` table can be passed to ``compile``.  Each charge is a
per-orbital vector; use ``charge(..., modulus=L)`` for cyclic quantum numbers
such as crystal momentum or parity:

```python
from nomad import *

n = 4
orbital_N = [1, 1, 1, 1]
orbital_spin_z2 = [1, -1, 1, -1]
orbital_momentum = charge([0, 1, 2, 3], modulus=4)

charges = {
    "N": orbital_N,
    "Sz2": orbital_spin_z2,
    "K": orbital_momentum,
}

basis = basis_sector(
    n_orbitals=n,
    charges=charges,
    target={"N": 2, "Sz2": 0, "K": 1},
)

H = adag(0) * a(0) + adag(2) * a(2)
op = compile(H, target="sparse", n_orbitals=n, sector={"N": 2, "K": 1}, charges=charges)
```

Symbolic indices can also carry charge metadata, for example
``index("p", charges={"K": 0, "irrep": 1})``.  ``term_charge_delta`` /
``operator_charge_delta`` expose the known charge deltas, and
``prune_by_charge`` removes terms that provably violate requested conservation
laws while keeping terms whose symbolic charge is not yet known.


### Typed index domains

Indices can carry finite domains so symbolic sums expand over the intended subset
instead of the whole spin-orbital basis:

```python
from nomad import *

n_occ, n_virt = 2, 3
occ = domain("occ", size=n_occ)
virt = domain("virt", size=n_virt, start=n_occ)

i, j = indices("i j", domain=occ)
a_, b = indices("a b", domain=virt)
t = tensor("t", [a_, i])

T1 = sum_(a_, i, t[a_, i] * adag(a_) * a(i))
expanded = expand_sums(T1, n_orbitals=n_occ + n_virt)
```

``domain(name, size=..., start=...)`` uses global orbital labels
``start..start+size-1`` for operators.  Tensor values are indexed by each
domain's local coordinate, so a tensor over ``virt × occ`` may be supplied as an
``n_virt × n_occ`` array even when virtual orbital labels start after the
occupied block.  For arbitrary subsets use ``domain("active", values=[...])``.
String-only domains such as ``indices("i j", domain="occ")`` can be made finite
at expansion time with ``domains={"occ": n_occ}`` (an ``int`` size),
``domains={"occ": [0, 2, 5]}`` (explicit orbital labels), or
``domains={"occ": domain("occ", size=n_occ, start=...)}`` (a full ``Domain``).
The override is matched by the ``domains`` key, so when it is a ``Domain`` only
its range/values are used and its own name is ignored. An override for an
already-finite domain — or for the default ``spin_orbital`` basis, which is sized
by ``n_orbitals`` — is rejected rather than silently applied.

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

## Development

Install the tooling and run the same checks CI does:

```bash
python -m pip install -e ".[test,dev]"
ruff format .     # auto-format
ruff check .      # lint
mypy              # strict type-check
pytest            # tests
```

A pre-commit hook that runs these automatically lives in `.githooks/`. Enable it
once per clone (it is a local Git setting, so cloning alone does not activate it):

```bash
git config core.hooksPath .githooks
```

The hook auto-formats staged Python files with `ruff format` and re-stages them,
then blocks the commit if lint, types, or tests fail. Use `git commit --no-verify`
to bypass it for work-in-progress commits.
