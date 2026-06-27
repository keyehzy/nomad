# Validation notes

Validated in the delivery environment with:

```bash
PYTHONPATH=/mnt/data/nomad-v1/python python -m compileall -q /mnt/data/nomad-v1/python
PYTHONPATH=/mnt/data/nomad-v1/python python -m pytest -q /mnt/data/nomad-v1/tests
```

The tests cover:

- CAR normal-ordering identities.
- Same-kind fermion anti-commutation cancellation.
- Delta elimination under symbolic summation.
- Dummy-index canonical equality.
- Tensor antisymmetry canonicalization.
- Particle-number charge pruning.
- Determinant sign handling.
- Matrix-free sparse backend against an explicitly generated dense matrix.
- Finite tensor-valued sum expansion.
- LaTeX and OpenFermion-source exports.
