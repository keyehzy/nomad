"""NOMAD: Normal-Ordered Matrix-free Algebra Device.

The public API is intentionally compact: build expressions with a Python DSL,
normal-order them, export them, or compile finite-basis expressions into a
matrix-free determinant-space operator.
"""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version

from .core import (
    Delta,
    Domain,
    Expr,
    Index,
    Op,
    Orbital,
    SparseOperator,
    TensorFactor,
    TensorSymbol,
    Term,
    a,
    adag,
    apply_ops_to_det,
    compile,
    create,
    delta,
    destroy,
    domain,
    expand_sums,
    generate_basis,
    index,
    indices,
    latex,
    normal_order,
    openfermion,
    particle_delta,
    prune_by_charge,
    spin_index,
    sum_,
    tensor,
    text,
)

__all__ = [
    "Delta",
    "Domain",
    "Expr",
    "Index",
    "Op",
    "Orbital",
    "SparseOperator",
    "TensorFactor",
    "TensorSymbol",
    "Term",
    "a",
    "adag",
    "apply_ops_to_det",
    "compile",
    "create",
    "delta",
    "destroy",
    "domain",
    "expand_sums",
    "generate_basis",
    "index",
    "indices",
    "latex",
    "normal_order",
    "openfermion",
    "particle_delta",
    "prune_by_charge",
    "spin_index",
    "sum_",
    "tensor",
    "text",
]

try:
    __version__ = _pkg_version("nomad-qoal")
except PackageNotFoundError:  # pragma: no cover - running from an uninstalled source tree
    __version__ = "0.0.0+unknown"
