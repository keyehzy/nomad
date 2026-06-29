"""NOMAD: Normal-Ordered Matrix-free Algebra Device.

The public API is intentionally compact: build expressions with a Python DSL,
normal-order them, export them, or compile finite-basis expressions into a
matrix-free determinant-space operator.
"""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version

from .core import (
    Charge,
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
    basis_sector,
    charge,
    charge_delta,
    compile,
    create,
    delta,
    destroy,
    determinant_basis,
    determinant_charges,
    domain,
    expand_sums,
    generate_basis,
    index,
    indices,
    latex,
    normal_order,
    openfermion,
    operator_charge_delta,
    particle_delta,
    prune_by_charge,
    spin_index,
    sum_,
    tensor,
    term_charge_delta,
    text,
)

__all__ = [
    "Charge",
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
    "basis_sector",
    "charge",
    "charge_delta",
    "compile",
    "create",
    "delta",
    "determinant_basis",
    "determinant_charges",
    "destroy",
    "domain",
    "expand_sums",
    "generate_basis",
    "index",
    "indices",
    "latex",
    "normal_order",
    "openfermion",
    "operator_charge_delta",
    "particle_delta",
    "prune_by_charge",
    "spin_index",
    "sum_",
    "tensor",
    "term_charge_delta",
    "text",
]

try:
    __version__ = _pkg_version("nomad-qoal")
except PackageNotFoundError:  # pragma: no cover - running from an uninstalled source tree
    __version__ = "0.0.0+unknown"
