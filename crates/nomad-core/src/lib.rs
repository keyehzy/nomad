//! NOMAD V1 Rust core.
//!
//! The Python package includes an executable reference implementation.  This
//! crate is the deterministic core intended for PyO3 binding: exact symbolic
//! terms, CAR normal ordering, delta/index unification hooks, and determinant
//! bitstring kernels.

pub mod determinant;
pub mod expr;
pub mod index;
pub mod normal_order;

pub use determinant::{apply_ops_to_det, generate_basis};
pub use expr::{Delta, Expr, Op, OpKind, Term};
pub use index::Mode;
pub use normal_order::normal_order;
