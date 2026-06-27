//! Optional PyO3 bridge placeholder.
//!
//! V1 ships a pure-Python frontend/fallback so the repository remains usable in
//! environments without a Rust compiler.  Building this crate with the `python`
//! feature exposes selected determinant kernels from `nomad-core`.

#[cfg(feature = "python")]
use pyo3::prelude::*;

#[cfg(feature = "python")]
#[pyfunction]
fn apply_ops_demo(det: u128) -> Option<(u128, i32)> {
    use nomad_core::expr::Op;
    use nomad_core::index::Mode;
    let ops = vec![Op::create(Mode::orbital(1)), Op::destroy(Mode::orbital(0))];
    nomad_core::apply_ops_to_det(det, &ops)
}

#[cfg(feature = "python")]
#[pymodule]
fn nomad_core_py(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(apply_ops_demo, m)?)?;
    Ok(())
}
