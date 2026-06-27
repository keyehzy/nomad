use crate::expr::{Op, OpKind};
use crate::index::Mode;

/// Apply a fermionic operator word to a determinant bitstring.
/// Operators are stored left-to-right and applied right-to-left.
pub fn apply_ops_to_det(det: u128, ops: &[Op]) -> Option<(u128, i32)> {
    let mut d = det;
    let mut sign = 1i32;
    for op in ops.iter().rev() {
        let p = match op.mode {
            Mode::Orbital(p) => p,
            Mode::Symbol(_) => return None,
        };
        if p >= 128 { return None; }
        let mask = 1u128 << p;
        let below = mask - 1;
        if (d & below).count_ones() % 2 == 1 { sign = -sign; }
        match op.kind {
            OpKind::Destroy => {
                if d & mask == 0 { return None; }
                d &= !mask;
            }
            OpKind::Create => {
                if d & mask != 0 { return None; }
                d |= mask;
            }
        }
    }
    Some((d, sign))
}

/// Generate finite-basis determinants in a particle-number sector.
pub fn generate_basis(n_orbitals: u32, particles: Option<u32>) -> Vec<u128> {
    assert!(n_orbitals <= 128);
    let limit = if n_orbitals == 128 { u128::MAX } else { 1u128 << n_orbitals };
    let mut basis = Vec::new();
    let mut d = 0u128;
    while d < limit {
        if particles.map_or(true, |n| d.count_ones() == n) {
            basis.push(d);
        }
        d += 1;
    }
    basis
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::expr::Op;
    use crate::index::Mode;

    #[test]
    fn creation_annihilation_sign() {
        // a†_2 a_0 |0b0011> = - |0b0110>
        let ops = vec![Op::create(Mode::orbital(2)), Op::destroy(Mode::orbital(0))];
        assert_eq!(apply_ops_to_det(0b0011, &ops), Some((0b0110, -1)));
    }
}
