use crate::expr::{Delta, Expr, OpKind, Term};

/// Fermionic CAR normal ordering using the local rewrite
/// `a_i a†_j = δ_ij - a†_j a_i`.
pub fn normal_order(expr: &Expr) -> Expr {
    let mut out = Expr::zero();
    for term in &expr.terms {
        out = out.plus(normal_order_term(term.clone()));
    }
    out.canonicalize();
    out
}

fn normal_order_term(term: Term) -> Expr {
    for i in 0..term.ops.len().saturating_sub(1) {
        if term.ops[i].kind == OpKind::Destroy && term.ops[i + 1].kind == OpKind::Create {
            let mut delta = term.clone();
            let left = delta.ops[i].mode.clone();
            let right = delta.ops[i + 1].mode.clone();
            delta.ops.remove(i + 1);
            delta.ops.remove(i);
            delta.deltas.push(Delta::new(left, right));

            let mut swapped = term.clone();
            swapped.coeff = -swapped.coeff;
            swapped.ops.swap(i, i + 1);

            return normal_order_term(delta).plus(normal_order_term(swapped));
        }
    }
    Expr::term(term)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::expr::{Expr, Op, Term};
    use crate::index::Mode;

    #[test]
    fn car_identity() {
        let i = Mode::symbol("i");
        let j = Mode::symbol("j");
        let e1 = Expr::term(Term::new(1, vec![Op::destroy(i.clone()), Op::create(j.clone())]));
        let e2 = Expr::term(Term::new(1, vec![Op::create(j.clone()), Op::destroy(i.clone())]));
        let out = normal_order(&e1.plus(e2));
        assert_eq!(out.terms.len(), 1);
        assert_eq!(out.terms[0].ops.len(), 0);
        assert_eq!(out.terms[0].deltas, vec![Delta::new(i, j)]);
    }
}
