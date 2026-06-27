use crate::index::Mode;
use std::collections::BTreeMap;

/// Fermionic operator kind.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub enum OpKind {
    Create,
    Destroy,
}

/// A single fermionic creation or annihilation operator.
#[derive(Clone, Debug, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub struct Op {
    pub kind: OpKind,
    pub mode: Mode,
}

impl Op {
    pub fn create(mode: Mode) -> Self {
        Self { kind: OpKind::Create, mode }
    }

    pub fn destroy(mode: Mode) -> Self {
        Self { kind: OpKind::Destroy, mode }
    }
}

/// Delta/equality edge produced by a CAR contraction.
#[derive(Clone, Debug, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub struct Delta {
    pub left: Mode,
    pub right: Mode,
}

impl Delta {
    pub fn new(left: Mode, right: Mode) -> Self {
        if right < left { Self { left: right, right: left } } else { Self { left, right } }
    }
}

/// A V1 graph term.  Coefficients are exact signed integers in the Rust core;
/// the Python frontend supports Fractions for user convenience.
#[derive(Clone, Debug, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub struct Term {
    pub coeff: i64,
    pub ops: Vec<Op>,
    pub deltas: Vec<Delta>,
}

impl Term {
    pub fn new(coeff: i64, ops: Vec<Op>) -> Self {
        Self { coeff, ops, deltas: Vec::new() }
    }

    pub fn structural_key(&self) -> (Vec<Op>, Vec<Delta>) {
        (self.ops.clone(), self.deltas.clone())
    }
}

/// A weighted sum of graph terms.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct Expr {
    pub terms: Vec<Term>,
}

impl Expr {
    pub fn zero() -> Self {
        Self { terms: Vec::new() }
    }

    pub fn term(term: Term) -> Self {
        let mut expr = Self { terms: vec![term] };
        expr.canonicalize();
        expr
    }

    pub fn add_term(&mut self, term: Term) {
        self.terms.push(term);
        self.canonicalize();
    }

    pub fn canonicalize(&mut self) {
        let mut buckets: BTreeMap<(Vec<Op>, Vec<Delta>), i64> = BTreeMap::new();
        for mut term in self.terms.drain(..) {
            if term.coeff == 0 { continue; }
            if !canonicalize_op_runs(&mut term) { continue; }
            term.deltas.sort();
            let key = term.structural_key();
            *buckets.entry(key).or_insert(0) += term.coeff;
        }
        self.terms = buckets
            .into_iter()
            .filter_map(|((ops, deltas), coeff)| (coeff != 0).then_some(Term { coeff, ops, deltas }))
            .collect();
    }

    pub fn plus(mut self, other: Expr) -> Expr {
        self.terms.extend(other.terms);
        self.canonicalize();
        self
    }
}

fn canonicalize_op_runs(term: &mut Term) -> bool {
    let mut coeff = term.coeff;
    let mut out = Vec::with_capacity(term.ops.len());
    let mut i = 0;
    while i < term.ops.len() {
        let mut j = i + 1;
        while j < term.ops.len() && term.ops[j].kind == term.ops[i].kind { j += 1; }
        let mut run = term.ops[i..j].to_vec();
        let mut inversions = 0;
        for a in 0..run.len() {
            for b in (a + 1)..run.len() {
                if run[a].mode > run[b].mode { inversions += 1; }
                if run[a].mode == run[b].mode { return false; }
            }
        }
        if inversions % 2 == 1 { coeff = -coeff; }
        run.sort_by(|a, b| a.mode.cmp(&b.mode));
        out.extend(run);
        i = j;
    }
    term.coeff = coeff;
    term.ops = out;
    true
}
