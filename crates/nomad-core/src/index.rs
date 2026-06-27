use std::cmp::Ordering;

/// A spin-orbital mode: either symbolic (for algebra/NCIR) or finite concrete
/// (for determinant-space kernels).
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub enum Mode {
    Symbol(String),
    Orbital(u32),
}

impl Mode {
    pub fn symbol<S: Into<String>>(name: S) -> Self {
        Self::Symbol(name.into())
    }

    pub fn orbital(value: u32) -> Self {
        Self::Orbital(value)
    }
}

impl Ord for Mode {
    fn cmp(&self, other: &Self) -> Ordering {
        use Mode::*;
        match (self, other) {
            (Orbital(a), Orbital(b)) => a.cmp(b),
            (Orbital(_), Symbol(_)) => Ordering::Less,
            (Symbol(_), Orbital(_)) => Ordering::Greater,
            (Symbol(a), Symbol(b)) => a.cmp(b),
        }
    }
}

impl PartialOrd for Mode {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
