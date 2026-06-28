"""NOMAD Python frontend and reference runtime.

This module is intentionally small, deterministic, and dependency-light.  The
data model is a weighted sum of NCIR/Wick term records, where normal ordering
may rewrite one term into many terms.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from fractions import Fraction
from itertools import count, product
from typing import Any

Number = int | float | Fraction
IndexKey = tuple[Any, ...]


# Monotonic source of hidden hygienic identities for freshly bound indices.
#
# Invariant: two uid spaces coexist -- this global counter (fresh bindings) and
# the small per-term canonical uids that `_rename_bound_dummies` assigns for
# display (0, 1, ... = len(mapping)), which do NOT consume this counter.  They
# never collide because the counter only advances and every bound index consumes
# at least one tick, so the next fresh uid always exceeds any per-term canonical
# uid.  The `avoid=` loop in `_fresh_bound_index_like` is therefore a defensive
# backstop (effectively a single iteration).
_BOUND_INDEX_UIDS = count()


# Display names of the form ``_<digits>`` are reserved for the canonical
# bound-dummy namespace produced by ``_rename_bound_dummies``; free indices may
# not use them, so a free index never visually collides with a rendered dummy.
_RESERVED_INDEX_NAME = re.compile(r"_\d+\Z")


@dataclass(frozen=True)
class Index:
    """A symbolic orbital/spin-orbital index.

    NOMAD treats indices as labels over a finite spin-orbital basis.  Optional
    metadata fields are present so the frontend can carry conservation-law
    annotations without committing to the future full charge-vector system.
    When an index is bound by ``sum_``, ``_uid`` carries its hidden hygienic
    identity; ``name`` remains only the human-readable display label.

    Display names of the form ``_<digits>`` are reserved for canonical bound
    dummies and are rejected for free (user-constructed) indices.
    """

    name: str
    spin_z2: int | None = None
    momentum: int | None = None
    _uid: int | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if not self.name or not isinstance(self.name, str):
            raise ValueError("Index name must be a non-empty string")
        if self._uid is None and _RESERVED_INDEX_NAME.match(self.name):
            raise ValueError(
                f"Index name {self.name!r} is reserved for canonical bound dummies; "
                "free indices may not use the '_<digits>' namespace"
            )
        if self._uid is not None and (
            isinstance(self._uid, bool) or not isinstance(self._uid, int) or self._uid < 0
        ):
            raise ValueError("Index hidden identity must be a non-negative integer")

    def __repr__(self) -> str:  # pragma: no cover - same as str for REPLs
        return self.name


def _coerce_summed_index(value: Index | str) -> Index:
    if isinstance(value, Index):
        return value
    if isinstance(value, str):
        return Index(value)
    raise TypeError("Summed indices must be Index objects or legacy string names")


def _optional_key(value: int | None) -> tuple[int, int]:
    return (0, 0) if value is None else (1, int(value))


def _index_metadata_key(index: Index) -> tuple[tuple[int, int], tuple[int, int]]:
    return (_optional_key(index.spin_z2), _optional_key(index.momentum))


def _index_identity(index: Index) -> IndexKey:
    if index._uid is not None:
        return ("bound", index._uid)
    return ("free", index.name, _index_metadata_key(index))


def _fresh_bound_index_like(
    index: Index, *, name: str | None = None, avoid: Iterable[IndexKey] = ()
) -> Index:
    avoid_keys = set(avoid)
    while True:
        candidate = Index(
            name or index.name, index.spin_z2, index.momentum, _uid=next(_BOUND_INDEX_UIDS)
        )
        if _index_identity(candidate) not in avoid_keys:
            return candidate


@dataclass(frozen=True)
class Orbital:
    """A concrete finite-basis spin-orbital label."""

    value: int

    def __post_init__(self) -> None:
        if self.value < 0:
            raise ValueError("Orbital labels must be non-negative")

    def __repr__(self) -> str:  # pragma: no cover - same as str for REPLs
        return str(self.value)


Mode = Index | Orbital


def _mode(m: Mode | int | str) -> Mode:
    if isinstance(m, (Index, Orbital)):
        return m
    if isinstance(m, int):
        return Orbital(m)
    if isinstance(m, str):
        return Index(m)
    raise TypeError(f"Expected Index, Orbital, int, or str mode, got {type(m)!r}")


def _mode_key(m: Mode) -> tuple[Any, ...]:
    if isinstance(m, Orbital):
        return (0, m.value)
    ident = _index_identity(m)
    if ident[0] == "bound":
        return (1, 0, ident[1], _index_metadata_key(m))
    return (1, 1, m.name, _index_metadata_key(m))


def _mode_text(m: Mode) -> str:
    return str(m.value) if isinstance(m, Orbital) else m.name


def index(name: str) -> Index:
    """Create a single symbolic index.

    ``p = index("p")`` is the common form.  Use :func:`indices` for several.
    """

    (out,) = indices(name)
    return out


def indices(names: str) -> tuple[Index, ...]:
    """Create a tuple of symbolic indices.

    ``p, q = indices("p q")`` is the common form.  Use :func:`index` for a
    single index.
    """

    parts = names.replace(",", " ").split()
    if not parts:
        raise ValueError("indices() needs at least one name")
    return tuple(Index(p) for p in parts)


def spin_index(name: str, spin_z2: int | None = None) -> Index:
    """Create an index carrying optional ``2*S_z`` metadata."""

    return Index(name, spin_z2=spin_z2)


@dataclass(frozen=True)
class TensorSymbol:
    name: str
    rank: int | None = None
    declaration: tuple[str, ...] = ()
    hermitian: bool = False
    antisymmetric_pairs: tuple[tuple[int, int], ...] = ()

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Tensor name cannot be empty")
        if self.rank is not None and self.rank < 0:
            raise ValueError("Tensor rank must be non-negative")
        for a, b in self.antisymmetric_pairs:
            if a == b:
                raise ValueError("Antisymmetric tensor pair cannot repeat a port")
            if self.rank is not None and not (0 <= a < self.rank and 0 <= b < self.rank):
                raise ValueError("Antisymmetric pair references a port outside tensor rank")

    def __getitem__(self, key: Mode | int | str | tuple[Mode | int | str, ...]) -> Expr:
        if not isinstance(key, tuple):
            key = (key,)
        ports = tuple(_mode(k) for k in key)
        if self.rank is not None and len(ports) != self.rank:
            raise ValueError(f"Tensor {self.name} has rank {self.rank}, got {len(ports)} ports")
        return Expr.from_term(Term(tensors=(TensorFactor(self, ports),)))


@dataclass(frozen=True)
class TensorFactor:
    symbol: TensorSymbol
    ports: tuple[Mode, ...]

    def key(self) -> tuple[Any, ...]:
        return (
            self.symbol.name,
            self.symbol.rank,
            self.symbol.hermitian,
            self.symbol.antisymmetric_pairs,
            tuple(_mode_key(p) for p in self.ports),
        )


@dataclass(frozen=True)
class Op:
    kind: str  # "create" | "destroy"
    mode: Mode
    statistics: str = "fermion"

    def __post_init__(self) -> None:
        if self.kind not in {"create", "destroy"}:
            raise ValueError("Op.kind must be 'create' or 'destroy'")
        if self.statistics != "fermion":
            raise NotImplementedError("NOMAD supports fermions only")

    def key(self) -> tuple[Any, ...]:
        return (self.kind, _mode_key(self.mode), self.statistics)


@dataclass(frozen=True)
class Delta:
    left: Mode
    right: Mode

    def key(self) -> tuple[Any, Any]:
        a, b = _mode_key(self.left), _mode_key(self.right)
        return (a, b) if a <= b else (b, a)


@dataclass(frozen=True)
class Term:
    coeff: Fraction = Fraction(1)
    tensors: tuple[TensorFactor, ...] = ()
    ops: tuple[Op, ...] = ()
    deltas: tuple[Delta, ...] = ()
    summed: tuple[Index, ...] = ()
    metadata: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "coeff", _to_fraction(self.coeff))
        object.__setattr__(
            self, "summed", tuple(dict.fromkeys(_coerce_summed_index(s) for s in self.summed))
        )

    def structural_key(self) -> tuple[Any, ...]:
        return (
            tuple(t.key() for t in self.tensors),
            tuple(d.key() for d in self.deltas),
            tuple(o.key() for o in self.ops),
            tuple(_mode_key(s) for s in self.summed),
            self.metadata,
        )

    def with_coeff(self, coeff: Number) -> Term:
        return Term(
            _to_fraction(coeff), self.tensors, self.ops, self.deltas, self.summed, self.metadata
        )

    def mul(self, other: Term) -> Term:
        left = _ensure_bound_index_identities(self)
        right = _ensure_bound_index_identities(other)
        right = _freshen_bound_indices(right, avoid=_bound_index_keys(left))
        return Term(
            left.coeff * right.coeff,
            left.tensors + right.tensors,
            left.ops + right.ops,
            left.deltas + right.deltas,
            tuple(dict.fromkeys(left.summed + right.summed)),
            tuple(sorted(left.metadata + right.metadata)),
        )


@dataclass(frozen=True)
class Expr:
    """A weighted sum of NCIR/Wick graph terms."""

    terms: tuple[Term, ...] = ()

    @staticmethod
    def zero() -> Expr:
        return Expr(())

    @staticmethod
    def one() -> Expr:
        return Expr((Term(),))

    @staticmethod
    def from_term(term: Term) -> Expr:
        return Expr((term,)).simplify()

    def simplify(self) -> Expr:
        buckets: dict[tuple[Any, ...], Term] = {}
        for term in self.terms:
            canon = _canonicalize_term(term)
            if canon is None or canon.coeff == 0:
                continue
            key = canon.structural_key()
            prev = buckets.get(key)
            if prev is None:
                buckets[key] = canon
            else:
                merged = prev.with_coeff(prev.coeff + canon.coeff)
                if merged.coeff == 0:
                    del buckets[key]
                else:
                    buckets[key] = merged
        ordered = sorted(buckets.values(), key=lambda t: t.structural_key())
        return Expr(tuple(ordered))

    def __add__(self, other: Any) -> Expr:
        other = as_expr(other)
        return Expr(self.terms + other.terms).simplify()

    def __radd__(self, other: Any) -> Expr:
        return as_expr(other).__add__(self)

    def __neg__(self) -> Expr:
        return Expr(tuple(t.with_coeff(-t.coeff) for t in self.terms)).simplify()

    def __sub__(self, other: Any) -> Expr:
        return self + (-as_expr(other))

    def __rsub__(self, other: Any) -> Expr:
        return as_expr(other) + (-self)

    def __mul__(self, other: Any) -> Expr:
        if isinstance(other, (int, Fraction, float)):
            return Expr(
                tuple(t.with_coeff(t.coeff * _to_fraction(other)) for t in self.terms)
            ).simplify()
        other = as_expr(other)
        return Expr(tuple(a.mul(b) for a in self.terms for b in other.terms)).simplify()

    def __rmul__(self, other: Any) -> Expr:
        if isinstance(other, (int, Fraction, float)):
            return self * other
        return as_expr(other).__mul__(self)

    def __bool__(self) -> bool:
        return bool(self.terms)

    def __repr__(self) -> str:  # pragma: no cover - exercised indirectly
        return text(self)


def _to_fraction(x: Number) -> Fraction:
    if isinstance(x, Fraction):
        return x
    if isinstance(x, bool):
        raise TypeError("Boolean coefficients are not supported")
    if isinstance(x, int):
        return Fraction(x)
    if isinstance(x, float):
        return Fraction.from_float(x).limit_denominator(10**12)
    raise TypeError(f"Unsupported scalar coefficient type: {type(x)!r}")


def as_expr(value: Any) -> Expr:
    if isinstance(value, Expr):
        return value
    if isinstance(value, Term):
        return Expr.from_term(value)
    if isinstance(value, (int, float, Fraction)):
        coeff = _to_fraction(value)
        return Expr.zero() if coeff == 0 else Expr.from_term(Term(coeff=coeff))
    raise TypeError(f"Cannot convert {type(value)!r} to NOMAD Expr")


def _substitute_mode(m: Mode, subst: Mapping[Any, Mode]) -> Mode:
    if isinstance(m, Index):
        ident = _index_identity(m)
        if ident in subst:
            return subst[ident]
        if m in subst:
            return subst[m]
    return m


def _replace_term_modes(
    term: Term, subst: Mapping[Any, Mode], *, replace_summed: bool = True
) -> Term:
    tensors = tuple(
        TensorFactor(t.symbol, tuple(_substitute_mode(p, subst) for p in t.ports))
        for t in term.tensors
    )
    ops = tuple(Op(o.kind, _substitute_mode(o.mode, subst), o.statistics) for o in term.ops)
    deltas = tuple(
        Delta(_substitute_mode(d.left, subst), _substitute_mode(d.right, subst))
        for d in term.deltas
    )
    if replace_summed:
        summed = tuple(
            s for s in (_substitute_mode(i, subst) for i in term.summed) if isinstance(s, Index)
        )
    else:
        summed = term.summed
    return Term(term.coeff, tensors, ops, deltas, summed, term.metadata)


def _bound_index_keys(term: Term) -> set[IndexKey]:
    return {_index_identity(i) for i in term.summed}


def _ensure_bound_index_identities(term: Term) -> Term:
    """Upgrade legacy name-bound summation entries to hygienic identities."""

    subst: dict[IndexKey, Mode] = {}
    occupied = _bound_index_keys(term)
    for index in term.summed:
        if index._uid is not None:
            continue
        key = _index_identity(index)
        if key not in subst:
            # `occupied` already tracks every fresh identity assigned below, so
            # it alone is a sufficient avoid-set (no need to re-scan `subst`).
            fresh = _fresh_bound_index_like(index, avoid=occupied)
            subst[key] = fresh
            occupied.add(_index_identity(fresh))
    if not subst:
        return term
    return _replace_term_modes(term, subst)


def _freshen_bound_indices(term: Term, *, avoid: Iterable[IndexKey] = ()) -> Term:
    """Rename bound identities in ``term`` when they collide with ``avoid``."""

    avoid_keys = set(avoid)
    occupied = _bound_index_keys(term) | avoid_keys
    subst: dict[IndexKey, Mode] = {}
    for index in term.summed:
        key = _index_identity(index)
        if key not in avoid_keys or key in subst:
            continue
        fresh = _fresh_bound_index_like(index, avoid=occupied)
        subst[key] = fresh
        occupied.add(_index_identity(fresh))
    if not subst:
        return term
    return _replace_term_modes(term, subst)


def tensor(
    name: str,
    indices_or_rank: int | Sequence[Index] | None = None,
    *,
    rank: int | None = None,
    hermitian: bool = False,
    antisymmetric_pairs: Sequence[tuple[int | Index, int | Index]] | None = None,
) -> TensorSymbol:
    """Declare a symbolic tensor.

    Examples:
        ``h = tensor("h", [p, q], hermitian=True)``
        ``g = tensor("g", 4, antisymmetric_pairs=[(0, 1), (2, 3)])``
    """

    declaration: tuple[str, ...] = ()
    if isinstance(indices_or_rank, int):
        if rank is not None and rank != indices_or_rank:
            raise ValueError("rank provided twice with conflicting values")
        rank = indices_or_rank
    elif indices_or_rank is not None:
        declaration = tuple(i.name if isinstance(i, Index) else str(i) for i in indices_or_rank)
        if rank is not None and rank != len(declaration):
            raise ValueError("rank does not match declaration length")
        rank = len(declaration)

    pair_positions: list[tuple[int, int]] = []
    for pair in antisymmetric_pairs or ():
        a, b = pair
        if isinstance(a, Index):
            if a.name not in declaration:
                raise ValueError(f"Index {a.name!r} is not in tensor declaration")
            ia = declaration.index(a.name)
        else:
            ia = int(a)
        if isinstance(b, Index):
            if b.name not in declaration:
                raise ValueError(f"Index {b.name!r} is not in tensor declaration")
            ib = declaration.index(b.name)
        else:
            ib = int(b)
        lo, hi = sorted((ia, ib))
        pair_positions.append((lo, hi))

    return TensorSymbol(name, rank, declaration, hermitian, tuple(sorted(set(pair_positions))))


def create(mode: Mode | int | str) -> Expr:
    return Expr.from_term(Term(ops=(Op("create", _mode(mode)),)))


def destroy(mode: Mode | int | str) -> Expr:
    return Expr.from_term(Term(ops=(Op("destroy", _mode(mode)),)))


adag = create
a = destroy


def delta(left: Mode | int | str, right: Mode | int | str) -> Expr:
    return Expr.from_term(Term(deltas=(Delta(_mode(left), _mode(right)),)))


def sum_(*args: Any) -> Expr:
    """Attach symbolic summation domains to an expression.

    NOMAD does not expand sums until a finite backend asks for it.  Deltas can
    reduce summation domains during canonicalization.
    """

    if len(args) < 2:
        raise ValueError("sum_ expects one or more indices followed by an expression")
    *idxs, expr = args
    expr = as_expr(expr).simplify()
    bindings: dict[IndexKey, Mode] = {}
    bound_indices: list[Index] = []
    for idx in idxs:
        if not isinstance(idx, Index):
            raise TypeError("sum_ indices must be Index objects")
        key = _index_identity(idx)
        if key in bindings:
            continue
        bound = _fresh_bound_index_like(idx)
        bindings[key] = bound
        bound_indices.append(bound)

    terms: list[Term] = []
    for term in expr.terms:
        rebound = _replace_term_modes(term, bindings, replace_summed=False)
        terms.append(
            Term(
                rebound.coeff,
                rebound.tensors,
                rebound.ops,
                rebound.deltas,
                tuple(dict.fromkeys(rebound.summed + tuple(bound_indices))),
                rebound.metadata,
            )
        )
    return Expr(tuple(terms)).simplify()


# ---------------------------------------------------------------------------
# Canonicalization
# ---------------------------------------------------------------------------


class _UnionFind:
    def __init__(self) -> None:
        self.parent: dict[Any, Any] = {}

    def add(self, x: Any) -> None:
        self.parent.setdefault(x, x)

    def find(self, x: Any) -> Any:
        self.add(x)
        p = self.parent[x]
        if p != x:
            self.parent[x] = self.find(p)
        return self.parent[x]

    def union(self, a: Any, b: Any) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            if repr(ra) <= repr(rb):
                self.parent[rb] = ra
            else:
                self.parent[ra] = rb

    def groups(self) -> dict[Any, list[Any]]:
        out: dict[Any, list[Any]] = {}
        for x in list(self.parent):
            out.setdefault(self.find(x), []).append(x)
        return out


def _node_for_mode(m: Mode) -> tuple[Any, ...]:
    if isinstance(m, Orbital):
        return ("const", m.value)
    return ("idx", _index_identity(m), m)


def _mode_from_node(node: tuple[Any, ...]) -> Mode:
    return Orbital(int(node[1])) if node[0] == "const" else node[2]


def _canonicalize_delta_constraints(term: Term) -> Term | None:
    if not term.deltas:
        return term

    uf = _UnionFind()
    for d in term.deltas:
        a, b = _node_for_mode(d.left), _node_for_mode(d.right)
        uf.add(a)
        uf.add(b)
        uf.union(a, b)

    summed_by_key = {_index_identity(i): i for i in term.summed}
    summed = set(summed_by_key)
    subst: dict[IndexKey, Mode] = {}
    retained: list[Delta] = []
    keep_summed = set(summed)
    rep_mode: Mode

    for nodes in uf.groups().values():
        idx_by_key: dict[IndexKey, Index] = {}
        consts = sorted({int(node[1]) for node in nodes if node[0] == "const"})
        for node in nodes:
            if node[0] == "idx":
                idx_by_key.setdefault(node[1], node[2])
        if len(consts) > 1:
            return None

        idx_items = tuple(idx_by_key.items())
        free = sorted(
            ((k, idx) for k, idx in idx_items if k not in summed),
            key=lambda item: _mode_key(item[1]),
        )
        bound = sorted(
            ((k, idx) for k, idx in idx_items if k in summed),
            key=lambda item: _mode_key(item[1]),
        )

        if free:
            _rep_key, rep_mode = free[0]
            for key, _idx in idx_items:
                subst[key] = rep_mode
                if key in summed:
                    keep_summed.discard(key)
            for _key, other in free[1:]:
                retained.append(Delta(rep_mode, other))
            if consts:
                retained.append(Delta(rep_mode, Orbital(consts[0])))
        else:
            # All symbols in the equality class are bound dummy indices or
            # constants, so the delta can be consumed by reducing the domain.
            if consts:
                rep_mode = Orbital(consts[0])
                for key, _idx in bound:
                    subst[key] = rep_mode
                    keep_summed.discard(key)
            elif bound:
                rep_key, rep_mode = bound[0]
                keep_summed.add(rep_key)
                for key, _idx in bound:
                    subst[key] = rep_mode
                    if key != rep_key:
                        keep_summed.discard(key)
            # Pure constant equalities have already been checked and vanish.

    tensors = tuple(
        TensorFactor(t.symbol, tuple(_substitute_mode(p, subst) for p in t.ports))
        for t in term.tensors
    )
    ops = tuple(Op(o.kind, _substitute_mode(o.mode, subst), o.statistics) for o in term.ops)

    # Deduplicate retained free-index constraints canonically.
    unique: dict[tuple[Any, Any], Delta] = {}
    for d in retained:
        if _mode_key(d.right) < _mode_key(d.left):
            d = Delta(d.right, d.left)
        if _mode_key(d.left) != _mode_key(d.right):
            unique[d.key()] = d
    deltas = tuple(unique[k] for k in sorted(unique))
    summed_ordered = tuple(s for s in term.summed if _index_identity(s) in keep_summed)
    return Term(term.coeff, tensors, ops, deltas, summed_ordered, term.metadata)


def _inversion_parity(keys: Sequence[tuple[Any, ...]]) -> int:
    inv = 0
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            if keys[i] > keys[j]:
                inv += 1
    return inv % 2


def _canonicalize_op_runs(term: Term) -> Term | None:
    ops = list(term.ops)
    coeff = term.coeff
    out: list[Op] = []
    i = 0
    while i < len(ops):
        j = i + 1
        while j < len(ops) and ops[j].kind == ops[i].kind:
            j += 1
        run = ops[i:j]
        keys = [_mode_key(o.mode) for o in run]
        if len(keys) != len(set(keys)):
            return None  # a_p a_p = 0 and a†_p a†_p = 0 for fermions.
        if _inversion_parity(keys):
            coeff = -coeff
        out.extend(sorted(run, key=lambda o: _mode_key(o.mode)))
        i = j
    return Term(coeff, term.tensors, tuple(out), term.deltas, term.summed, term.metadata)


def _canonicalize_tensor_factor(factor: TensorFactor) -> tuple[Fraction, TensorFactor] | None:
    coeff = Fraction(1)
    ports = list(factor.ports)
    for a, b in factor.symbol.antisymmetric_pairs:
        if _mode_key(ports[a]) == _mode_key(ports[b]):
            return None
        if _mode_key(ports[a]) > _mode_key(ports[b]):
            ports[a], ports[b] = ports[b], ports[a]
            coeff = -coeff
    return coeff, TensorFactor(factor.symbol, tuple(ports))


def _canonicalize_tensors(term: Term) -> Term | None:
    coeff = term.coeff
    factors = []
    for f in term.tensors:
        out = _canonicalize_tensor_factor(f)
        if out is None:
            return None
        c, ff = out
        coeff *= c
        factors.append(ff)
    factors.sort(key=lambda f: f.key())
    return Term(coeff, tuple(factors), term.ops, term.deltas, term.summed, term.metadata)


def _rename_bound_dummies(term: Term) -> Term:
    bound = {_index_identity(i): i for i in term.summed}
    if not bound:
        return term
    mapping: dict[IndexKey, Index] = {}

    def visit(m: Mode) -> None:
        if isinstance(m, Index):
            key = _index_identity(m)
            if key in bound and key not in mapping:
                source = bound[key]
                # Small per-term canonical uids (0, 1, ...) live in a space
                # separate from the global `_BOUND_INDEX_UIDS` counter; see the
                # invariant noted at that counter's definition.
                mapping[key] = Index(
                    f"_{len(mapping)}", source.spin_z2, source.momentum, _uid=len(mapping)
                )

    for t in term.tensors:
        for p in t.ports:
            visit(p)
    for o in term.ops:
        visit(o.mode)
    for d in term.deltas:
        visit(d.left)
        visit(d.right)

    rebound = _replace_term_modes(term, mapping, replace_summed=False)
    return Term(
        rebound.coeff,
        rebound.tensors,
        rebound.ops,
        rebound.deltas,
        tuple(mapping.values()),
        rebound.metadata,
    )


def _canonicalize_term(term: Term) -> Term | None:
    if term.coeff == 0:
        return None
    prepared = _ensure_bound_index_identities(term)
    t = _canonicalize_delta_constraints(prepared)
    if t is None:
        return None
    t = _canonicalize_op_runs(t)
    if t is None:
        return None
    t = _canonicalize_tensors(t)
    if t is None:
        return None
    t = _rename_bound_dummies(t)
    # Deltas are commutative scalar constraints.
    deltas = tuple(sorted(t.deltas, key=lambda d: d.key()))
    return Term(t.coeff, t.tensors, t.ops, deltas, t.summed, t.metadata)


# ---------------------------------------------------------------------------
# Normal ordering and symmetry pruning
# ---------------------------------------------------------------------------


def normal_order(expr: Any) -> Expr:
    """Normal-order a fermionic expression using the CAR local rewrite.

    The implemented rewrite is ``a_i a†_j -> δ_ij - a†_j a_i``.  Same-kind
    fermionic anti-commutation is handled by canonical sorting inside each
    normal block.
    """

    expr = as_expr(expr)
    memo: dict[tuple[Any, ...], Expr] = {}

    def rec(term: Term) -> Expr:
        key = (term.coeff, term.structural_key())
        cached = memo.get(key)
        if cached is not None:
            return cached
        ops = term.ops
        for i in range(len(ops) - 1):
            if ops[i].kind == "destroy" and ops[i + 1].kind == "create":
                # a_i a†_j = δ_ij - a†_j a_i
                delta_term = Term(
                    term.coeff,
                    term.tensors,
                    ops[:i] + ops[i + 2 :],
                    term.deltas + (Delta(ops[i].mode, ops[i + 1].mode),),
                    term.summed,
                    term.metadata,
                )
                swapped_term = Term(
                    -term.coeff,
                    term.tensors,
                    ops[:i] + (ops[i + 1], ops[i]) + ops[i + 2 :],
                    term.deltas,
                    term.summed,
                    term.metadata,
                )
                ans = rec(delta_term) + rec(swapped_term)
                memo[key] = ans
                return ans
        ans = Expr.from_term(term)
        memo[key] = ans
        return ans

    out = Expr.zero()
    for term in expr.terms:
        out += rec(term)
    return out.simplify()


def particle_delta(term: Term) -> int:
    return sum(1 if o.kind == "create" else -1 for o in term.ops)


def prune_by_charge(expr: Any, *, delta_n: int = 0) -> Expr:
    """Compile-time U(1) particle-number pruning."""

    expr = as_expr(expr).simplify()
    return Expr(tuple(t for t in expr.terms if particle_delta(t) == delta_n)).simplify()


# ---------------------------------------------------------------------------
# Finite expansion, determinant backend, and exports
# ---------------------------------------------------------------------------


def _eval_tensor_value(values: Any, ports: tuple[Mode, ...]) -> Fraction:
    idx = tuple(p.value for p in ports if isinstance(p, Orbital))
    if len(idx) != len(ports):
        raise ValueError("Cannot evaluate tensor with symbolic ports")
    if callable(values):
        val = values(*idx)
    elif isinstance(values, Mapping):
        val = values[idx]
    else:
        val = values
        for i in idx:
            val = val[i]
    return _to_fraction(float(val) if hasattr(val, "item") else val)


def expand_sums(
    expr: Any, *, n_orbitals: int, tensor_values: Mapping[str, Any] | None = None
) -> Expr:
    """Expand symbolic sums over a finite spin-orbital basis.

    Tensor factors with concrete ports are evaluated when their symbol appears
    in ``tensor_values``.  Remaining symbolic tensors are preserved; sparse
    compilation requires all tensors to be evaluated.
    """

    tensor_values = tensor_values or {}
    expr = as_expr(expr).simplify()
    out = Expr.zero()
    for term in expr.terms:
        summed = list(term.summed)
        domains = [range(n_orbitals) for _ in summed]
        for values in product(*domains):
            subst = {
                _index_identity(index): Orbital(v) for index, v in zip(summed, values, strict=True)
            }
            tensors = []
            coeff = term.coeff
            for tf in term.tensors:
                ports = tuple(_substitute_mode(p, subst) for p in tf.ports)
                if tf.symbol.name in tensor_values:
                    coeff *= _eval_tensor_value(tensor_values[tf.symbol.name], ports)
                else:
                    tensors.append(TensorFactor(tf.symbol, ports))
            ops = tuple(Op(o.kind, _substitute_mode(o.mode, subst), o.statistics) for o in term.ops)
            deltas = tuple(
                Delta(_substitute_mode(d.left, subst), _substitute_mode(d.right, subst))
                for d in term.deltas
            )
            out += Expr.from_term(Term(coeff, tuple(tensors), ops, deltas, (), term.metadata))
    return out.simplify()


def _require_finite_numeric(expr: Expr) -> None:
    for term in expr.terms:
        if term.tensors:
            raise ValueError("Sparse/export backends require evaluated tensor coefficients")
        if term.deltas:
            raise ValueError("Sparse/export backends require eliminated delta constraints")
        if term.summed:
            raise ValueError("Sparse/export backends require expanded sums")
        for op in term.ops:
            if not isinstance(op.mode, Orbital):
                raise ValueError("Sparse/export backends require concrete orbital labels")


def apply_ops_to_det(det: int, ops: Sequence[Op]) -> tuple[int, int] | None:
    """Apply an operator word to a determinant bitstring.

    Operators are stored left-to-right, so application to ``|D>`` proceeds from
    the rightmost operator to the leftmost one.
    """

    d = int(det)
    sign = 1
    for op in reversed(ops):
        if not isinstance(op.mode, Orbital):
            raise ValueError("determinant backend only accepts concrete orbitals")
        p = op.mode.value
        mask = 1 << p
        phase = -1 if ((d & (mask - 1)).bit_count() % 2) else 1
        if op.kind == "destroy":
            if not (d & mask):
                return None
            sign *= phase
            d &= ~mask
        else:
            if d & mask:
                return None
            sign *= phase
            d |= mask
    return d, sign


def generate_basis(
    n_orbitals: int, sector: Mapping[str, int] | None = None, spin_z2: Sequence[int] | None = None
) -> tuple[int, ...]:
    sector = sector or {}
    required_n = sector.get("N")
    required_sz2 = sector.get("Sz2", sector.get("Sz"))
    if spin_z2 is None:
        spin_z2 = tuple(1 if i % 2 == 0 else -1 for i in range(n_orbitals))
    out = []
    for det in range(1 << n_orbitals):
        if required_n is not None and det.bit_count() != required_n:
            continue
        if required_sz2 is not None:
            sz = sum(spin_z2[i] for i in range(n_orbitals) if det & (1 << i))
            if sz != required_sz2:
                continue
        out.append(det)
    return tuple(out)


@dataclass
class SparseOperator:
    expr: Expr
    n_orbitals: int
    sector: Mapping[str, int] = field(default_factory=dict)
    basis: tuple[int, ...] = field(init=False)
    index_of: dict[int, int] = field(init=False)

    def __post_init__(self) -> None:
        self.expr = self.expr.simplify()
        _require_finite_numeric(self.expr)
        self.basis = generate_basis(self.n_orbitals, self.sector)
        self.index_of = {d: i for i, d in enumerate(self.basis)}

    @property
    def shape(self) -> tuple[int, int]:
        n = len(self.basis)
        return (n, n)

    def matvec(self, x: Sequence[Number]) -> list[float]:
        if len(x) != len(self.basis):
            raise ValueError(
                f"Input vector length {len(x)} does not match basis size {len(self.basis)}"
            )
        y = [0.0 for _ in self.basis]
        for col, det in enumerate(self.basis):
            amp = float(x[col])
            if amp == 0.0:
                continue
            for term in self.expr.terms:
                applied = apply_ops_to_det(det, term.ops)
                if applied is None:
                    continue
                out_det, sign = applied
                row = self.index_of.get(out_det)
                if row is not None:
                    y[row] += float(term.coeff) * sign * amp
        return y

    def to_dense(self) -> list[list[float]]:
        n = len(self.basis)
        mat = [[0.0 for _ in range(n)] for _ in range(n)]
        for col in range(n):
            e = [0.0 for _ in range(n)]
            e[col] = 1.0
            y = self.matvec(e)
            for row, val in enumerate(y):
                mat[row][col] = val
        return mat


def compile(  # noqa: A001 - public API intentionally named compile
    expr: Any,
    *,
    target: str = "sparse",
    n_orbitals: int | None = None,
    sector: Mapping[str, int] | None = None,
    tensor_values: Mapping[str, Any] | None = None,
) -> Any:
    if target != "sparse":
        raise NotImplementedError("NOMAD executable backend is target='sparse'")
    if n_orbitals is None:
        raise ValueError("n_orbitals is required for sparse compilation")
    lowered = expand_sums(normal_order(expr), n_orbitals=n_orbitals, tensor_values=tensor_values)
    return SparseOperator(lowered, n_orbitals=n_orbitals, sector=sector or {})


def openfermion(expr: Any) -> str:
    """Emit Python source constructing an OpenFermion FermionOperator."""

    expr = as_expr(expr).simplify()
    _require_finite_numeric(expr)
    lines = ["from openfermion import FermionOperator", "op = FermionOperator.zero()"]
    for term in expr.terms:
        word = " ".join(
            f"{op.mode.value}^" if op.kind == "create" else f"{op.mode.value}" for op in term.ops
        )
        coeff = (
            str(term.coeff.numerator)
            if term.coeff.denominator == 1
            else f"({term.coeff.numerator}/{term.coeff.denominator})"
        )
        lines.append(f"op += FermionOperator({word!r}, {coeff})")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Human-readable rendering
# ---------------------------------------------------------------------------


def _format_coeff(c: Fraction) -> str:
    if c.denominator == 1:
        return str(c.numerator)
    return f"{c.numerator}/{c.denominator}"


def text(expr: Any) -> str:
    expr = as_expr(expr).simplify()
    if not expr.terms:
        return "0"
    parts = []
    for term in expr.terms:
        factors = []
        if term.summed:
            factors.append("Σ_" + ",".join(_mode_text(i) for i in term.summed))
        for d in term.deltas:
            factors.append(f"δ({_mode_text(d.left)},{_mode_text(d.right)})")
        for t in term.tensors:
            factors.append(f"{t.symbol.name}[{','.join(_mode_text(p) for p in t.ports)}]")
        for o in term.ops:
            factors.append(("a†" if o.kind == "create" else "a") + f"({_mode_text(o.mode)})")
        body = " ".join(factors) if factors else "1"
        if term.coeff == 1 and factors:
            parts.append(body)
        elif term.coeff == -1 and factors:
            parts.append("-" + body)
        else:
            parts.append(
                f"{_format_coeff(term.coeff)}*{body}" if factors else _format_coeff(term.coeff)
            )
    out = " + ".join(parts)
    return out.replace("+ -", "- ")


def latex(expr: Any) -> str:
    expr = as_expr(expr).simplify()
    if not expr.terms:
        return "0"
    parts = []
    for term in expr.terms:
        factors = []
        if term.summed:
            factors.append(r"\sum_{" + ",".join(_mode_text(i) for i in term.summed) + "}")
        for d in term.deltas:
            factors.append(r"\delta_{" + _mode_text(d.left) + "," + _mode_text(d.right) + "}")
        for t in term.tensors:
            factors.append(t.symbol.name + "_{" + ",".join(_mode_text(p) for p in t.ports) + "}")
        for o in term.ops:
            if o.kind == "create":
                factors.append(r"a^\dagger_{" + _mode_text(o.mode) + "}")
            else:
                factors.append(r"a_{" + _mode_text(o.mode) + "}")
        body = " ".join(factors) if factors else "1"
        c = term.coeff
        if c == 1 and factors:
            parts.append(body)
        elif c == -1 and factors:
            parts.append("-" + body)
        else:
            parts.append((_format_coeff(c) + " " + body).strip())
    return " + ".join(parts).replace("+ -", "- ")
