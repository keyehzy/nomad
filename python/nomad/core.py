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
FiniteValues = range | tuple[int, ...]


def _validate_nonnegative_int(value: int, what: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{what} must be a non-negative integer")
    return value


def _coerce_domain_values(values: Iterable[int], *, context: str) -> tuple[int, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{context} values must be an iterable of integer orbital labels")
    out = tuple(_validate_nonnegative_int(v, f"{context} value") for v in values)
    if len(set(out)) != len(out):
        raise ValueError(f"{context} values must not contain duplicates")
    return out


@dataclass(frozen=True, eq=False)
class Domain:
    """A symbolic index domain with optional finite expansion data.

    ``Domain("spin_orbital")`` is the default full-basis domain.  Other
    domains can be made finite either by size/start (global orbital labels
    ``start .. start + size - 1``) or by an explicit list of global labels.
    During tensor evaluation, axes are indexed by the local position inside the
    domain, while operators use the global orbital label.

    Domain identity is *canonical*: two domains compare equal when they share a
    name and expand to the same ordered global labels, regardless of whether
    that mapping was given as ``size``/``start`` or as an explicit ``values``
    list (see :meth:`key`).  Differently-named or differently-ordered domains
    stay distinct; a δ between them is resolved at finite expansion instead.
    """

    name: str
    size: int | None = None
    start: int = 0
    values: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        if not self.name or not isinstance(self.name, str):
            raise ValueError("Domain name must be a non-empty string")
        if self.size is not None:
            object.__setattr__(
                self, "size", _validate_nonnegative_int(self.size, "Domain size")
            )
        object.__setattr__(self, "start", _validate_nonnegative_int(self.start, "Domain start"))
        if self.values is not None:
            if self.start != 0:
                raise ValueError("Domain start cannot be combined with explicit values")
            values = _coerce_domain_values(self.values, context=f"Domain {self.name!r}")
            if self.size is not None and self.size != len(values):
                raise ValueError("Domain size does not match the number of explicit values")
            object.__setattr__(self, "values", values)
            object.__setattr__(self, "size", len(values))
        elif self.size is None and self.start != 0:
            raise ValueError("Domain start requires a finite size")

    def finite_values(self) -> FiniteValues | None:
        if self.values is not None:
            return self.values
        if self.size is not None:
            return range(self.start, self.start + self.size)
        return None

    def key(self) -> tuple[Any, ...]:
        """Canonical, totally-ordered domain identity.

        Identity is the name plus the ordered global labels the domain expands
        to.  The ``size``/``start`` form and an explicit ``values`` list that
        produce the *same ordered* labels therefore compare equal, while a
        not-yet-finite (string/default) domain stays distinct from any finite
        one.  The tuple contains no ``None`` so terms carrying typed indices
        remain sortable during canonicalization.
        """
        finite = self.finite_values()
        if finite is None:
            return (self.name, False, ())
        return (self.name, True, tuple(finite))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Domain):
            return NotImplemented
        return self.key() == other.key()

    def __hash__(self) -> int:
        return hash(self.key())

    def __repr__(self) -> str:  # pragma: no cover - same as str for REPLs
        return self.name


SPIN_ORBITAL_DOMAIN = Domain("spin_orbital")


def _coerce_domain(value: Domain | str | None) -> Domain:
    if value is None:
        return SPIN_ORBITAL_DOMAIN
    if isinstance(value, Domain):
        return value
    if isinstance(value, str):
        return Domain(value)
    raise TypeError("Index domain must be a Domain object, a domain name, or None")


def _is_default_spin_orbital_domain(value: Domain) -> bool:
    return value == SPIN_ORBITAL_DOMAIN


def _domain_metadata_key(value: Domain) -> tuple[Any, ...]:
    return value.key()


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
    domain: Domain | str | None = None
    _uid: int | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if not self.name or not isinstance(self.name, str):
            raise ValueError("Index name must be a non-empty string")
        object.__setattr__(self, "domain", _coerce_domain(self.domain))
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


def _index_domain(index: Index) -> Domain:
    return _coerce_domain(index.domain)


def _index_metadata_key(index: Index) -> tuple[tuple[int, int], tuple[int, int], tuple[Any, ...]]:
    return (
        _optional_key(index.spin_z2),
        _optional_key(index.momentum),
        _domain_metadata_key(_index_domain(index)),
    )


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
            name or index.name,
            index.spin_z2,
            index.momentum,
            _index_domain(index),
            _uid=next(_BOUND_INDEX_UIDS),
        )
        if _index_identity(candidate) not in avoid_keys:
            return candidate


@dataclass(frozen=True)
class Orbital:
    """A concrete finite-basis spin-orbital label.

    ``value`` is the global orbital label used by operators.  ``local_value`` is
    optional provenance from a finite domain expansion and is intentionally not
    part of equality; tensor evaluation can use it to index compact domain-local
    arrays.
    """

    value: int
    domain: Domain | str | None = field(default=None, compare=False, repr=False)
    local_value: int | None = field(default=None, compare=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "value", _validate_nonnegative_int(self.value, "Orbital label"))
        if self.domain is not None:
            object.__setattr__(self, "domain", _coerce_domain(self.domain))
        if self.local_value is not None:
            object.__setattr__(
                self,
                "local_value",
                _validate_nonnegative_int(self.local_value, "Orbital local label"),
            )

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


def domain(
    name: str,
    size: int | None = None,
    *,
    start: int = 0,
    values: Iterable[int] | None = None,
) -> Domain:
    """Create an index domain.

    ``domain("occ", size=5)`` expands to global orbital labels ``0..4``.
    ``domain("virt", size=7, start=5)`` expands to ``5..11`` while still using
    local tensor-axis labels ``0..6``.  ``values=...`` may be used for arbitrary
    subsets.
    """

    return Domain(name, size=size, start=start, values=None if values is None else tuple(values))


def index(
    name: str,
    domain: Domain | str | None = None,
    *,
    spin_z2: int | None = None,
    momentum: int | None = None,
) -> Index:
    """Create a single symbolic index.

    ``p = index("p")`` is the common form.  A domain can be supplied as either
    a :class:`Domain` object or a domain name, e.g. ``i = index("i", "occ")``.
    Use :func:`indices` for several.
    """

    (out,) = indices(name, domain=domain, spin_z2=spin_z2, momentum=momentum)
    return out


def indices(
    names: str,
    domain: Domain | str | None = None,
    *,
    spin_z2: int | None = None,
    momentum: int | None = None,
) -> tuple[Index, ...]:
    """Create a tuple of symbolic indices.

    ``p, q = indices("p q")`` is the common form.  Use :func:`index` for a
    single index.
    """

    parts = names.replace(",", " ").split()
    if not parts:
        raise ValueError("indices() needs at least one name")
    return tuple(Index(p, spin_z2=spin_z2, momentum=momentum, domain=domain) for p in parts)


def spin_index(
    name: str, spin_z2: int | None = None, *, domain: Domain | str | None = None
) -> Index:
    """Create an index carrying optional ``2*S_z`` metadata."""

    return Index(name, spin_z2=spin_z2, domain=domain)


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


def _substitute_mode(m: Mode, subst: Mapping[IndexKey, Mode]) -> Mode:
    if isinstance(m, Index):
        ident = _index_identity(m)
        if ident in subst:
            return subst[ident]
    return m


def _replace_term_modes(
    term: Term, subst: Mapping[IndexKey, Mode], *, replace_summed: bool = True
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


def _domains_may_overlap(left: Domain, right: Domain) -> bool:
    left_values = left.finite_values()
    right_values = right.finite_values()
    if left_values is None or right_values is None:
        return True
    return not set(left_values).isdisjoint(right_values)


def _domain_contains_orbital(value: Domain, orbital: int) -> bool | None:
    values = value.finite_values()
    if values is not None:
        return orbital in values
    if _is_default_spin_orbital_domain(value):
        # Preserve the legacy symbolic assumption that a concrete orbital label
        # belongs to the full spin-orbital basis; finite backends still validate
        # labels against n_orbitals when expanding/compiling.
        return True
    return None


def _delta_action(left: Mode, right: Mode) -> str:
    """Classify whether a delta may be consumed symbolically.

    Deltas across different index domains are only eliminated when the domains
    are known-compatible.  Otherwise they are retained until finite expansion,
    where concrete orbital equalities can be evaluated safely.
    """

    if isinstance(left, Orbital) and isinstance(right, Orbital):
        return "drop" if left.value == right.value else "zero"
    if isinstance(left, Index) and isinstance(right, Index):
        if _index_domain(left) == _index_domain(right):
            return "unify"
        if _domains_may_overlap(_index_domain(left), _index_domain(right)):
            return "retain"
        return "zero"
    if isinstance(left, Index) and isinstance(right, Orbital):
        contains = _domain_contains_orbital(_index_domain(left), right.value)
        if contains is False:
            return "zero"
        return "unify" if contains is True else "retain"
    if isinstance(left, Orbital) and isinstance(right, Index):
        contains = _domain_contains_orbital(_index_domain(right), left.value)
        if contains is False:
            return "zero"
        return "unify" if contains is True else "retain"
    raise TypeError("Delta endpoints must be Index or Orbital modes")


def _canonicalize_delta_constraints(term: Term) -> Term | None:
    if not term.deltas:
        return term

    uf = _UnionFind()
    retained_input: list[Delta] = []
    for d in term.deltas:
        action = _delta_action(d.left, d.right)
        if action == "zero":
            return None
        if action == "drop":
            continue
        if action == "retain":
            retained_input.append(d)
            continue
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
            # All symbols in the equality class are compatible bound dummy
            # indices and/or constants, so the delta can be consumed by reducing
            # the finite domain.
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
    retained.extend(
        Delta(_substitute_mode(d.left, subst), _substitute_mode(d.right, subst))
        for d in retained_input
    )

    # Deduplicate retained free/cross-domain constraints canonically.
    unique: dict[tuple[Any, Any], Delta] = {}
    for d in retained:
        action = _delta_action(d.left, d.right)
        if action == "zero":
            return None
        if action == "drop" or _mode_key(d.left) == _mode_key(d.right):
            continue
        if _mode_key(d.right) < _mode_key(d.left):
            d = Delta(d.right, d.left)
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
                    f"_{len(mapping)}",
                    source.spin_z2,
                    source.momentum,
                    _index_domain(source),
                    _uid=len(mapping),
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


def _canonicalize_term_once(term: Term) -> Term | None:
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


def _canonicalize_term(term: Term) -> Term | None:
    """Canonicalize a term, iterating a single pass to a fixed point.

    One pass is not self-consistent: ``_rename_bound_dummies`` assigns the
    canonical dummy ids (``_0, _1, ...``) only *after* ``_canonicalize_op_runs``
    and ``_canonicalize_tensors`` have sorted (and signed) operators using the
    pre-canonical hygienic uids.  Relabeling can therefore leave a same-kind
    operator run (e.g. ``a†_p a†_q``) unsorted under its new ids, so a lone pass
    is neither idempotent nor independent of the order indices were bound in
    (e.g. the listing order in ``sum_``).  ``Expr.simplify`` relies on both
    properties for equality and term de-duplication, so we re-run until the
    structural key stops changing.  Convergence is effectively immediate; the
    loop records history only so a (so-far unobserved) cycle resolves to a
    deterministic representative instead of spinning forever.

    Terms with at most one bound dummy take a fast path: a bound dummy always
    sorts ahead of any free index or orbital regardless of its uid (bound mode
    keys are ``(1, 0, ...)`` < free ``(1, 1, ...)``), so reordering under
    relabeling needs >=2 dummies sharing a same-kind operator run or an
    antisymmetric tensor pair.  Below that threshold the first pass is already a
    fixed point, so the (common) zero-/one-dummy case skips the re-runs.
    """

    t = _canonicalize_term_once(term)
    if t is None:
        return None
    if len(t.summed) <= 1:
        return t
    history: list[Term] = []
    keys: list[tuple[Any, ...]] = []
    while True:
        key = t.structural_key()
        if key in keys:
            # At a fixed point the detected cycle has length 1 and ``min`` returns
            # that point (the common path).  A length >1 cycle is the defensive
            # case -- not produced by any known input -- where ``min`` collapses
            # the orbit to a single deterministic representative.
            cycle = history[keys.index(key) :]
            return min(cycle, key=lambda x: x.structural_key())
        history.append(t)
        keys.append(key)
        nxt = _canonicalize_term_once(t)
        if nxt is None:
            return None
        t = nxt


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


def _tensor_axis_value(port: Mode) -> int:
    if not isinstance(port, Orbital):
        raise ValueError("Cannot evaluate tensor with symbolic ports")
    return port.value if port.local_value is None else port.local_value


def _eval_tensor_value(values: Any, ports: tuple[Mode, ...]) -> Fraction:
    idx = tuple(_tensor_axis_value(p) for p in ports)
    global_idx = tuple(p.value for p in ports if isinstance(p, Orbital))
    if len(global_idx) != len(ports):
        raise ValueError("Cannot evaluate tensor with symbolic ports")
    if callable(values):
        val = values(*idx)
    elif isinstance(values, Mapping):
        try:
            val = values[idx]
        except KeyError:
            if global_idx == idx:
                raise
            val = values[global_idx]
    else:
        val = values
        for i in idx:
            val = val[i]
    return _to_fraction(float(val) if hasattr(val, "item") else val)


def _points_from_values(values: Iterable[int], *, context: str) -> tuple[tuple[int, int], ...]:
    concrete = _coerce_domain_values(values, context=context)
    return tuple((local, global_value) for local, global_value in enumerate(concrete))


def _points_from_size(size: int, *, start: int = 0) -> tuple[tuple[int, int], ...]:
    checked_size = _validate_nonnegative_int(size, "Domain override size")
    checked_start = _validate_nonnegative_int(start, "Domain override start")
    return tuple((local, checked_start + local) for local in range(checked_size))


def _points_from_domain_spec(name: str, spec: Any) -> tuple[tuple[int, int], ...]:
    if isinstance(spec, Domain):
        values = spec.finite_values()
        if values is None:
            raise ValueError(f"Domain override {name!r} must have finite size or values")
        return _points_from_values(values, context=f"Domain override {name!r}")
    if isinstance(spec, int):
        return _points_from_size(spec)
    return _points_from_values(spec, context=f"Domain override {name!r}")


def _domain_expansion_points(
    value: Domain,
    *,
    n_orbitals: int | None,
    domains: Mapping[str, Any] | None,
) -> tuple[tuple[int, int], ...]:
    override = None
    if domains is not None and value.name in domains:
        override = _points_from_domain_spec(value.name, domains[value.name])
    own = value.finite_values()
    if own is not None:
        # A concrete domain already fixes its own orbital labels.  Matching an
        # override by name would silently discard the domain's start/values, so
        # reject the conflict rather than shadow it.
        if override is not None:
            raise ValueError(
                f"Domain {value.name!r} is already finite; drop the expand_sums "
                f"override for {value.name!r} or give the domain a distinct name"
            )
        points = _points_from_values(own, context=f"Domain {value.name!r}")
    elif override is not None:
        points = override
    elif _is_default_spin_orbital_domain(value):
        if n_orbitals is None:
            raise ValueError("n_orbitals is required to expand spin_orbital sums")
        points = _points_from_size(n_orbitals)
    else:
        raise ValueError(
            f"Domain {value.name!r} has no finite size or values; pass "
            "domain(..., size=...), domain(..., values=...), or an expand_sums "
            "domain override"
        )

    if n_orbitals is not None:
        checked_n = _validate_nonnegative_int(n_orbitals, "n_orbitals")
        for _local, global_value in points:
            if global_value >= checked_n:
                raise ValueError(
                    f"Domain {value.name!r} contains orbital {global_value}, outside "
                    f"n_orbitals={checked_n}"
                )
    return points


def expand_sums(
    expr: Any,
    *,
    n_orbitals: int | None = None,
    tensor_values: Mapping[str, Any] | None = None,
    domains: Mapping[str, Any] | None = None,
) -> Expr:
    """Expand symbolic sums over finite index domains.

    Untyped/default indices expand over ``range(n_orbitals)``.  Typed indices
    expand over the finite data carried by their :class:`Domain`.  A domain that
    is not yet finite (e.g. a string-only domain name) instead draws its range
    from an override supplied by domain name via ``domains``, whose values may be
    a :class:`Domain`, an ``int`` size, or an iterable of orbital labels;
    supplying an override for an already-finite domain is rejected so a concrete
    domain's start/values are never silently discarded.
    Tensor factors with concrete ports are
    evaluated when their symbol appears in ``tensor_values``; for typed domains,
    tensor axes use domain-local coordinates while operators use global orbital
    labels.  Remaining symbolic tensors are preserved; sparse compilation
    requires all tensors to be evaluated.
    """

    if n_orbitals is not None:
        _validate_nonnegative_int(n_orbitals, "n_orbitals")
    tensor_values = tensor_values or {}
    expr = as_expr(expr).simplify()
    out = Expr.zero()
    for term in expr.terms:
        summed = list(term.summed)
        domain_points = [
            _domain_expansion_points(
                _index_domain(index),
                n_orbitals=n_orbitals,
                domains=domains,
            )
            for index in summed
        ]
        for assignment in product(*domain_points):
            subst = {
                _index_identity(index): Orbital(
                    global_value, _index_domain(index), local_value=local_value
                )
                for index, (local_value, global_value) in zip(
                    summed, assignment, strict=True
                )
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
    domains: Mapping[str, Any] | None = None,
) -> Any:
    if target != "sparse":
        raise NotImplementedError("NOMAD executable backend is target='sparse'")
    if n_orbitals is None:
        raise ValueError("n_orbitals is required for sparse compilation")
    lowered = expand_sums(
        normal_order(expr),
        n_orbitals=n_orbitals,
        tensor_values=tensor_values,
        domains=domains,
    )
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


def _summed_text(index: Index) -> str:
    label = _mode_text(index)
    domain_value = _index_domain(index)
    if _is_default_spin_orbital_domain(domain_value):
        return label
    return f"{label}∈{domain_value.name}"


def _summed_latex(index: Index) -> str:
    label = _mode_text(index)
    domain_value = _index_domain(index)
    if _is_default_spin_orbital_domain(domain_value):
        return label
    return label + r"\in " + domain_value.name


def text(expr: Any) -> str:
    expr = as_expr(expr).simplify()
    if not expr.terms:
        return "0"
    parts = []
    for term in expr.terms:
        factors = []
        if term.summed:
            factors.append("Σ_" + ",".join(_summed_text(i) for i in term.summed))
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
            factors.append(r"\sum_{" + ",".join(_summed_latex(i) for i in term.summed) + "}")
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
