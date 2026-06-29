"""NOMAD Python frontend and reference runtime.

This module is intentionally small, deterministic, and dependency-light.  The
data model is a weighted sum of NCIR/Wick term records, where normal ordering
may rewrite one term into many terms.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from fractions import Fraction
from itertools import combinations, count, islice, product
from math import comb
from numbers import Integral
from typing import Any, cast, overload

Number = int | float | Fraction
IndexKey = tuple[Any, ...]
FiniteValues = range | tuple[int, ...]
# Per-index charge metadata: a name->value mapping or an iterable of such pairs.
IndexCharges = Mapping[str, int] | Iterable[tuple[str, int]]


def _validate_int(value: Any, what: str) -> int:
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{what} must be an integer")
    return int(value)


def _validate_nonnegative_int(value: int, what: str) -> int:
    checked = _validate_int(value, what)
    if checked < 0:
        raise ValueError(f"{what} must be a non-negative integer")
    return checked


def _validate_positive_int(value: int, what: str) -> int:
    checked = _validate_int(value, what)
    if checked <= 0:
        raise ValueError(f"{what} must be a positive integer")
    return checked


def _coerce_domain_values(values: Iterable[int], *, context: str) -> tuple[int, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{context} values must be an iterable of integer orbital labels")
    out = tuple(_validate_nonnegative_int(v, f"{context} value") for v in values)
    if len(set(out)) != len(out):
        raise ValueError(f"{context} values must not contain duplicates")
    return out


def _coerce_charge_values(values: Iterable[int], *, context: str) -> tuple[int, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{context} values must be an iterable of integer charges")
    return tuple(_validate_int(v, f"{context} value") for v in values)


@dataclass(frozen=True)
class Charge:
    """Per-orbital additive charge values.

    ``values[p]`` is the charge carried by occupying orbital ``p``.  When
    ``modulus`` is supplied, sector comparisons and charge deltas are evaluated
    modulo that positive integer, which covers crystal momentum, parity-like
    labels, and any other additive cyclic quantum number.
    """

    values: tuple[int, ...]
    modulus: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "values",
            _coerce_charge_values(self.values, context="Charge"),
        )
        if self.modulus is not None:
            object.__setattr__(self, "modulus", _validate_positive_int(self.modulus, "modulus"))


def charge(values: Iterable[int], *, modulus: int | None = None) -> Charge:
    """Create a per-orbital additive charge vector.

    A bare sequence such as ``[1, 1, 1, 1]`` can be used directly in most APIs;
    this helper is only needed when the charge is conserved modulo an integer.
    """

    return Charge(tuple(values), modulus=modulus)


def _coerce_charge_spec(
    name: str,
    spec: Any,
    *,
    n_orbitals: int | None = None,
) -> Charge:
    if isinstance(spec, Charge):
        out = spec
    elif isinstance(spec, Mapping):
        if "values" not in spec:
            raise ValueError(
                f"Charge {name!r} mapping specifications must contain a 'values' entry"
            )
        out = Charge(tuple(spec["values"]), modulus=spec.get("modulus"))
    elif (
        isinstance(spec, tuple)
        and len(spec) == 2
        and not isinstance(spec[0], Integral)
        and (spec[1] is None or isinstance(spec[1], Integral))
    ):
        out = Charge(tuple(spec[0]), modulus=cast("int | None", spec[1]))
    else:
        out = Charge(tuple(spec))

    if n_orbitals is not None and len(out.values) != n_orbitals:
        raise ValueError(
            f"Charge {name!r} has length {len(out.values)}, expected n_orbitals={n_orbitals}"
        )
    return out


def _coerce_charges(
    charges: Mapping[str, Any] | None,
    *,
    n_orbitals: int | None = None,
) -> dict[str, Charge]:
    out: dict[str, Charge] = {}
    for name, spec in (charges or {}).items():
        if not isinstance(name, str) or not name:
            raise ValueError("Charge names must be non-empty strings")
        out[name] = _coerce_charge_spec(name, spec, n_orbitals=n_orbitals)
    return dict(sorted(out.items()))


def _coerce_index_charges(charges: Any) -> tuple[tuple[str, int], ...]:
    if charges is None:
        return ()
    if isinstance(charges, (str, bytes)):
        raise TypeError("Index charges must be a mapping or iterable of (name, value) pairs")
    raw_items = charges.items() if isinstance(charges, Mapping) else charges
    out: dict[str, int] = {}
    for raw_name, raw_value in raw_items:
        if not isinstance(raw_name, str) or not raw_name:
            raise ValueError("Index charge names must be non-empty strings")
        value = _validate_int(raw_value, f"Index charge {raw_name!r}")
        if raw_name in out and out[raw_name] != value:
            raise ValueError(f"Index charge {raw_name!r} was provided more than once")
        out[raw_name] = value
    return tuple(sorted(out.items()))


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
            object.__setattr__(self, "size", _validate_nonnegative_int(self.size, "Domain size"))
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
    metadata fields let the frontend carry additive charge annotations before a
    finite orbital charge table is available.  ``spin_z2`` and ``momentum`` are
    legacy convenience aliases for ``charges={"Sz2": ..., "K": ...}``.
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
    charges: Any = None

    def __post_init__(self) -> None:
        if not self.name or not isinstance(self.name, str):
            raise ValueError("Index name must be a non-empty string")
        object.__setattr__(self, "domain", _coerce_domain(self.domain))
        charge_map = dict(_coerce_index_charges(self.charges))
        if self.spin_z2 is not None:
            spin_z2 = _validate_int(self.spin_z2, "Index spin_z2")
            if "Sz2" in charge_map and charge_map["Sz2"] != spin_z2:
                raise ValueError("Index spin_z2 conflicts with charges['Sz2']")
            charge_map.setdefault("Sz2", spin_z2)
            object.__setattr__(self, "spin_z2", spin_z2)
        elif "Sz2" in charge_map:
            object.__setattr__(self, "spin_z2", charge_map["Sz2"])
        if self.momentum is not None:
            momentum = _validate_int(self.momentum, "Index momentum")
            if "K" in charge_map and charge_map["K"] != momentum:
                raise ValueError("Index momentum conflicts with charges['K']")
            charge_map.setdefault("K", momentum)
            object.__setattr__(self, "momentum", momentum)
        elif "K" in charge_map:
            object.__setattr__(self, "momentum", charge_map["K"])
        object.__setattr__(self, "charges", tuple(sorted(charge_map.items())))
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


def _index_charge_items(index: Index) -> tuple[tuple[str, int], ...]:
    # ``Index.__post_init__`` canonicalizes the public constructor's flexible
    # ``charges`` input into this exact tuple shape.  The cast keeps the runtime
    # representation precise without rejecting mapping inputs at construction.
    return cast(tuple[tuple[str, int], ...], index.charges)


def _index_domain(index: Index) -> Domain:
    return _coerce_domain(index.domain)


def _index_metadata_key(index: Index) -> tuple[tuple[tuple[str, int], ...], tuple[Any, ...]]:
    return (
        _index_charge_items(index),
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
            charges=_index_charge_items(index),
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


# Tensor axes are domain-local: an expanded ``Orbital`` carrying ``local_value``
# addresses a tensor by that local coordinate, while operators (and deltas) use
# the global orbital label.  Keying/rendering tensor ports by the *global* label
# would conflate distinct local entries of one symbol across offset domains --
# e.g. occ-local 1 and virt-local 0 both sitting at global orbital 2 -- merging
# them or vanishing an antisymmetric pair that is off-diagonal in local axes.
# So tensor-port identity and display follow the same local axis that
# ``_tensor_axis_value`` evaluates; ``_mode_key``/``_mode_text`` stay global for
# operators.  Symbolic ports and explicit user orbitals (``local_value is None``)
# are unchanged, so this only affects ports produced by typed-domain expansion.
def _tensor_port_axis(port: Mode) -> int | None:
    if isinstance(port, Orbital):
        return port.value if port.local_value is None else port.local_value
    return None


def _tensor_port_key(port: Mode) -> tuple[Any, ...]:
    axis = _tensor_port_axis(port)
    return (0, axis) if axis is not None else _mode_key(port)


def _tensor_port_text(port: Mode) -> str:
    axis = _tensor_port_axis(port)
    return str(axis) if axis is not None else _mode_text(port)


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
    charges: IndexCharges | None = None,
) -> Index:
    """Create a single symbolic index.

    ``p = index("p")`` is the common form.  A domain can be supplied as either
    a :class:`Domain` object or a domain name, e.g. ``i = index("i", "occ")``.
    Use :func:`indices` for several.
    """

    (out,) = indices(
        name,
        domain=domain,
        spin_z2=spin_z2,
        momentum=momentum,
        charges=charges,
    )
    return out


def indices(
    names: str,
    domain: Domain | str | None = None,
    *,
    spin_z2: int | None = None,
    momentum: int | None = None,
    charges: IndexCharges | None = None,
) -> tuple[Index, ...]:
    """Create a tuple of symbolic indices.

    ``p, q = indices("p q")`` is the common form.  Use :func:`index` for a
    single index.
    """

    parts = names.replace(",", " ").split()
    if not parts:
        raise ValueError("indices() needs at least one name")
    return tuple(
        Index(p, spin_z2=spin_z2, momentum=momentum, domain=domain, charges=charges) for p in parts
    )


def spin_index(
    name: str,
    spin_z2: int | None = None,
    *,
    domain: Domain | str | None = None,
    charges: IndexCharges | None = None,
) -> Index:
    """Create an index carrying optional ``2*S_z`` metadata."""

    return Index(name, spin_z2=spin_z2, domain=domain, charges=charges)


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
            tuple(_tensor_port_key(p) for p in self.ports),
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

    def charge_delta(self, *, charges: Mapping[str, Any] | None = None) -> dict[str, int]:
        return operator_charge_delta(self, charges=charges)


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

    def charge_delta(self, *, charges: Mapping[str, Any] | None = None) -> dict[str, int]:
        return term_charge_delta(self, charges=charges)

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


def _domain_local_value(value: Domain, orbital: int) -> int | None:
    """Domain-local axis position of a global orbital label.

    Mirrors the provenance :func:`expand_sums` attaches to expanded orbitals so
    that pinning a typed index to a concrete in-domain orbital keeps addressing
    tensors by their domain-local coordinate.  Returns ``None`` for a non-finite
    (default/string) domain, whose tensor axes already coincide with the global
    label.
    """

    values = value.finite_values()
    if values is None:
        return None
    return values.index(orbital)


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
                # Pin each dummy to the constant in its *own* domain so the
                # domain-local tensor axis survives (a single shared Orbital
                # would lose per-index local_value when the dummies span
                # different offset domains).
                pinned = consts[0]
                for key, idx in bound:
                    pin_domain = _index_domain(idx)
                    subst[key] = Orbital(
                        pinned,
                        pin_domain,
                        local_value=_domain_local_value(pin_domain, pinned),
                    )
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
        if _tensor_port_key(ports[a]) == _tensor_port_key(ports[b]):
            return None
        if _tensor_port_key(ports[a]) > _tensor_port_key(ports[b]):
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
                    charges=_index_charge_items(source),
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

    Terms with at most one bound dummy *and no surviving deltas* take a fast
    path: a bound dummy always sorts ahead of any free index or orbital
    regardless of its uid (bound mode keys are ``(1, 0, ...)`` < free
    ``(1, 1, ...)``), so reordering under relabeling needs >=2 dummies sharing a
    same-kind operator run or an antisymmetric tensor pair.  A leftover delta is
    the other way one pass can fall short of a fixed point: a retained
    cross-domain delta between summed dummies can become consumable on the next
    pass once a sibling delta pins one of its endpoints to a constant (e.g.
    ``δ(i,j) δ(i,1)`` over overlapping domains pins ``i=1``, exposing ``δ(1,j)``
    for the following pass).  So the fast path must not fire while any delta
    survives, even with a single dummy.  With neither condition present the
    first pass is already a fixed point, so the (common) zero-/one-dummy,
    delta-free case skips the re-runs.
    """

    t = _canonicalize_term_once(term)
    if t is None:
        return None
    if len(t.summed) <= 1 and not t.deltas:
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


def _mode_charge_value(mode: Mode, name: str, specs: Mapping[str, Charge]) -> int | None:
    if name == "N":
        if isinstance(mode, Orbital):
            if name in specs:
                return _orbital_charge_value(mode.value, name, specs[name])
            return 1
        if isinstance(mode, Index):
            items = dict(_index_charge_items(mode))
            if name in items:
                return items[name]
            # Without an explicit per-orbital ``N`` table every mode carries one
            # particle.  With a non-uniform table the symbolic index's particle
            # number is genuinely unknown until it expands to a concrete orbital,
            # so report it as such rather than assuming 1 (which would let
            # ``prune_by_charge`` drop terms whose expansions still qualify).
            return None if name in specs else 1
        return 1
    if isinstance(mode, Orbital):
        spec = specs.get(name)
        if spec is None:
            return None
        return _orbital_charge_value(mode.value, name, spec)
    return dict(_index_charge_items(mode)).get(name)


def _mode_unknown_charge_key(mode: Mode, name: str) -> tuple[Any, ...]:
    if isinstance(mode, Orbital):
        return ("orbital", mode.value, name)
    return ("index", _index_identity(mode), name)


def _charge_delta_sign(op: Op) -> int:
    return 1 if op.kind == "create" else -1


def _normalize_charge_value(value: int, spec: Charge | None) -> int:
    if spec is None or spec.modulus is None:
        return value
    return value % spec.modulus


def _orbital_charge_value(orbital: int, name: str, spec: Charge) -> int:
    checked = _validate_nonnegative_int(orbital, "Orbital label")
    if checked >= len(spec.values):
        raise ValueError(
            f"Charge {name!r} has length {len(spec.values)} and does not cover orbital {checked}"
        )
    return spec.values[checked]


def _charge_delta_matches(
    actual: int,
    expected: int,
    spec: Charge | None,
) -> bool:
    if spec is None or spec.modulus is None:
        return actual == expected
    return (actual - expected) % spec.modulus == 0


def _term_charge_delta_analysis(
    term: Term,
    specs: Mapping[str, Charge],
    *,
    names: Iterable[str] = (),
) -> dict[str, int]:
    candidate_names = set(names)
    candidate_names.add("N")
    candidate_names.update(specs)
    for op in term.ops:
        if isinstance(op.mode, Index):
            candidate_names.update(name for name, _value in _index_charge_items(op.mode))

    known: dict[str, int] = {}
    for name in sorted(candidate_names):
        numeric = 0
        unknown: dict[tuple[Any, ...], int] = {}
        for op in term.ops:
            signed = _charge_delta_sign(op)
            value = _mode_charge_value(op.mode, name, specs)
            if value is None:
                key = _mode_unknown_charge_key(op.mode, name)
                unknown[key] = unknown.get(key, 0) + signed
            else:
                numeric += signed * value
        unknown = {key: coeff for key, coeff in unknown.items() if coeff}
        if not unknown:
            known[name] = _normalize_charge_value(numeric, specs.get(name))
    return known


def operator_charge_delta(op: Op, *, charges: Mapping[str, Any] | None = None) -> dict[str, int]:
    """Known additive charge delta for one creation/annihilation operator.

    Concrete orbital operators use the supplied per-orbital charge table.  Index
    operators use metadata stored on the :class:`Index`.  Unknown symbolic
    charges are omitted; particle number ``N`` is always known unless explicitly
    overridden by a per-orbital ``N`` charge table.
    """

    specs = _coerce_charges(charges)
    candidate_names = {"N", *specs.keys()}
    if isinstance(op.mode, Index):
        candidate_names.update(name for name, _value in _index_charge_items(op.mode))

    out: dict[str, int] = {}
    signed = _charge_delta_sign(op)
    for name in sorted(candidate_names):
        value = _mode_charge_value(op.mode, name, specs)
        if value is not None:
            out[name] = _normalize_charge_value(signed * value, specs.get(name))
    return out


def term_charge_delta(term: Term, *, charges: Mapping[str, Any] | None = None) -> dict[str, int]:
    """Known additive charge delta for a term's operator word.

    If a charge is symbolic but cancels structurally (for example
    ``a†(p) a(p)``), it is reported as zero.  Charges that remain genuinely
    unknown are omitted so pruning never discards a term unless a violation is
    certain.
    """

    return _term_charge_delta_analysis(term, _coerce_charges(charges))


def charge_delta(value: Op | Term | Expr, *, charges: Mapping[str, Any] | None = None) -> Any:
    """Return additive charge deltas for an operator, term, or expression.

    Expressions must have a single common known delta across all terms; use
    :func:`term_charge_delta` when per-term information is desired.
    """

    if isinstance(value, Op):
        return operator_charge_delta(value, charges=charges)
    if isinstance(value, Term):
        return term_charge_delta(value, charges=charges)
    if isinstance(value, Expr):
        deltas = tuple(term_charge_delta(t, charges=charges) for t in value.terms)
        if not deltas:
            return {}
        first = deltas[0]
        if all(d == first for d in deltas):
            return first
        raise ValueError("Expression contains terms with different charge deltas")
    raise TypeError(f"Cannot compute charge delta for {type(value)!r}")


def _charge_name_set(names: Iterable[str], *, context: str) -> tuple[str, ...]:
    out = []
    seen: set[str] = set()
    for name in names:
        if not isinstance(name, str) or not name:
            raise ValueError(f"{context} charge names must be non-empty strings")
        if name not in seen:
            out.append(name)
            seen.add(name)
    return tuple(out)


def _coerce_charge_target(
    target: Mapping[str, int] | None,
    *,
    charges: Mapping[str, Any] | None = None,
) -> dict[str, int]:
    out: dict[str, int] = {}
    for name, value in (target or {}).items():
        if not isinstance(name, str) or not name:
            raise ValueError("Target charge names must be non-empty strings")
        out[name] = _validate_int(value, f"Target charge {name!r}")

    # Legacy determinant sectors accepted {"Sz": ...} as an alias for Sz2.  Keep
    # that behavior unless the user explicitly supplies a separate charge named
    # "Sz".
    if "Sz" in out and (charges is None or "Sz" not in charges):
        sz_value = out.pop("Sz")
        if "Sz2" in out and out["Sz2"] != sz_value:
            raise ValueError("Target cannot specify conflicting 'Sz' and 'Sz2' values")
        out["Sz2"] = sz_value
    return dict(sorted(out.items()))


def _required_charge_deltas(
    *,
    delta_n: int | None,
    target: Mapping[str, int] | None,
    conserve: Iterable[str] | None,
    delta: Mapping[str, int] | None,
    charges: Mapping[str, Any] | None,
) -> dict[str, int]:
    required: dict[str, int] = {}
    for name in _coerce_charge_target(target, charges=charges):
        required[name] = 0
    if conserve is not None:
        # Route conserve names through the same coercion as target/delta so the
        # legacy ``Sz`` -> ``Sz2`` alias is honored consistently across all three.
        for name in _coerce_charge_target(
            dict.fromkeys(_charge_name_set(conserve, context="conserve"), 0),
            charges=charges,
        ):
            required[name] = 0
    # ``delta_n`` is the convenience shortcut for the ``N`` entry of ``delta``;
    # apply it first so an explicit ``delta={"N": ...}`` overrides it (including
    # the implicit ``delta_n=0`` default) rather than being silently clobbered.
    if delta_n is not None:
        required["N"] = _validate_int(delta_n, "delta_n")
    for name, value in _coerce_charge_target(delta, charges=charges).items():
        required[name] = value
    return dict(sorted(required.items()))


def prune_by_charge(
    expr: Any,
    *,
    delta_n: int | None = 0,
    charges: Mapping[str, Any] | None = None,
    target: Mapping[str, int] | None = None,
    conserve: Iterable[str] | None = None,
    delta: Mapping[str, int] | None = None,
) -> Expr:
    """Prune terms whose additive charge delta violates conservation laws.

    The legacy call ``prune_by_charge(expr)`` keeps only particle-number
    conserving terms (``ΔN = 0``).

    Conservation is requested by name with ``target={...}`` or
    ``conserve=[...]``; each named charge must have zero term delta.  Only the
    *names* in ``target`` matter here -- its values select a sector elsewhere
    (see :func:`basis_sector`) and are ignored for pruning.  Use
    ``delta={"Q": q}`` to require a specific non-zero delta instead; ``delta``
    entries override the conservation defaults, including ``delta_n``.

    ``delta_n`` defaults to ``0``, so particle number is conserved alongside any
    ``target`` / ``conserve`` names; pass ``delta_n=None`` to drop that default,
    or ``delta={"N": k}`` to require a specific ``ΔN``.

    A term with an unknown symbolic delta is kept.  It will be pruned later once
    indices are expanded to concrete orbitals, or retained if the violation
    cannot be proven.
    """

    expr = as_expr(expr).simplify()
    specs = _coerce_charges(charges)
    required = _required_charge_deltas(
        delta_n=delta_n,
        target=target,
        conserve=conserve,
        delta=delta,
        charges=charges,
    )
    if not required:
        return cast(Expr, expr)

    kept = []
    required_names = tuple(required)
    for term in expr.terms:
        known = _term_charge_delta_analysis(term, specs, names=required_names)
        violates = False
        for name, expected in required.items():
            actual = known.get(name)
            if actual is None:
                continue
            if not _charge_delta_matches(actual, expected, specs.get(name)):
                violates = True
                break
        if not violates:
            kept.append(term)
    return Expr(tuple(kept)).simplify()


# ---------------------------------------------------------------------------
# Finite expansion, determinant backend, and exports
# ---------------------------------------------------------------------------


def _tensor_axis_value(port: Mode) -> int:
    if not isinstance(port, Orbital):
        raise ValueError("Cannot evaluate tensor with symbolic ports")
    return port.value if port.local_value is None else port.local_value


def _eval_tensor_value(values: Any, ports: tuple[Mode, ...]) -> Fraction:
    # Tensor axes are domain-local: every index addresses the tensor by its
    # domain-local coordinate, never its global orbital label.  The callable,
    # mapping, and nested-sequence backends all honour this single contract
    # (``_tensor_axis_value`` rejects symbolic ports up front).
    idx = tuple(_tensor_axis_value(p) for p in ports)
    if callable(values):
        val = values(*idx)
    elif isinstance(values, Mapping):
        val = values[idx]
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
    elif _is_default_spin_orbital_domain(value):
        # The default basis is sized by n_orbitals, never by a domains override;
        # accepting one here would silently shrink a default sum out from under
        # n_orbitals.  Reject it for the same reason an already-finite domain
        # rejects a name collision above.
        if override is not None:
            raise ValueError(
                "the default spin_orbital domain is sized by n_orbitals, not by an "
                "expand_sums override; drop domains['spin_orbital'] or give the index "
                "a named domain"
            )
        if n_orbitals is None:
            raise ValueError("n_orbitals is required to expand spin_orbital sums")
        points = _points_from_size(n_orbitals)
    elif override is not None:
        points = override
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
    a :class:`Domain`, an ``int`` size, or an iterable of orbital labels.  An
    override is matched purely by the ``domains`` key (the index's domain name);
    when it is a :class:`Domain`, only its finite range/values are used and its
    own ``name`` is ignored.  Supplying an override for an already-finite domain
    -- or for the default ``spin_orbital`` basis, which is sized by ``n_orbitals``
    -- is rejected so a domain's start/values are never silently discarded.
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
                for index, (local_value, global_value) in zip(summed, assignment, strict=True)
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


def _default_spin_z2(n_orbitals: int) -> tuple[int, ...]:
    return tuple(1 if i % 2 == 0 else -1 for i in range(n_orbitals))


def _complete_charge_specs(
    n_orbitals: int,
    charges: Mapping[str, Any] | None,
    target: Mapping[str, int] | None,
    *,
    spin_z2: Sequence[int] | None = None,
    require_target: bool = False,
) -> dict[str, Charge]:
    checked_n = _validate_nonnegative_int(n_orbitals, "n_orbitals")
    target_map = _coerce_charge_target(target, charges=charges)
    specs = _coerce_charges(charges, n_orbitals=checked_n)

    if "N" in target_map and "N" not in specs:
        specs["N"] = Charge(tuple(1 for _ in range(checked_n)))
    if "Sz2" in target_map and "Sz2" not in specs:
        values = tuple(spin_z2) if spin_z2 is not None else _default_spin_z2(checked_n)
        specs["Sz2"] = _coerce_charge_spec("Sz2", values, n_orbitals=checked_n)

    if require_target:
        missing = sorted(name for name in target_map if name not in specs)
        if missing:
            names = ", ".join(repr(name) for name in missing)
            raise ValueError(f"No per-orbital charge values supplied for target charge(s): {names}")
    return dict(sorted(specs.items()))


def _det_charge_value(det: int, name: str, spec: Charge) -> int:
    total = 0
    for orbital, value in enumerate(spec.values):
        if det & (1 << orbital):
            total += value
    return _normalize_charge_value(total, spec)


def determinant_charges(
    det: int,
    *,
    n_orbitals: int,
    charges: Mapping[str, Any] | None = None,
) -> dict[str, int]:
    """Compute additive charges of a determinant bitstring."""

    checked_n = _validate_nonnegative_int(n_orbitals, "n_orbitals")
    checked_det = _validate_nonnegative_int(det, "determinant")
    if checked_det >= (1 << checked_n):
        raise ValueError(f"determinant {checked_det} does not fit in n_orbitals={checked_n}")

    specs = _coerce_charges(charges, n_orbitals=checked_n)
    out: dict[str, int] = {}
    if "N" not in specs:
        out["N"] = checked_det.bit_count()
    for name, spec in specs.items():
        out[name] = _det_charge_value(checked_det, name, spec)
    return dict(sorted(out.items()))


def _det_matches_charge_target(
    det: int,
    specs: Mapping[str, Charge],
    target: Mapping[str, int],
) -> bool:
    for name, wanted in target.items():
        spec = specs[name]
        actual = _det_charge_value(det, name, spec)
        if not _charge_delta_matches(actual, wanted, spec):
            return False
    return True


def _iter_fixed_weight_determinants(n_orbitals: int, n_particles: int) -> Iterable[int]:
    """Yield determinants with exactly ``n_particles`` occupied orbitals.

    The sequence is emitted in the same numeric order as scanning all bitstrings
    with ``range(1 << n_orbitals)`` and keeping a fixed popcount, but it visits
    only the requested fixed-weight determinants.  The implementation is the
    standard "next higher integer with the same number of set bits" (snoob)
    recurrence.
    """

    if n_particles > n_orbitals:
        return
    if n_particles == 0:
        yield 0
        return

    det = (1 << n_particles) - 1
    limit = 1 << n_orbitals
    while det < limit:
        yield det
        smallest = det & -det
        ripple = det + smallest
        det = (((ripple ^ det) >> 2) // smallest) | ripple


def _coerce_orbital_labels(values: Iterable[int], *, context: str) -> tuple[int, ...]:
    return tuple(sorted(_coerce_domain_values(values, context=context)))


def _coerce_spin_orbital_blocks(
    spin_up_orbs: int | Sequence[int],
    spin_down_orbs: int | Sequence[int],
    *,
    n_orbitals: int | None = None,
) -> tuple[tuple[int, ...], tuple[int, ...], int]:
    if isinstance(spin_up_orbs, Integral) and isinstance(spin_down_orbs, Integral):
        n_up = _validate_nonnegative_int(cast(Any, spin_up_orbs), "spin_up_orbs")
        n_down = _validate_nonnegative_int(cast(Any, spin_down_orbs), "spin_down_orbs")
        total = n_up + n_down
        if n_orbitals is not None:
            checked_n = _validate_nonnegative_int(n_orbitals, "n_orbitals")
            if checked_n != total:
                raise ValueError(
                    "n_orbitals must equal spin_up_orbs + spin_down_orbs "
                    "when spin orbital counts are supplied"
                )
        up_orbitals = tuple(range(n_up))
        down_orbitals = tuple(range(n_up, total))
        return up_orbitals, down_orbitals, total

    if isinstance(spin_up_orbs, Integral) or isinstance(spin_down_orbs, Integral):
        raise TypeError("spin_up_orbs and spin_down_orbs must both be counts or both be sequences")

    up_values = cast(Sequence[int], spin_up_orbs)
    down_values = cast(Sequence[int], spin_down_orbs)
    up_orbitals = _coerce_orbital_labels(up_values, context="spin_up_orbs")
    down_orbitals = _coerce_orbital_labels(down_values, context="spin_down_orbs")
    overlap = set(up_orbitals) & set(down_orbitals)
    if overlap:
        first = min(overlap)
        raise ValueError(f"spin-up and spin-down orbital sets overlap at orbital {first}")

    inferred_n = max(up_orbitals + down_orbitals, default=-1) + 1
    if n_orbitals is None:
        checked_n = inferred_n
    else:
        checked_n = _validate_nonnegative_int(n_orbitals, "n_orbitals")
        for orbital in up_orbitals + down_orbitals:
            if orbital >= checked_n:
                raise ValueError(f"spin orbital {orbital} does not fit in n_orbitals={checked_n}")
    return up_orbitals, down_orbitals, checked_n


def _iter_orbital_combination_masks(orbitals: Sequence[int], n_particles: int) -> Iterable[int]:
    if n_particles > len(orbitals):
        return
    if n_particles == 0:
        yield 0
        return
    if tuple(orbitals) == tuple(range(len(orbitals))):
        yield from _iter_fixed_weight_determinants(len(orbitals), n_particles)
        return
    for combo in combinations(orbitals, n_particles):
        det = 0
        for orbital in combo:
            det |= 1 << orbital
        yield det


def _sector_size(n_orbitals: int, n_particles: int) -> int:
    if n_particles < 0 or n_particles > n_orbitals:
        return 0
    return comb(n_orbitals, n_particles)


@dataclass(frozen=True, eq=False)
class DeterminantBasis(Sequence[int]):
    """Lazy determinant sequence produced by :func:`determinant_basis`."""

    _size: int
    _iter_factory: Callable[[], Iterable[int]] = field(repr=False)
    _description: str = "determinants"

    def __len__(self) -> int:
        return self._size

    def __iter__(self) -> Iterator[int]:
        return iter(self._iter_factory())

    @overload
    def __getitem__(self, index: int) -> int: ...

    @overload
    def __getitem__(self, index: slice) -> tuple[int, ...]: ...

    def __getitem__(self, index: int | slice) -> int | tuple[int, ...]:
        if isinstance(index, slice):
            start, stop, step = index.indices(self._size)
            if step == 1:
                return tuple(islice(self, start, stop))
            return tuple(self[i] for i in range(start, stop, step))

        offset = index + self._size if index < 0 else index
        if offset < 0 or offset >= self._size:
            raise IndexError("determinant basis index out of range")
        return next(islice(self, offset, None))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Sequence):
            return False
        if len(self) != len(other):
            return False
        return all(left == right for left, right in zip(self, other, strict=True))

    def __repr__(self) -> str:  # pragma: no cover - convenience for REPLs
        return f"DeterminantBasis(size={self._size}, {self._description})"

    def to_tuple(self) -> tuple[int, ...]:
        """Materialize the basis as a tuple."""

        return tuple(self)


def _spin_blocks_are_ordered(
    low_orbitals: Sequence[int],
    high_orbitals: Sequence[int],
) -> bool:
    return not low_orbitals or not high_orbitals or max(low_orbitals) < min(high_orbitals)


def _iter_spin_resolved_determinants(
    up_orbitals: Sequence[int],
    down_orbitals: Sequence[int],
    n_up: int,
    n_down: int,
) -> Iterable[int]:
    if n_up > len(up_orbitals) or n_down > len(down_orbitals):
        return

    if _spin_blocks_are_ordered(up_orbitals, down_orbitals):
        lower_orbitals, lower_count = up_orbitals, n_up
        higher_orbitals, higher_count = down_orbitals, n_down
    elif _spin_blocks_are_ordered(down_orbitals, up_orbitals):
        lower_orbitals, lower_count = down_orbitals, n_down
        higher_orbitals, higher_count = up_orbitals, n_up
    else:
        combined = (
            up_mask | down_mask
            for down_mask in _iter_orbital_combination_masks(down_orbitals, n_down)
            for up_mask in _iter_orbital_combination_masks(up_orbitals, n_up)
        )
        yield from sorted(combined)
        return

    lower_masks = tuple(_iter_orbital_combination_masks(lower_orbitals, lower_count))
    for higher_mask in _iter_orbital_combination_masks(higher_orbitals, higher_count):
        for lower_mask in lower_masks:
            yield higher_mask | lower_mask


def determinant_basis(
    n_orbitals: int | None = None,
    *,
    N: int | None = None,
    spin_up_orbs: int | Sequence[int] | None = None,
    spin_down_orbs: int | Sequence[int] | None = None,
    N_up: int | None = None,
    N_down: int | None = None,
) -> DeterminantBasis:
    """Generate a lazy determinant basis directly in a particle-number sector.

    The spin-orbital form ``determinant_basis(n_orbitals=n, N=k)`` returns all
    determinants with exactly ``k`` occupied orbitals, in ascending determinant
    order.  Omitting ``N`` returns the full determinant basis.

    The spin-resolved form
    ``determinant_basis(spin_up_orbs=nup, spin_down_orbs=ndown, N_up=ku, N_down=kd)``
    treats the first ``nup`` orbital labels as spin-up and the next ``ndown`` as
    spin-down.  ``spin_up_orbs`` and ``spin_down_orbs`` may also be explicit,
    disjoint sequences of orbital labels, which is useful for interleaved spin
    layouts such as ``[0, 2, ...]`` / ``[1, 3, ...]``.

    Only combinations inside the requested sector are generated; the full
    ``2**n`` bitstring space is not scanned.
    """

    spin_args = (spin_up_orbs, spin_down_orbs, N_up, N_down)
    if any(arg is not None for arg in spin_args):
        if spin_up_orbs is None or spin_down_orbs is None or N_up is None or N_down is None:
            raise ValueError(
                "spin-resolved determinant_basis requires spin_up_orbs, "
                "spin_down_orbs, N_up, and N_down"
            )
        checked_n_up = _validate_nonnegative_int(N_up, "N_up")
        checked_n_down = _validate_nonnegative_int(N_down, "N_down")
        if N is not None:
            checked_N = _validate_nonnegative_int(N, "N")
            if checked_N != checked_n_up + checked_n_down:
                raise ValueError("N must equal N_up + N_down when both are supplied")
        up_orbitals, down_orbitals, _checked_n = _coerce_spin_orbital_blocks(
            spin_up_orbs,
            spin_down_orbs,
            n_orbitals=n_orbitals,
        )
        size = _sector_size(len(up_orbitals), checked_n_up) * _sector_size(
            len(down_orbitals), checked_n_down
        )
        description = (
            f"spin_up_orbs={len(up_orbitals)}, spin_down_orbs={len(down_orbitals)}, "
            f"N_up={checked_n_up}, N_down={checked_n_down}"
        )
        return DeterminantBasis(
            size,
            lambda: _iter_spin_resolved_determinants(
                up_orbitals, down_orbitals, checked_n_up, checked_n_down
            ),
            description,
        )

    if n_orbitals is None:
        raise ValueError("n_orbitals is required unless spin-resolved orbital blocks are supplied")
    checked_n = _validate_nonnegative_int(n_orbitals, "n_orbitals")
    if N is None:
        return DeterminantBasis(
            1 << checked_n,
            lambda: range(1 << checked_n),
            f"n_orbitals={checked_n}",
        )
    checked_N = _validate_nonnegative_int(N, "N")
    return DeterminantBasis(
        _sector_size(checked_n, checked_N),
        lambda: _iter_fixed_weight_determinants(checked_n, checked_N),
        f"n_orbitals={checked_n}, N={checked_N}",
    )


def _is_unit_particle_charge(spec: Charge) -> bool:
    return spec.modulus is None and all(value == 1 for value in spec.values)


def _spin_orbital_partition(spec: Charge) -> tuple[tuple[int, ...], tuple[int, ...]] | None:
    if spec.modulus is not None:
        return None
    up_orbitals: list[int] = []
    down_orbitals: list[int] = []
    for orbital, value in enumerate(spec.values):
        if value == 1:
            up_orbitals.append(orbital)
        elif value == -1:
            down_orbitals.append(orbital)
        else:
            return None
    return tuple(up_orbitals), tuple(down_orbitals)


def _spin_resolved_basis_from_counts(
    *,
    n_orbitals: int,
    up_orbitals: Sequence[int],
    down_orbitals: Sequence[int],
    n_up: int,
    n_down: int,
) -> Sequence[int]:
    if n_up < 0 or n_down < 0:
        return ()
    return determinant_basis(
        n_orbitals=n_orbitals,
        spin_up_orbs=up_orbitals,
        spin_down_orbs=down_orbitals,
        N_up=n_up,
        N_down=n_down,
    )


def _candidate_basis_for_charge_target(
    n_orbitals: int,
    specs: Mapping[str, Charge],
    target: Mapping[str, int],
) -> Sequence[int]:
    if not target:
        return determinant_basis(n_orbitals=n_orbitals)

    n_target = target.get("N")
    sz2_target = target.get("Sz2")
    unit_particle_number = n_target is not None and _is_unit_particle_charge(specs["N"])
    spin_partition = None
    if sz2_target is not None:
        spin_partition = _spin_orbital_partition(specs["Sz2"])

    if (
        unit_particle_number
        and n_target is not None
        and sz2_target is not None
        and spin_partition is not None
    ):
        if n_target < 0:
            return ()
        n_up_times_two = n_target + sz2_target
        n_down_times_two = n_target - sz2_target
        if n_up_times_two % 2 != 0 or n_down_times_two % 2 != 0:
            return ()
        up_orbitals, down_orbitals = spin_partition
        return _spin_resolved_basis_from_counts(
            n_orbitals=n_orbitals,
            up_orbitals=up_orbitals,
            down_orbitals=down_orbitals,
            n_up=n_up_times_two // 2,
            n_down=n_down_times_two // 2,
        )

    if unit_particle_number and n_target is not None:
        if n_target < 0:
            return ()
        return determinant_basis(n_orbitals=n_orbitals, N=n_target)

    if spin_partition is not None and sz2_target is not None:
        up_orbitals, down_orbitals = spin_partition
        pieces: list[int] = []
        for n_up in range(len(up_orbitals) + 1):
            n_down = n_up - sz2_target
            if 0 <= n_down <= len(down_orbitals):
                pieces.extend(
                    _spin_resolved_basis_from_counts(
                        n_orbitals=n_orbitals,
                        up_orbitals=up_orbitals,
                        down_orbitals=down_orbitals,
                        n_up=n_up,
                        n_down=n_down,
                    )
                )
        return tuple(sorted(pieces))

    return determinant_basis(n_orbitals=n_orbitals)


def basis_sector(
    n_orbitals: int,
    charges: Mapping[str, Any] | None = None,
    target: Mapping[str, int] | None = None,
    *,
    spin_z2: Sequence[int] | None = None,
) -> tuple[int, ...]:
    """Generate a determinant basis in an additive-charge sector.

    ``charges`` maps each conserved quantity name to per-orbital additive
    values.  A value may be a plain sequence, a :class:`Charge`, a
    ``(values, modulus)`` pair, or a mapping like
    ``{"values": [...], "modulus": L}``.  ``target`` selects the sector;
    only target names are filtered.  ``N`` defaults to one particle per occupied
    orbital, and the legacy ``Sz2`` sector defaults to alternating ``(+1, -1)``
    spin labels when no explicit charge vector is supplied.
    """

    checked_n = _validate_nonnegative_int(n_orbitals, "n_orbitals")
    target_map = _coerce_charge_target(target, charges=charges)
    specs = _complete_charge_specs(
        checked_n,
        charges,
        target_map,
        spin_z2=spin_z2,
        require_target=True,
    )
    candidates = _candidate_basis_for_charge_target(checked_n, specs, target_map)
    out = []
    for det in candidates:
        if _det_matches_charge_target(det, specs, target_map):
            out.append(det)
    return tuple(out)


def generate_basis(
    n_orbitals: int,
    sector: Mapping[str, int] | None = None,
    spin_z2: Sequence[int] | None = None,
    charges: Mapping[str, Any] | None = None,
) -> tuple[int, ...]:
    charge_specs: dict[str, Any] = dict(charges or {})
    if spin_z2 is not None and "Sz2" not in charge_specs and "Sz" not in charge_specs:
        charge_specs["Sz2"] = tuple(spin_z2)
    return basis_sector(
        n_orbitals,
        charges=charge_specs,
        target=sector,
        spin_z2=spin_z2,
    )


def _validate_basis(basis: Sequence[int], *, n_orbitals: int) -> tuple[int, ...]:
    checked_n = _validate_nonnegative_int(n_orbitals, "n_orbitals")
    limit = 1 << checked_n
    out = tuple(_validate_nonnegative_int(det, "basis determinant") for det in basis)
    if len(set(out)) != len(out):
        raise ValueError("basis contains duplicate determinants")
    for det in out:
        if det >= limit:
            raise ValueError(f"basis determinant {det} does not fit in n_orbitals={checked_n}")
    return out


@dataclass
class SparseOperator:
    expr: Expr
    n_orbitals: int
    sector: Mapping[str, int] = field(default_factory=dict)
    basis: Sequence[int] | None = None
    charges: Mapping[str, Any] | None = None
    index_of: dict[int, int] = field(init=False)

    def __post_init__(self) -> None:
        self.n_orbitals = _validate_nonnegative_int(self.n_orbitals, "n_orbitals")
        self.expr = self.expr.simplify()
        _require_finite_numeric(self.expr)
        sector = _coerce_charge_target(self.sector, charges=self.charges)
        self.sector = sector
        if self.basis is None:
            basis = basis_sector(self.n_orbitals, charges=self.charges, target=sector)
        else:
            basis = _validate_basis(self.basis, n_orbitals=self.n_orbitals)
        self.basis = basis
        self.index_of = {d: i for i, d in enumerate(basis)}

    @property
    def shape(self) -> tuple[int, int]:
        basis = cast(tuple[int, ...], self.basis)
        n = len(basis)
        return (n, n)

    def matvec(self, x: Sequence[Number]) -> list[float]:
        basis = cast(tuple[int, ...], self.basis)
        if len(x) != len(basis):
            raise ValueError(f"Input vector length {len(x)} does not match basis size {len(basis)}")
        y = [0.0 for _ in basis]
        for col, det in enumerate(basis):
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
        basis = cast(tuple[int, ...], self.basis)
        n = len(basis)
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
    charges: Mapping[str, Any] | None = None,
    basis: Sequence[int] | None = None,
    tensor_values: Mapping[str, Any] | None = None,
    domains: Mapping[str, Any] | None = None,
) -> Any:
    """Compile an expression into an executable sparse operator.

    ``sector`` / ``charges`` restrict the operator to an additive-charge sector
    (see :func:`basis_sector`).  Compiling into a sector *projects* onto it:
    terms that do not conserve the sector's charges connect different sectors
    and are pruned, so e.g. a number-non-conserving operator compiled into a
    fixed-``N`` sector keeps only its number-conserving block.  Pass an explicit
    ``basis=...`` to supply a precomputed determinant list instead of deriving
    it from the sector.
    """

    if target != "sparse":
        raise NotImplementedError("NOMAD executable backend is target='sparse'")
    if n_orbitals is None:
        raise ValueError("n_orbitals is required for sparse compilation")
    checked_n = _validate_nonnegative_int(n_orbitals, "n_orbitals")
    sector = _coerce_charge_target(sector, charges=charges)
    effective_charges: Mapping[str, Any] | None = charges
    if sector:
        # Complete and validate the per-orbital table for every sector charge so
        # the projection prune below is well-defined.  This runs even when an
        # explicit ``basis`` is supplied: the basis only replaces sector-derived
        # determinant enumeration, but the operator is still projected onto the
        # sector, so a named charge that lacks a table must raise here rather than
        # let pruning silently skip it.
        effective_charges = _complete_charge_specs(checked_n, charges, sector, require_target=True)

    ordered = prune_by_charge(
        normal_order(expr),
        delta_n=None,
        charges=effective_charges,
        target=sector,
    )
    lowered = expand_sums(
        ordered,
        n_orbitals=n_orbitals,
        tensor_values=tensor_values,
        domains=domains,
    )
    lowered = prune_by_charge(
        lowered,
        delta_n=None,
        charges=effective_charges,
        target=sector,
    )
    return SparseOperator(
        lowered,
        n_orbitals=checked_n,
        sector=sector,
        charges=effective_charges,
        basis=basis,
    )


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
            factors.append(f"{t.symbol.name}[{','.join(_tensor_port_text(p) for p in t.ports)}]")
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
            factors.append(
                t.symbol.name + "_{" + ",".join(_tensor_port_text(p) for p in t.ports) + "}"
            )
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
