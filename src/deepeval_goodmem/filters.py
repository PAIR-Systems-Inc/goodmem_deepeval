"""Safe construction of GoodMem metadata filter expressions.

GoodMem filters are expression strings evaluated server-side. Building one by
interpolating caller data straight into the string is an injection risk, so
these helpers quote values and refuse field names that cannot be expressed
safely.

``val()`` returns JSON, so every comparison needs an explicit cast, and the
cast must match the stored JSON type. :func:`from_mapping` takes it from the
Python type of each value:

* ``str``   -> ``CAST(val('$.category') AS TEXT) = 'feat'``
* ``bool``  -> ``CAST(val('$.flag') AS BOOLEAN) = true``
* ``int`` / ``float`` -> ``CAST(val('$.n') AS NUMERIC) = 5``

Casting and escaping were established against a live server (v1.0.320), not
assumed. A stored ``true`` compared as ``TEXT`` with ``'True'`` or
``'true'``, or compared uncast, is accepted with HTTP 200 and matches
nothing, which reads as "nothing stored"; ``AS BOOLEAN`` matches. A stored
``5`` matches ``AS NUMERIC`` ``= 5`` and ``= 5.0`` (``AS TEXT`` ``= '5.0'``
and uncast do not; ``AS NUMBER`` is an HTTP 500). The grammar escapes with a
backslash: SQL-style ``''`` doubling and double-quoted strings are both
rejected with HTTP 400, and a raw newline inside a literal is rejected
outright.
"""

from __future__ import annotations

from collections.abc import Mapping
from decimal import Decimal
import math
import re
from typing import Any

# A JSONPath member we are willing to build without escaping games.
_SAFE_FIELD = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")

# The filter grammar has no encoding for these inside a literal.
_FORBIDDEN_IN_LITERAL = re.compile(r"[\x00-\x1f\x7f]")


def _quote(value: str) -> str:
    """Single-quote a literal, backslash-escaping backslashes and quotes.

    Order matters: backslashes are escaped first so the backslash introduced
    for a quote is not escaped a second time.
    """
    if _FORBIDDEN_IN_LITERAL.search(value):
        raise ValueError(
            "Metadata filter values cannot contain control characters; the "
            "GoodMem filter grammar rejects them."
        )
    escaped = value.replace("\\", "\\\\").replace("'", "\\'")
    return f"'{escaped}'"


def _check_field(field: str) -> str:
    # fullmatch, not match with "$": "$" also matches before a trailing
    # newline, which would put the newline into the expression.
    if not isinstance(field, str) or not _SAFE_FIELD.fullmatch(field):
        raise ValueError(
            f"Unsupported metadata field name {field!r}. Use letters, digits "
            "and underscores, or pass a filter expression directly."
        )
    return field


def _number(value: float) -> str:
    # Written as plain decimals ("100000000000000000000", not repr()'s
    # "1e+20"), and NaN/infinity refused: no number literal expresses them.
    # int()/float() first, so a subclass cannot substitute its own repr.
    if isinstance(value, float):
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"Metadata filter numbers must be finite, not {value!r}.")
        return format(Decimal(repr(number)), "f")
    return str(int(value))


def _render(field: str, value: Any) -> tuple[str, str]:
    """Return ``(cast, literal)`` for ``value``, from its Python type."""
    # bool first: it is a subclass of int. A stored boolean compared as TEXT
    # 'True' (what str(True) gives) matches nothing.
    if isinstance(value, bool):
        return "BOOLEAN", "true" if value else "false"
    if isinstance(value, (int, float)):
        return "NUMERIC", _number(value)
    if isinstance(value, str):
        return "TEXT", _quote(value)
    raise ValueError(
        f"Unsupported metadata filter value {value!r} for {field!r} "
        f"({type(value).__name__}); use str, int, float or bool. Converting it "
        "to text would build a filter that silently matches nothing. For "
        "anything else, pass an expression with filter=."
    )


def text_equals(field: str, value: str) -> str:
    """Build an equality comparison against a text metadata field."""
    return f"CAST(val('$.{_check_field(field)}') AS TEXT) = {_quote(value)}"


def _equals(field: str, value: Any) -> str:
    field = _check_field(field)
    cast, literal = _render(field, value)
    return f"CAST(val('$.{field}') AS {cast}) = {literal}"


def from_mapping(metadata_filter: Mapping[str, Any]) -> str | None:
    """AND-join a mapping of field/value pairs into one filter expression.

    Each value is compared as its own type: ``str`` as ``TEXT``, ``bool`` as
    ``BOOLEAN`` and ``int``/``float`` as ``NUMERIC``, so the value's type must
    match the type stored in the metadata. ``None`` and any other type raise
    ``ValueError`` rather than being turned into text that matches nothing.
    Returns ``None`` for an empty mapping so callers can skip the filter.
    """
    clauses = [_equals(field, value) for field, value in metadata_filter.items()]
    if not clauses:
        return None
    if len(clauses) == 1:
        return clauses[0]
    return " AND ".join(f"({clause})" for clause in clauses)


def combine(*expressions: str | None) -> str | None:
    """AND-join already-built expressions, ignoring ``None``."""
    present = [e for e in expressions if e]
    if not present:
        return None
    if len(present) == 1:
        return present[0]
    return " AND ".join(f"({e})" for e in present)
