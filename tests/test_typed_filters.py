"""metadata_filter values are compared as their own type (fails against 0.2.1).

0.2.1 turned every value into ``text_equals(field, str(value))``, so
``{"flag": True}`` was sent as ``CAST(val('$.flag') AS TEXT) = 'True'``. Live
(v1.0.320) the server accepts that with HTTP 200 and matches nothing -- it
reads exactly like "nothing stored". ``AS BOOLEAN`` / ``AS NUMERIC`` match.

Each test drives ``GoodMemRetriever.search`` through the real SDK over the
mock transport and checks the expression that reached the wire.
"""

from __future__ import annotations

import ast
from pathlib import Path
import re
from typing import Any

import pytest

from goodmem_deepeval import GoodMemRetriever, filters
from tests.conftest import Recorder, ndjson_events, ndjson_response

SPACE = "01a0cfc4-59e7-719e-a126-7297caeadb45"
RETRIEVE = "/v1/memories:retrieve"
README = Path(__file__).resolve().parent.parent / "README.md"


def sent_filter(
    recorder: Recorder,
    client: Any,
    metadata_filter: dict[str, Any],
    per_call: dict[str, Any] | None = None,
) -> str:
    recorder.route(
        "POST", RETRIEVE, ndjson_response(ndjson_events("retrieve_vector.ndjson"))
    )
    GoodMemRetriever(
        space_id=SPACE, client=client, metadata_filter=metadata_filter
    ).search("canary", metadata_filter=per_call)
    return str(recorder.last_body["spaceKeys"][0]["filter"])


def refused(recorder: Recorder, client: Any, metadata_filter: Any) -> str:
    retriever = GoodMemRetriever(
        space_id=SPACE, client=client, metadata_filter=metadata_filter
    )
    with pytest.raises(ValueError) as excinfo:
        retriever.search("canary")
    assert recorder.requests == [], "nothing may be sent for a refused filter"
    return str(excinfo.value)


# ---------------------------------------------------------- the defect
@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (True, "CAST(val('$.f') AS BOOLEAN) = true"),
        (False, "CAST(val('$.f') AS BOOLEAN) = false"),
        (5, "CAST(val('$.f') AS NUMERIC) = 5"),
        (0, "CAST(val('$.f') AS NUMERIC) = 0"),
        (-3, "CAST(val('$.f') AS NUMERIC) = -3"),
        (2.5, "CAST(val('$.f') AS NUMERIC) = 2.5"),
        (5.0, "CAST(val('$.f') AS NUMERIC) = 5.0"),
        (1e20, "CAST(val('$.f') AS NUMERIC) = 100000000000000000000"),
        (1.5e-7, "CAST(val('$.f') AS NUMERIC) = 0.00000015"),
    ],
)
def test_bools_and_numbers_are_cast_to_their_own_type(
    recorder, client, value, expected
):
    assert sent_filter(recorder, client, {"f": value}) == expected


def test_a_mixed_mapping_casts_each_value_and_keeps_its_order(recorder, client):
    expression = sent_filter(recorder, client, {"flag": True, "n": 5, "category": "x"})
    assert expression == (
        "(CAST(val('$.flag') AS BOOLEAN) = true)"
        " AND (CAST(val('$.n') AS NUMERIC) = 5)"
        " AND (CAST(val('$.category') AS TEXT) = 'x')"
    )


def test_a_per_call_filter_is_typed_and_anded_with_the_constructors(recorder, client):
    expression = sent_filter(
        recorder, client, {"category": "x"}, per_call={"archived": False}
    )
    assert expression == (
        "(CAST(val('$.category') AS TEXT) = 'x')"
        " AND (CAST(val('$.archived') AS BOOLEAN) = false)"
    )


def test_none_is_refused_before_a_request(recorder, client):
    """0.2.1 sent CAST(val('$.flag') AS TEXT) = 'None' and matched nothing."""
    message = refused(recorder, client, {"flag": None})
    assert "Unsupported metadata filter value None for 'flag'" in message
    assert "filter=" in message


@pytest.mark.parametrize(
    "value",
    [[1], ("a",), {"a": 1}, b"bytes", object()],
    ids=["list", "tuple", "dict", "bytes", "object"],
)
def test_unsupported_types_are_refused_not_stringified(recorder, client, value):
    message = refused(recorder, client, {"f": value})
    assert "use str, int, float or bool" in message


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_numbers_are_refused(recorder, client, value):
    assert "finite" in refused(recorder, client, {"f": value})


def test_a_field_name_with_a_trailing_newline_is_refused(recorder, client):
    """``^...$`` with re.match accepted "category\\n"."""
    assert "Unsupported metadata field name" in refused(
        recorder, client, {"category\n": "x"}
    )


def test_a_float_subclass_cannot_change_the_literal(recorder, client):
    class Sneaky(float):
        def __repr__(self) -> str:
            return "1) OR (1=1"

    expression = sent_filter(recorder, client, {"f": Sneaky(2.5)})
    assert expression == "CAST(val('$.f') AS NUMERIC) = 2.5"


# ------------------------------------------------------------ controls
@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("x", "CAST(val('$.f') AS TEXT) = 'x'"),
        ("O'Brien", "CAST(val('$.f') AS TEXT) = 'O\\'Brien'"),
        ("a\\b", "CAST(val('$.f') AS TEXT) = 'a\\\\b'"),
        ("x' OR '1'='1", "CAST(val('$.f') AS TEXT) = 'x\\' OR \\'1\\'=\\'1'"),
        ("5", "CAST(val('$.f') AS TEXT) = '5'"),
        ("true", "CAST(val('$.f') AS TEXT) = 'true'"),
    ],
)
def test_strings_are_unchanged(recorder, client, value, expected):
    assert sent_filter(recorder, client, {"f": value}) == expected
    assert filters.text_equals("f", value) == expected


def test_control_characters_in_strings_are_still_refused(recorder, client):
    assert "control characters" in refused(recorder, client, {"f": "a\nb"})


@pytest.mark.parametrize("field", ["", "1st", "a b", "x') OR ('1'='1", "a.b"])
def test_unsafe_field_names_are_still_refused(recorder, client, field):
    assert "Unsupported metadata field name" in refused(recorder, client, {field: 1})


def test_an_empty_mapping_is_still_no_filter(recorder, client):
    recorder.route(
        "POST", RETRIEVE, ndjson_response(ndjson_events("retrieve_vector.ndjson"))
    )
    GoodMemRetriever(space_id=SPACE, client=client, metadata_filter={}).search("q")
    assert filters.from_mapping({}) is None
    assert recorder.last_body["spaceKeys"] == [{"spaceId": SPACE}]


# ---------------------------------------------------------------- README
def test_the_readme_filter_table_is_what_the_builder_sends():
    """Every `{mapping}` | `expression` row in the README's filter table."""
    rows = re.findall(
        r"^\| `(\{.*?\})` \| `(CAST.*?)` \|", README.read_text(), re.MULTILINE
    )
    assert len(rows) >= 3, "the README's filter table went missing"
    for mapping, expected in rows:
        assert filters.from_mapping(ast.literal_eval(mapping)) == expected
