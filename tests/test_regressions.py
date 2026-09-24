"""Every test here fails against deepeval-goodmem 0.1.0.

The fixtures are bytes a live server (v1.0.320) actually sent.
"""

from __future__ import annotations

import json
from typing import Any

from deepeval.test_case import LLMTestCase
import httpx
import pytest

from deepeval_goodmem import (
    GoodMemRetrievalHealthMetric,
    GoodMemRetriever,
    GoodMemSpaceError,
    filters,
)
from tests.conftest import Recorder, load_json, ndjson_events, ndjson_response

SPACE = "01a0cfc4-59e7-719e-a126-7297caeadb45"
RETRIEVE = "/v1/memories:retrieve"


def make(recorder: Recorder, client: Any, fixture: str, **kwargs: Any) -> GoodMemRetriever:
    recorder.route("POST", RETRIEVE, ndjson_response(ndjson_events(fixture)))
    return GoodMemRetriever(space_id=SPACE, client=client, **kwargs)


# --------------------------------------------------------------- statuses
def test_a_broken_reranker_is_reported_not_swallowed(recorder, client):
    """0.1.0's NDJSON parser had no `status` branch at all. Live, a bogus
    reranker makes the server send three status events and still return
    fallback chunks; 0.1.0 reported success:true with no mention of them.
    """
    raw = ndjson_events("retrieve_broken_reranker.ndjson")
    assert sum("status" in e for e in raw) == 3, "fixture must carry the statuses"

    r = make(
        recorder,
        client,
        "retrieve_broken_reranker.ndjson",
        reranker_id="00000000-0000-0000-0000-000000000000",
    )
    out = r.search("canary")

    # Contract Q4a: the chunks the server returned are kept, and flagged.
    assert len(out["hits"]) == 3
    assert out["partial"] is True
    codes = [s["code"] for s in out["statuses"]]
    assert "NOT_FOUND" in codes and "RERANKING_FAILED" in codes


def test_feature_disabled_is_informational_whatever_its_details(recorder, client):
    """Contract Q1: the code alone decides."""
    events = ndjson_events("retrieve_vector.ndjson")
    events.insert(
        0,
        {
            "status": {
                "code": "FEATURE_DISABLED",
                "message": "Reranking disabled: no reranker configured.",
                "details": {"feature": "reranking", "required_param": "reranker_id"},
            }
        },
    )
    recorder.route("POST", RETRIEVE, ndjson_response(events))
    out = GoodMemRetriever(space_id=SPACE, client=client).search("canary")

    assert out["statuses"] == []
    assert out["partial"] is False
    assert len(out["hits"]) == 3


def test_an_unknown_status_code_is_surfaced_not_dropped(recorder, client):
    """Contract Q3: the SDK decodes an unrecognised code as None."""
    events = ndjson_events("retrieve_vector.ndjson")
    events.insert(0, {"status": {"code": "A_CODE_FROM_THE_FUTURE", "message": "hi"}})
    recorder.route("POST", RETRIEVE, ndjson_response(events))
    out = GoodMemRetriever(space_id=SPACE, client=client).search("q")

    assert out["partial"] is True
    assert out["statuses"][0]["code"] == "UNKNOWN"
    assert out["statuses"][0]["unrecognized"] is True
    assert len(out["hits"]) == 3, "an unknown code must never discard chunks"


def test_a_failed_retrieval_is_empty_flagged_and_warned_not_raised(recorder, client):
    """Contract Q4b. 0.1.0 returned success:true, totalResults:0 and a message
    blaming indexing after 60 seconds of polling."""
    recorder.route(
        "POST",
        RETRIEVE,
        ndjson_response([{"status": {"code": "NOT_FOUND", "message": "space gone"}}]),
    )
    with pytest.warns(UserWarning, match="NOT_FOUND"):
        out = GoodMemRetriever(space_id=SPACE, client=client).search("q")
    assert out["hits"] == []
    assert out["partial"] is True
    assert [s["code"] for s in out["statuses"]] == ["NOT_FOUND"]


def test_a_genuinely_empty_result_is_quiet(recorder, client):
    r = make(recorder, client, "retrieve_empty.ndjson")
    out = r.search("xyzzy")
    assert out["hits"] == []
    assert out["partial"] is False
    assert out["statuses"] == []


def test_retrieval_makes_exactly_one_request(recorder, client):
    """0.1.0 polled up to 12 times by default (60s / 5s), re-invoking the LLM
    each time when llm_id was set. Measured live at 63.8s on an empty space."""
    r = make(recorder, client, "retrieve_empty.ndjson")
    r.search("anything")
    assert len(recorder.requests) == 1


# ----------------------------------------------------------------- scores
def test_vector_scores_are_negative_and_server_order_is_kept(recorder, client):
    r = make(recorder, client, "retrieve_vector.ndjson")
    out = r.search("canary")
    scores = [h["score"] for h in out["hits"]]

    assert scores[0] < 0
    assert scores != sorted(scores, reverse=True)
    assert "OTTER-52-BEACON" in out["hits"][0]["chunk_text"]
    assert out["score_kind"] == "vector"


def test_reranker_scores_are_a_different_scale(recorder, client):
    r = make(recorder, client, "retrieve_reranked.ndjson", reranker_id="rr")
    out = r.search("canary")
    assert out["score_kind"] == "reranker"
    assert [h["score"] for h in out["hits"]] == sorted(
        [h["score"] for h in out["hits"]], reverse=True
    )


def test_min_score_is_ignored_without_a_reranker(recorder, client):
    """0.1.0 forwarded relevance_threshold regardless, documented as 0-1."""
    r = make(recorder, client, "retrieve_vector.ndjson", min_score=0.5)
    assert len(r.search("canary")["hits"]) == 3


def test_no_relevance_threshold_is_ever_sent(recorder, client):
    r = make(recorder, client, "retrieve_reranked.ndjson", reranker_id="rr", min_score=0.0)
    r.search("canary")
    assert "relevance_threshold" not in recorder.last_body["postProcessor"]["config"]


# ------------------------------------------------------------- deepeval
def test_search_is_a_deepeval_retriever_span():
    from deepeval.tracing import observe  # noqa: F401

    assert hasattr(GoodMemRetriever.search, "__wrapped__"), "search must be @observe'd"


def test_retrieve_returns_the_shape_retrieval_context_wants(recorder, client):
    r = make(recorder, client, "retrieve_vector.ndjson")
    context = r.retrieve("canary")
    assert isinstance(context, list)
    assert all(isinstance(c, str) and c for c in context)
    assert any("OTTER-52-BEACON" in c for c in context)


def test_to_test_case_populates_retrieval_context_and_diagnostics(recorder, client):
    r = make(recorder, client, "retrieve_vector.ndjson")
    result = r.search("what is the canary")
    case = r.to_test_case(result, actual_output="OTTER-52-BEACON", expected_output="OTTER-52-BEACON")

    assert isinstance(case, LLMTestCase)
    assert case.input == "what is the canary"
    assert len(case.retrieval_context) == 3
    diagnostics = case.metadata["goodmem"]
    assert diagnostics["partial"] is False
    assert diagnostics["score_kind"] == "vector"
    assert len(diagnostics["chunk_ids"]) == 3


def test_to_test_case_carries_a_degraded_retrieval_into_metadata(recorder, client):
    r = make(
        recorder, client, "retrieve_broken_reranker.ndjson", reranker_id="bogus"
    )
    case = r.to_test_case(r.search("canary"), actual_output="whatever")
    assert case.metadata["goodmem"]["partial"] is True
    assert any(
        s["code"] == "RERANKING_FAILED" for s in case.metadata["goodmem"]["statuses"]
    )


def test_to_test_case_keeps_caller_metadata(recorder, client):
    r = make(recorder, client, "retrieve_vector.ndjson")
    case = r.to_test_case(
        r.search("q"), actual_output="a", metadata={"run": "nightly"}
    )
    assert case.metadata["run"] == "nightly"
    assert "goodmem" in case.metadata


# ------------------------------------------------------------- the metric
def _case(partial: bool, statuses: list[dict[str, Any]], chunks: int = 2) -> LLMTestCase:
    return LLMTestCase(
        input="q",
        actual_output="a",
        retrieval_context=[f"chunk {i}" for i in range(chunks)],
        metadata={"goodmem": {"partial": partial, "statuses": statuses, "score_kind": "vector"}},
    )


def test_health_metric_passes_a_clean_retrieval():
    metric = GoodMemRetrievalHealthMetric()
    assert metric.measure(_case(False, [])) == 1.0
    assert metric.is_successful() is True
    assert "no problems" in metric.reason


def test_health_metric_fails_a_degraded_retrieval():
    metric = GoodMemRetrievalHealthMetric()
    assert metric.measure(_case(True, [{"code": "RERANKING_FAILED"}])) == 0.0
    assert metric.is_successful() is False
    assert "RERANKING_FAILED" in metric.reason
    assert metric.score_breakdown["status_codes"] == ["RERANKING_FAILED"]


def test_health_metric_skips_a_case_it_knows_nothing_about():
    """A non-GoodMem case must not be failed by a metric that cannot judge it."""
    metric = GoodMemRetrievalHealthMetric()
    metric.measure(LLMTestCase(input="q", actual_output="a", retrieval_context=["c"]))
    assert metric.skipped is True
    assert metric.success is None


def test_health_metric_can_require_hits():
    metric = GoodMemRetrievalHealthMetric(require_hits=True)
    assert metric.measure(_case(False, [], chunks=0)) == 0.0
    assert "no chunks" in metric.reason


def test_health_metric_a_measure_matches_measure():
    import asyncio

    metric = GoodMemRetrievalHealthMetric()
    degraded = _case(True, [{"code": "NOT_FOUND"}])
    assert asyncio.run(metric.a_measure(degraded)) == 0.0
    assert metric.is_successful() is False


# ---------------------------------------------------------------- filters
def test_metadata_filter_is_escaped_not_interpolated(recorder, client):
    """0.1.0 took metadata_filter as a raw string. Live, `x' OR '1'='1`
    subverted the filter and returned a row it should not have."""
    r = make(
        recorder,
        client,
        "retrieve_vector.ndjson",
        metadata_filter={"category": "x' OR '1'='1"},
    )
    r.search("canary")
    expression = recorder.last_body["spaceKeys"][0]["filter"]
    assert expression == "CAST(val('$.category') AS TEXT) = 'x\\' OR \\'1\\'=\\'1'"


def test_filter_refuses_control_characters():
    with pytest.raises(ValueError, match="control characters"):
        filters.text_equals("category", "a\nb")


# ----------------------------------------------------------------- spaces
def _space(name: str, space_id: str, embedder: str) -> dict[str, Any]:
    space = json.loads(json.dumps(load_json("spaces_list.json")["spaces"][0]))
    space["spaceId"] = space_id
    space["name"] = name
    for se in space["spaceEmbedders"]:
        se["spaceId"] = space_id
        se["embedderId"] = embedder
    return space


def _page(items: list[dict[str, Any]]) -> httpx.Response:
    return httpx.Response(200, json={"spaces": items, "nextToken": None})


def test_attaching_by_name_refuses_a_different_embedder(recorder, client):
    recorder.route("GET", "/v1/spaces", _page([_space("docs", SPACE, "emb-1")]))
    r = GoodMemRetriever(space_name="docs", embedder_id="emb-2", client=client)
    with pytest.raises(GoodMemSpaceError, match="not emb-2"):
        r.search("q")


def test_attaching_by_name_reuses_a_matching_space(recorder, client):
    recorder.route("GET", "/v1/spaces", _page([_space("docs", SPACE, "emb-1")]))
    recorder.route("POST", RETRIEVE, ndjson_response(ndjson_events("retrieve_vector.ndjson")))
    r = GoodMemRetriever(space_name="docs", embedder_id="emb-1", client=client)
    assert r.search("q")["space_ids"] == [SPACE]


# ------------------------------------------------------------- validation
def test_credentials_are_not_model_fields():
    """0.1.0 kept the key as a plain attribute on every operation object."""
    r = GoodMemRetriever(space_id=SPACE, base_url="https://x", api_key="gm_SECRET")
    assert "api_key" not in type(r).model_fields
    assert "gm_SECRET" not in json.dumps(r.model_dump(), default=str)
    assert "gm_SECRET" not in repr(r)


def test_a_retriever_needs_somewhere_to_search():
    with pytest.raises(ValueError, match="space_id, space_ids or space_name"):
        GoodMemRetriever()


def test_an_empty_query_is_rejected_before_a_request(recorder, client):
    r = GoodMemRetriever(space_id=SPACE, client=client)
    with pytest.raises(ValueError, match="query cannot be empty"):
        r.search("  ")
    assert recorder.requests == []


def test_errors_propagate_instead_of_becoming_a_json_string(recorder, client):
    """0.1.0's run() caught everything and returned
    {"success": false, "error": "Client error '400' ... check MDN"} as a str,
    discarding the server's actual reason."""
    recorder.route("POST", RETRIEVE, httpx.Response(400, json={
        "errors": [{"field": "spaceKeys[0].spaceId", "message": "Invalid space ID format"}]
    }))
    r = GoodMemRetriever(space_id="not-a-uuid", client=client)
    with pytest.raises(Exception) as excinfo:
        r.search("q")
    assert "Invalid space ID format" in str(excinfo.value)


def test_fixtures_are_real_server_bytes():
    assert load_json("space.json")["spaceId"] == SPACE
    codes = [
        e["status"]["code"]
        for e in ndjson_events("retrieve_broken_reranker.ndjson")
        if "status" in e
    ]
    assert "RERANKING_FAILED" in codes


# ------------------------------------------------ coverage gaps (2026-09-24)
def test_an_ambiguous_space_name_is_an_error(recorder, client):
    """GoodMem does not require space names to be unique."""
    recorder.route(
        "GET", "/v1/spaces",
        _page([_space("docs", SPACE, "e1"), _space("docs", "other-id", "e1")]),
    )
    r = GoodMemRetriever(space_name="docs", client=client)
    with pytest.raises(GoodMemSpaceError, match="2 spaces are named"):
        r.search("q")


def test_a_missing_space_is_not_created_unless_asked(recorder, client):
    recorder.route("GET", "/v1/spaces", _page([]))
    r = GoodMemRetriever(space_name="nope", client=client)
    with pytest.raises(GoodMemSpaceError, match="create_space=True"):
        r.search("q")
    assert all(req.method == "GET" for req in recorder.requests), "no create was sent"


def test_space_ids_searches_every_space_in_order(recorder, client):
    r = make(recorder, client, "retrieve_vector.ndjson", space_ids=["b-space"])
    out = r.search("q")
    assert recorder.last_body["spaceKeys"] == [{"spaceId": SPACE}, {"spaceId": "b-space"}]
    assert out["space_ids"] == [SPACE, "b-space"]


def test_an_injected_client_is_not_closed_by_the_retriever(recorder, client):
    """The connection pool and TLS settings of an injected client are the
    caller's; a second search on the same client must still work."""
    recorder.route("POST", RETRIEVE, ndjson_response(ndjson_events("retrieve_vector.ndjson")))
    r = GoodMemRetriever(space_id=SPACE, client=client)
    r.search("one")
    r.search("two")
    assert len(recorder.requests) == 2


def test_a_min_score_that_removes_every_reranked_hit_says_so(recorder, client):
    """Measured live 2026-09-24: Voyage rerank-2.5 0.27..0.93, Jina
    jina-reranker-v3 -0.14..0.43 on the same documents. A threshold tuned
    for one empties the other; an empty retrieval_context must not look
    like a miss."""
    r = make(recorder, client, "retrieve_reranked.ndjson", reranker_id="rr", min_score=0.9)
    with pytest.warns(UserWarning, match="removed all 3 reranked hit"):
        out = r.search("canary")
    assert out["hits"] == []
    assert out["partial"] is False, "the threshold, not the server, emptied it"
