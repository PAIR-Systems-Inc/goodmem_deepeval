"""A failed reranker's fallback hits are vector hits (fails against 0.2.1).

With a reranker configured and the reranker failing, GoodMem v1.0.320 sends
``NOT_FOUND`` (naming the reranker) and ``RERANKING_FAILED`` and still returns
the vector-stage hits, scored as negative vector distances. 0.2.1 labelled
them ``"reranker"`` from configuration and applied ``min_score`` to them:
live, ``min_score=0.0`` returned 0 of the 3 hits the server sent.

Every stream here is the captured ``retrieve_broken_reranker.ndjson`` or
``retrieve_reranked.ndjson``, reordered or with one event added or removed,
decoded by the real SDK over the mock transport.
"""

from __future__ import annotations

from typing import Any
import warnings

import pytest

from goodmem_deepeval import GoodMemRetrievalHealthMetric, GoodMemRetriever
from tests.conftest import Recorder, ndjson_events, ndjson_response

SPACE = "01a0cfc4-59e7-719e-a126-7297caeadb45"
RETRIEVE = "/v1/memories:retrieve"
MISSING = "00000000-0000-0000-0000-000000000000"


def _code(event: dict[str, Any]) -> str | None:
    return event.get("status", {}).get("code")


def broken() -> list[dict[str, Any]]:
    return ndjson_events("retrieve_broken_reranker.ndjson")


def search(
    recorder: Recorder,
    client: Any,
    events: list[dict[str, Any]],
    *,
    reranker_id: str = MISSING,
    **kwargs: Any,
) -> tuple[dict[str, Any], list[str]]:
    recorder.route("POST", RETRIEVE, ndjson_response(events))
    retriever = GoodMemRetriever(
        space_id=SPACE, client=client, reranker_id=reranker_id, **kwargs
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = retriever.search("canary")
    return out, [str(w.message) for w in caught]


def raw_scores(events: list[dict[str, Any]]) -> list[float]:
    return [
        e["retrievedItem"]["chunk"]["relevanceScore"]
        for e in events
        if "retrievedItem" in e
    ]


# ---------------------------------------------------------- the defect
def test_fallback_hits_are_vector_scored_as_the_server_sent_them(recorder, client):
    events = broken()
    out, _ = search(recorder, client, events)

    assert out["score_kind"] == "vector"
    assert [h["score_kind"] for h in out["hits"]] == ["vector"] * 3
    # Raw, in the server's order: negative vector distances.
    assert [h["score"] for h in out["hits"]] == raw_scores(events)
    assert all(h["score"] < 0 for h in out["hits"])
    # Contract Q4a: the hits are kept and the retrieval is flagged.
    assert out["partial"] is True
    assert {s["code"] for s in out["statuses"]} == {"NOT_FOUND", "RERANKING_FAILED"}
    # The reranker was still requested; only the response decides the kind.
    assert recorder.last_body["postProcessor"]["config"]["reranker_id"] == MISSING


@pytest.mark.parametrize("min_score", [0.0, 0.9, -0.1])
def test_min_score_does_not_discard_fallback_hits(recorder, client, min_score):
    """Measured live on 0.2.1: min_score=0.0 returned 0 of the 3 hits the
    server sent, with a warning blaming the reranker's score range."""
    out, messages = search(recorder, client, broken(), min_score=min_score)

    assert len(out["hits"]) == 3
    assert out["partial"] is True
    assert not [m for m in messages if "min_score" in m]
    assert not [m for m in messages if "returned nothing" in m]


def test_reranking_failed_after_the_hits_still_counts(recorder, client):
    """Decided once the whole stream is in, not when the first hit arrives."""
    events = broken()
    statuses = [e for e in events if "status" in e]
    reordered = [e for e in events if "status" not in e] + statuses

    out, _ = search(recorder, client, reordered, min_score=0.0)
    assert out["score_kind"] == "vector"
    assert len(out["hits"]) == 3
    assert out["partial"] is True


@pytest.mark.parametrize(
    "status",
    [
        # As captured: details name the reranker.
        None,
        {
            "code": "NOT_FOUND",
            "message": "Not found",
            "details": {"rerankerId": MISSING},
        },
        {"code": "NOT_FOUND", "message": f"Reranker not found: {MISSING}"},
    ],
    ids=["reranker_id-in-details", "rerankerId-in-details", "reranker-in-message"],
)
def test_a_not_found_naming_the_reranker_alone_means_not_reranked(
    recorder, client, status
):
    events = [e for e in broken() if _code(e) != "RERANKING_FAILED"]
    if status is not None:
        events = [e for e in events if _code(e) != "NOT_FOUND"]
        events.insert(0, {"status": status})
    assert "RERANKING_FAILED" not in {_code(e) for e in events}

    out, messages = search(recorder, client, events, min_score=0.0)
    assert out["score_kind"] == "vector"
    assert len(out["hits"]) == 3
    assert out["partial"] is True
    assert not [m for m in messages if "min_score" in m]


def test_the_degraded_retrieval_reaches_the_test_case_as_vector(recorder, client):
    out, _ = search(recorder, client, broken(), min_score=0.0)
    retriever = GoodMemRetriever(space_id=SPACE, client=client)
    case = retriever.to_test_case(out, actual_output="whatever")

    assert len(case.retrieval_context) == 3
    assert case.metadata["goodmem"]["score_kind"] == "vector"
    assert case.metadata["goodmem"]["partial"] is True
    metric = GoodMemRetrievalHealthMetric()
    assert metric.measure(case) == 0.0
    assert "RERANKING_FAILED" in metric.reason


# ------------------------------------------------------------ controls
def test_a_working_reranker_is_unchanged(recorder, client):
    events = ndjson_events("retrieve_reranked.ndjson")
    out, _ = search(recorder, client, events, reranker_id="rr")
    assert out["score_kind"] == "reranker"
    assert [h["score_kind"] for h in out["hits"]] == ["reranker"] * 3
    assert [h["score"] for h in out["hits"]] == raw_scores(events)
    assert out["partial"] is False

    out, _ = search(recorder, client, events, reranker_id="rr", min_score=0.0)
    assert [h["score"] for h in out["hits"]] == [
        s for s in raw_scores(events) if s >= 0.0
    ]


@pytest.mark.parametrize(
    "status",
    [
        {"code": "A_CODE_FROM_THE_FUTURE", "message": "reranker says hi"},
        {
            "code": "NOT_FOUND",
            "message": "Space not found",
            "details": {"space_id": SPACE},
        },
        {"code": "SUMMARIZATION_FAILED", "message": "llm down"},
    ],
    ids=["unknown-code", "not-found-about-a-space", "summarization-failed"],
)
def test_an_unrelated_status_keeps_reranker_scores(recorder, client, status):
    events = ndjson_events("retrieve_reranked.ndjson")
    events.append({"status": status})

    out, _ = search(recorder, client, events, reranker_id="rr", min_score=0.0)
    assert out["score_kind"] == "reranker"
    assert [h["score_kind"] for h in out["hits"]] == ["reranker"]
    assert out["partial"] is True
    # The threshold still applies: these are reranker scores.
    assert [h["score"] for h in out["hits"]] == [
        s for s in raw_scores(events) if s >= 0.0
    ]


def test_a_threshold_that_empties_genuinely_reranked_hits_still_warns(recorder, client):
    out, messages = search(
        recorder,
        client,
        ndjson_events("retrieve_reranked.ndjson"),
        reranker_id="rr",
        min_score=0.9,
    )
    assert out["hits"] == []
    assert any("removed all 3 reranked hit(s)" in m for m in messages)


def test_a_failed_reranker_with_no_hits_is_empty_flagged_and_warned(recorder, client):
    """Contract Q4b is unchanged: problem and no hits is empty + flag. Only
    ``score_kind`` changes: nothing was reranked, so it is ``"vector"``."""
    events = [e for e in broken() if "status" in e]
    out, messages = search(recorder, client, events, min_score=0.0)
    assert out["hits"] == []
    assert out["partial"] is True
    assert out["score_kind"] == "vector"
    assert any("returned nothing" in m for m in messages)
    assert not [m for m in messages if "min_score" in m]
