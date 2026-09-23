"""Live end-to-end tests. Skipped unless a server is configured.

    GOODMEM_BASE_URL=https://localhost:8080 \
    GOODMEM_API_KEY=gm_… \
    GOODMEM_EMBEDDER_ID=… \
    GOODMEM_VERIFY_SSL=0 \
    pytest -m integration

There is no default credential. Everything created here is deleted in the
fixture teardown, and the teardown asserts it is gone.
"""

from __future__ import annotations

import os
import time
from typing import Any
import uuid

from deepeval.test_case import LLMTestCase
from goodmem import Goodmem
import pytest

from deepeval_goodmem import (
    GoodMemRetrievalHealthMetric,
    GoodMemRetriever,
    GoodMemSpaceError,
)
from deepeval_goodmem._spaces import find_by_name

pytestmark = pytest.mark.integration

BASE_URL = os.getenv("GOODMEM_BASE_URL")
API_KEY = os.getenv("GOODMEM_API_KEY")
EMBEDDER_ID = os.getenv("GOODMEM_EMBEDDER_ID")
RERANKER_ID = os.getenv("GOODMEM_RERANKER_ID")
VERIFY_SSL = os.getenv("GOODMEM_VERIFY_SSL", "1") not in ("0", "false", "False")

if not (BASE_URL and API_KEY and EMBEDDER_ID):
    pytest.skip(
        "set GOODMEM_BASE_URL, GOODMEM_API_KEY and GOODMEM_EMBEDDER_ID",
        allow_module_level=True,
    )

CANARY = "The DeepEval live-test canary is OTTER-52-BEACON."
FACTS = [
    (CANARY, {"category": "canary"}),
    ("DeepEval reads LLMTestCase.retrieval_context for its RAG metrics.", {"category": "deepeval"}),
    ("A GoodMem vector score is a negative inner product.", {"category": "scores"}),
]


@pytest.fixture(scope="module")
def live() -> Any:
    client = Goodmem(base_url=BASE_URL, api_key=API_KEY, verify=VERIFY_SSL)
    name = f"deepeval-goodmem-e2e-{uuid.uuid4().hex[:8]}"
    space = client.spaces.create(
        name=name,
        space_embedders=[{"embedderId": EMBEDDER_ID, "defaultRetrievalWeight": 1.0}],
        default_chunking_config={
            "recursive": {
                "chunkSize": 256,
                "chunkOverlap": 25,
                "separators": ["\n\n", "\n", ". ", " ", ""],
                "keepStrategy": "KEEP_END",
                "separatorIsRegex": False,
                "lengthMeasurement": "CHARACTER_COUNT",
            }
        },
    )
    memory_ids = []
    for text, metadata in FACTS:
        memory = client.memories.create(
            space_id=space.space_id,
            content_type="text/plain",
            original_content=text,
            metadata=metadata,
        )
        memory_ids.append(memory.memory_id)

    deadline = time.time() + 180
    while time.time() < deadline:
        states = [client.memories.get(id=m).processing_status for m in memory_ids]
        if all(s == "COMPLETED" for s in states):
            break
        if any(s == "FAILED" for s in states):
            pytest.fail(f"ingestion failed: {states}")
        time.sleep(2)
    else:
        pytest.fail("memories did not finish indexing within 180s")

    yield {
        "client": client,
        "space_id": space.space_id,
        "space_name": name,
        "canary_id": memory_ids[0],
    }

    client.spaces.delete(id=space.space_id)
    assert find_by_name(client, name) == [], "teardown left the space behind"
    client.close()


@pytest.fixture
def retriever(live: Any) -> GoodMemRetriever:
    return GoodMemRetriever(space_id=live["space_id"], client=live["client"], limit=5)


def test_exact_fact_is_recalled(retriever: GoodMemRetriever) -> None:
    out = retriever.search("what is the live-test canary")
    assert any("OTTER-52-BEACON" in h["chunk_text"] for h in out["hits"])
    assert out["score_kind"] == "vector"
    assert out["partial"] is False


def test_vector_scores_come_back_as_the_server_sent_them(retriever) -> None:
    scores = [h["score"] for h in retriever.search("canary")["hits"]]
    assert scores, "expected hits"
    assert any(s < 0 for s in scores), "0.1.0 documented these as 0-1"


def test_retrieve_gives_the_shape_retrieval_context_wants(retriever) -> None:
    context = retriever.retrieve("what is the live-test canary")
    assert context and all(isinstance(c, str) and c for c in context)
    assert any("OTTER-52-BEACON" in c for c in context)


def test_a_broken_reranker_is_flagged_and_keeps_its_fallback_chunks(live) -> None:
    """The R1 reproduction: 0.1.0 reported success:true and never mentioned
    the two statuses the server sent."""
    r = GoodMemRetriever(
        space_id=live["space_id"],
        client=live["client"],
        reranker_id="00000000-0000-0000-0000-000000000000",
    )
    out = r.search("canary")
    assert out["hits"], "the server's fallback chunks must be kept (contract Q4a)"
    assert out["partial"] is True
    codes = {s["code"] for s in out["statuses"]}
    assert "RERANKING_FAILED" in codes


def test_a_working_reranker_is_not_reported_as_degraded(live) -> None:
    if not RERANKER_ID:
        pytest.skip("set GOODMEM_RERANKER_ID")
    r = GoodMemRetriever(
        space_id=live["space_id"], client=live["client"], reranker_id=RERANKER_ID
    )
    out = r.search("canary")
    assert out["hits"]
    assert out["statuses"] == [], "FEATURE_DISABLED is informational"
    assert out["partial"] is False
    assert out["score_kind"] == "reranker"


def test_empty_search_returns_immediately(retriever) -> None:
    """0.1.0 polled for 63.8s here and then blamed indexing."""
    started = time.time()
    out = retriever.search("canary", metadata_filter={"category": "no-such-category"})
    elapsed = time.time() - started
    assert out["hits"] == []
    assert out["partial"] is False
    assert elapsed < 5, f"took {elapsed:.1f}s; 0.1.0 took 63.8s"


def test_a_quote_in_a_filter_value_is_escaped_not_injected(retriever) -> None:
    """Live, 0.1.0's raw metadata_filter let `x' OR '1'='1` return a row."""
    out = retriever.search("canary", metadata_filter={"category": "x' OR '1'='1"})
    assert out["hits"] == [], "the value must be a literal, matching nothing"


def test_metadata_filter_scopes_the_search(retriever) -> None:
    out = retriever.search("anything", metadata_filter={"category": "scores"})
    assert out["hits"]
    assert all(h["metadata"].get("category") == "scores" for h in out["hits"])


def test_attaching_by_name_refuses_a_different_embedder(live) -> None:
    r = GoodMemRetriever(
        space_name=live["space_name"],
        embedder_id="00000000-0000-0000-0000-000000000000",
        client=live["client"],
    )
    with pytest.raises(GoodMemSpaceError, match="Retrieval across mismatched"):
        r.search("canary")


def test_test_case_and_health_metric_over_a_real_retrieval(live, retriever) -> None:
    """The whole point of the package: a live retrieval becomes an
    LLMTestCase that DeepEval's RAG metrics can consume."""
    result = retriever.search("what is the live-test canary")
    case = retriever.to_test_case(
        result,
        actual_output="The canary is OTTER-52-BEACON.",
        expected_output="OTTER-52-BEACON",
    )

    assert isinstance(case, LLMTestCase)
    assert case.retrieval_context and any(
        "OTTER-52-BEACON" in c for c in case.retrieval_context
    )
    assert live["canary_id"] in case.metadata["goodmem"]["memory_ids"]

    metric = GoodMemRetrievalHealthMetric()
    assert metric.measure(case) == 1.0
    assert metric.is_successful() is True


def test_health_metric_fails_a_live_degraded_retrieval(live) -> None:
    r = GoodMemRetriever(
        space_id=live["space_id"],
        client=live["client"],
        reranker_id="00000000-0000-0000-0000-000000000000",
    )
    case = r.to_test_case(r.search("canary"), actual_output="whatever")
    metric = GoodMemRetrievalHealthMetric()
    assert metric.measure(case) == 0.0
    assert metric.is_successful() is False
    assert "RERANKING_FAILED" in metric.reason
