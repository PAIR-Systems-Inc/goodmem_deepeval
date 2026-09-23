"""GoodMem retrieval, traced as a DeepEval retriever span."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any
import warnings

from deepeval.test_case import LLMTestCase
from deepeval.tracing import observe, update_retriever_span
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from deepeval_goodmem._connection import GoodMemConnection, split_connection_kwargs
from deepeval_goodmem._results import abstract_reply, classify, hits_from_events
from deepeval_goodmem._spaces import GoodMemSpaceError, resolve
from deepeval_goodmem.filters import combine, from_mapping

if TYPE_CHECKING:
    from collections.abc import Sequence


logger = logging.getLogger(__name__)


class GoodMemRetriever(BaseModel):
    """Retrieve from GoodMem spaces inside a DeepEval trace.

    :meth:`search` is decorated with ``@observe(type="retriever")``, so each
    call becomes a retriever span carrying the query, the latency, ``top_k``
    and the embedder, alongside every hit's score, the kind of score it is,
    and any degradation the server reported::

        from deepeval import evaluate
        from deepeval.metrics import ContextualRelevancyMetric
        from deepeval_goodmem import GoodMemRetriever

        retriever = GoodMemRetriever(space_name="docs", limit=5)
        result = retriever.search("how do I rotate an API key?")

        case = retriever.to_test_case(result, actual_output=answer)
        evaluate([case], [ContextualRelevancyMetric()])

    Credentials are constructor keywords, not fields: a retriever is routinely
    rendered into a traceback, a notebook cell and a trace payload, so the
    connection lives on a private attribute and never appears in
    ``model_dump()`` or ``repr()``.

    Scores are reported exactly as the server sends them, in the server's
    order. A GoodMem vector score is a negative inner product -- the best
    match is the *lowest* number -- and a reranker score is a different scale
    that also goes negative, so they are never mixed, re-sorted or compared.
    ``score_kind`` on each hit says which one you have, and ``min_score`` is
    honoured only with a reranker configured.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    space_id: str | None = None
    space_ids: list[str] = Field(default_factory=list)
    space_name: str | None = Field(
        default=None,
        description=(
            "Attach by name. An existing space is reused when its embedder "
            "matches embedder_id; a different embedder is an error."
        ),
    )
    embedder_id: str | None = None
    create_space: bool = False

    limit: int = Field(default=5, gt=0)
    fetch_k: int | None = Field(
        default=None,
        gt=0,
        description="Candidates to retrieve before reranking. Defaults to limit.",
    )
    reranker_id: str | None = None
    min_score: float | None = Field(
        default=None,
        description="Only applied when reranker_id is set. See the class docstring.",
    )
    filter: str | None = Field(
        default=None,
        description="A GoodMem filter expression applied to every space searched.",
    )
    metadata_filter: dict[str, Any] | None = Field(
        default=None,
        description="Field/value pairs, safely quoted and AND-ed into the filter.",
    )
    llm_id: str | None = Field(
        default=None,
        description="Enables the server's abstract reply. Costs an LLM call per search.",
    )
    llm_temperature: float | None = None

    _conn: GoodMemConnection = PrivateAttr()
    _resolved_space_id: str | None = PrivateAttr(default=None)

    def __init__(self, **data: Any) -> None:
        conn = split_connection_kwargs(data)
        super().__init__(**data)
        self._conn = conn
        if not (self.space_id or self.space_ids or self.space_name):
            raise ValueError(
                "Provide space_id, space_ids or space_name so the retriever "
                "knows what to search."
            )

    # ------------------------------------------------------------- internals
    def _targets(self, client: Any) -> list[str]:
        ids = list(self.space_ids)
        if self.space_id:
            ids.insert(0, self.space_id)
        if self.space_name:
            if self._resolved_space_id is None:
                self._resolved_space_id = resolve(
                    client,
                    name=self.space_name,
                    embedder_id=self.embedder_id,
                    create=self.create_space,
                )
            ids.insert(0, self._resolved_space_id)
        return list(dict.fromkeys(ids))

    def _expression(self, extra: dict[str, Any] | None) -> str | None:
        return combine(
            self.filter,
            from_mapping(self.metadata_filter) if self.metadata_filter else None,
            from_mapping(extra) if extra else None,
        )

    # --------------------------------------------------------------- traced
    @observe(type="retriever", name="GoodMem Retriever")
    def search(
        self,
        query: str,
        *,
        limit: int | None = None,
        metadata_filter: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Retrieve for one query, as a DeepEval retriever span.

        Returns ``{"query", "hits", "score_kind", "statuses", "partial",
        "abstract_reply", "space_ids"}``.

        ``partial`` is True when the server reported a real problem during
        this retrieval, whether or not hits came back; ``statuses`` says what
        the problem was. A retrieval that reported a problem and returned
        nothing is an empty ``hits`` with ``partial=True`` and a warning -- it
        is never raised, and it is distinguishable from "no matches" by the
        flag. (Retrieval status contract, Q4a/Q4b.)
        """
        if not query or not query.strip():
            raise ValueError("query cannot be empty.")

        want = limit if limit is not None else self.limit
        reranked = bool(self.reranker_id)
        expression = self._expression(metadata_filter)

        update_retriever_span(embedder=self.embedder_id, top_k=want)

        with self._conn.session() as client:
            targets = self._targets(client)
            kwargs: dict[str, Any] = {
                "message": query,
                "requested_size": self.fetch_k or want,
                "fetch_memory": True,
                "stream": False,
            }
            if expression is None:
                kwargs["space_ids"] = targets
            else:
                kwargs["space_keys"] = [
                    {"spaceId": sid, "filter": expression} for sid in targets
                ]
            if reranked:
                kwargs["reranker_id"] = self.reranker_id
                kwargs["max_results"] = want
            if self.llm_id:
                kwargs["llm_id"] = self.llm_id
                if self.llm_temperature is not None:
                    kwargs["llm_temp"] = self.llm_temperature

            events = list(client.memories.retrieve(**kwargs))

        statuses, degraded = classify(events)
        hits = hits_from_events(events, reranked=reranked)

        if reranked and self.min_score is not None:
            hits = [
                h for h in hits if h["score"] is None or h["score"] >= self.min_score
            ]
        # Server order is authoritative and is not re-sorted here.
        hits = hits[:want]

        if degraded and not hits:
            summary = "; ".join(
                f"{s.get('code', 'UNKNOWN')}: {s.get('message', '')}" for s in statuses
            )
            warnings.warn(
                f"GoodMem retrieval returned nothing and reported a problem: {summary}",
                stacklevel=2,
            )
            logger.warning(
                "GoodMem retrieval failed with no results; statuses=%s", statuses
            )

        return {
            "query": query,
            "hits": hits,
            "score_kind": "reranker" if reranked else "vector",
            "statuses": statuses,
            "partial": degraded,
            "abstract_reply": abstract_reply(events),
            "space_ids": targets,
        }

    # ---------------------------------------------------------- convenience
    def retrieve(
        self,
        query: str,
        *,
        limit: int | None = None,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[str]:
        """Retrieve and return just the chunk text, for ``retrieval_context``.

        The shape DeepEval's RAG metrics consume. Use :meth:`search` when you
        also need the scores or the server's statuses.
        """
        result = self.search(query, limit=limit, metadata_filter=metadata_filter)
        return [h["chunk_text"] for h in result["hits"] if h.get("chunk_text")]

    def to_test_case(
        self,
        result: dict[str, Any],
        *,
        actual_output: str,
        expected_output: str | None = None,
        context: Sequence[str] | None = None,
        **fields: Any,
    ) -> LLMTestCase:
        """Build an ``LLMTestCase`` from a :meth:`search` result.

        ``retrieval_context`` is the retrieved chunk text, which is what
        ``ContextualPrecisionMetric``, ``ContextualRecallMetric``,
        ``ContextualRelevancyMetric`` and ``FaithfulnessMetric`` read.

        The retrieval's diagnostics travel in ``metadata["goodmem"]`` so a
        metric can see them -- see
        :class:`~deepeval_goodmem.metrics.GoodMemRetrievalHealthMetric`.
        Scoring an evaluation without knowing the retrieval was degraded is
        how a broken reranker turns into a mysterious drop in recall.
        """
        metadata = dict(fields.pop("metadata", None) or {})
        metadata["goodmem"] = {
            "partial": result.get("partial", False),
            "statuses": result.get("statuses", []),
            "score_kind": result.get("score_kind"),
            "scores": [h.get("score") for h in result.get("hits", [])],
            "chunk_ids": [h.get("chunk_id") for h in result.get("hits", [])],
            "memory_ids": [h.get("memory_id") for h in result.get("hits", [])],
            "space_ids": result.get("space_ids", []),
        }
        return LLMTestCase(
            input=result["query"],
            actual_output=actual_output,
            expected_output=expected_output,
            context=list(context) if context is not None else None,
            retrieval_context=[
                h["chunk_text"] for h in result.get("hits", []) if h.get("chunk_text")
            ],
            metadata=metadata,
            **fields,
        )


__all__ = ["GoodMemRetriever", "GoodMemSpaceError"]
