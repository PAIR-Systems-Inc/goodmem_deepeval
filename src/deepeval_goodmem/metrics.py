"""A DeepEval metric for the health of the retrieval itself."""

from __future__ import annotations

from typing import Any

from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase


class GoodMemRetrievalHealthMetric(BaseMetric):
    """Fail a test case whose retrieval was degraded.

    Every other RAG metric scores the *content* that came back and cannot
    tell a thin corpus from a broken reranker: both look like poor recall.
    This one reads the diagnostics
    :meth:`~deepeval_goodmem.retriever.GoodMemRetriever.to_test_case` puts in
    ``metadata["goodmem"]`` and scores the retrieval itself, so a run that
    silently lost a stage shows up as its own failure rather than as a
    mysterious drop in someone else's number::

        evaluate(cases, [ContextualRecallMetric(),
                         GoodMemRetrievalHealthMetric()])

    ``1.0`` when the server reported nothing wrong, ``0.0`` when it did. No
    LLM is involved, so this costs nothing and never flakes. A test case
    carrying no GoodMem metadata is skipped rather than failed.
    """

    # BaseMetric declares threshold as optional; this metric always has one.
    threshold: float

    def __init__(self, threshold: float = 1.0, *, require_hits: bool = False) -> None:
        self.threshold = threshold
        self.require_hits = require_hits
        self.async_mode = False
        self.include_reason = True

    @property
    def __name__(self) -> str:
        return "GoodMem Retrieval Health"

    def measure(self, test_case: LLMTestCase, *args: Any, **kwargs: Any) -> float:
        diagnostics = (test_case.metadata or {}).get("goodmem")
        if diagnostics is None:
            # Not a GoodMem-sourced case; do not invent a verdict for it.
            self.skipped = True
            self.score = None
            self.success = None
            self.reason = "No GoodMem retrieval metadata on this test case."
            return 0.0

        statuses = diagnostics.get("statuses") or []
        partial = bool(diagnostics.get("partial"))
        empty = not (test_case.retrieval_context or [])

        problems = []
        if partial:
            codes = ", ".join(str(s.get("code", "UNKNOWN")) for s in statuses)
            problems.append(f"the server reported {codes}")
        if self.require_hits and empty:
            problems.append("no chunks were retrieved")

        self.score = 0.0 if problems else 1.0
        self.success = self.score >= self.threshold
        self.reason = (
            "Retrieval completed with no problems reported."
            if not problems
            else "Retrieval was degraded: " + "; and ".join(problems) + "."
        )
        self.score_breakdown = {
            "partial": partial,
            "status_codes": [str(s.get("code", "UNKNOWN")) for s in statuses],
            "num_chunks": len(test_case.retrieval_context or []),
            "score_kind": diagnostics.get("score_kind"),
        }
        return self.score

    async def a_measure(
        self, test_case: LLMTestCase, *args: Any, **kwargs: Any
    ) -> float:
        # Reading a dict needs no I/O, so there is nothing to await.
        return self.measure(test_case, *args, **kwargs)

    def is_successful(self) -> bool | None:
        return self.success


__all__ = ["GoodMemRetrievalHealthMetric"]
