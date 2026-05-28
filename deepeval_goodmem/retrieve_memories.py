"""GoodMem Retrieve Memories operation."""

from typing import Any

from deepeval_goodmem._base import GoodMemOperation


class GoodMemRetrieveMemories(GoodMemOperation):
    """Run similarity-based semantic retrieval across GoodMem spaces.

    Returns matching chunks ranked by relevance, with optional full memory
    definitions. When `wait_for_indexing` is enabled, the operation polls
    until results land or `max_wait_seconds` elapses.
    """

    name: str = "goodmem_retrieve_memories"
    description: str = (
        "Perform similarity-based semantic retrieval across one or more "
        "GoodMem spaces. Returns matching chunks ranked by relevance."
    )

    def _run(
        self,
        query: str,
        space_ids: str,
        max_results: int = 5,
        include_memory_definition: bool = True,
        wait_for_indexing: bool = True,
        max_wait_seconds: float = 60.0,
        poll_interval: float = 5.0,
        reranker_id: str | None = None,
        llm_id: str | None = None,
        relevance_threshold: float | None = None,
        llm_temperature: float | None = None,
        chronological_resort: bool | None = None,
        metadata_filter: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Retrieve semantically similar memories.

        Args:
            query: A natural language query used to find semantically similar
                memory chunks.
            space_ids: One or more space UUIDs to search across, separated by
                commas (e.g., 'id1,id2').
            max_results: Maximum number of matching chunks to return.
            include_memory_definition: Fetch the full memory metadata
                (source document info, processing status) alongside the
                matched chunks.
            wait_for_indexing: Poll for results when none come back on the
                first call. Enable this when memories were just added and may
                still be undergoing chunking and embedding.
            max_wait_seconds: Maximum time in seconds to poll for results when
                `wait_for_indexing` is enabled.
            poll_interval: Seconds to sleep between polling attempts when
                `wait_for_indexing` is enabled.
            reranker_id: UUID of a reranker model to refine the order of
                retrieved chunks via direct query-chunk scoring.
            llm_id: UUID of an LLM that will produce a contextual summary
                (`abstractReply`) over the retrieved chunks.
            relevance_threshold: Minimum relevance score (0-1) below which
                results are dropped. Only applied when a post-processor is
                configured.
            llm_temperature: Creativity setting for LLM generation (0-2).
                Only used when `llm_id` is also provided.
            chronological_resort: Reorder final results by creation time after
                reranking and thresholding.
            metadata_filter: SQL-style JSONPath expression applied server-side
                to narrow results by metadata. The same filter is attached to
                every space key in the request. Empty strings are ignored.
                Example: `CAST(val('$.category') AS TEXT) = 'feat'`.
            **kwargs: Additional keyword arguments (unused).

        Returns:
            Dictionary with the retrieval results.
        """
        return self._client().retrieve_memories(
            query=query,
            space_ids=space_ids,
            max_results=max_results,
            include_memory_definition=include_memory_definition,
            wait_for_indexing=wait_for_indexing,
            max_wait_seconds=max_wait_seconds,
            poll_interval=poll_interval,
            reranker_id=reranker_id,
            llm_id=llm_id,
            relevance_threshold=relevance_threshold,
            llm_temperature=llm_temperature,
            chronological_resort=chronological_resort,
            metadata_filter=metadata_filter,
        )
