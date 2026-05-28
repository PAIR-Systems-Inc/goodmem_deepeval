"""GoodMem Create Space operation."""

from typing import Any

from deepeval_goodmem._base import GoodMemOperation


class GoodMemCreateSpace(GoodMemOperation):
    """Create a new GoodMem space or reuse an existing one.

    A space is a logical container for related memories, configured with an
    embedder that converts text to vector embeddings. If a space with the
    given name already exists, its ID is returned instead of a duplicate.
    """

    name: str = "goodmem_create_space"
    description: str = (
        "Create a new GoodMem space or reuse an existing one. "
        "A space is a logical container for organizing related memories, "
        "configured with an embedder for vector search."
    )

    def _run(
        self,
        name: str,
        embedder_id: str,
        chunking_strategy: str = "recursive",
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Create a space or return an existing one.

        Args:
            name: A unique name for the space.
            embedder_id: The ID of the embedder model that converts text into
                vector representations for similarity search.
            chunking_strategy: The chunking strategy for text processing.
                One of `recursive`, `sentence`, or `none`.
            chunk_size: Maximum chunk size in characters (for recursive/sentence).
            chunk_overlap: Overlap between consecutive chunks in characters.
            **kwargs: Additional keyword arguments (unused).

        Returns:
            Dictionary with the operation result.
        """
        return self._client().create_space(
            name=name,
            embedder_id=embedder_id,
            chunking_strategy=chunking_strategy,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
