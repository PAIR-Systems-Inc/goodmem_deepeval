"""GoodMem Get Memory operation."""

from typing import Any

from deepeval_goodmem._base import GoodMemOperation


class GoodMemGetMemory(GoodMemOperation):
    """Fetch a specific GoodMem memory by its ID.

    Returns the memory metadata, processing status, and optionally the
    original document content.
    """

    name: str = "goodmem_get_memory"
    description: str = (
        "Fetch a specific GoodMem memory by its ID, including metadata, "
        "processing status, and optionally the original content."
    )

    def _run(
        self,
        memory_id: str,
        include_content: bool = True,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Fetch a memory by ID.

        Args:
            memory_id: The UUID of the memory to fetch.
            include_content: Fetch the original document content of the memory
                in addition to its metadata.
            **kwargs: Additional keyword arguments (unused).

        Returns:
            Dictionary with the memory data.
        """
        return self._client().get_memory(
            memory_id=memory_id,
            include_content=include_content,
        )
