"""GoodMem Create Memory operation."""

from typing import Any

from deepeval_goodmem._base import GoodMemOperation


class GoodMemCreateMemory(GoodMemOperation):
    """Store a document as a new memory in a GoodMem space.

    The memory is processed asynchronously: chunked into searchable pieces
    and embedded into vectors. Accepts a local file path or plain text.
    """

    name: str = "goodmem_create_memory"
    description: str = (
        "Store a document as a new memory in a GoodMem space. "
        "Accepts a local file path or plain text. "
        "The memory is chunked and embedded asynchronously."
    )

    def _run(
        self,
        space_id: str,
        text_content: str | None = None,
        file_path: str | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Create a memory in the specified space.

        Args:
            space_id: The UUID of the space to store the memory in.
            text_content: Plain text content to store as memory. If both
                `file_path` and `text_content` are provided, the file takes
                priority.
            file_path: Local file path to upload as memory (PDF, DOCX, image,
                etc.). Content type is auto-detected from the file extension.
            metadata: Optional key-value metadata as a dictionary.
            **kwargs: Additional keyword arguments (unused).

        Returns:
            Dictionary with the operation result.
        """
        return self._client().create_memory(
            space_id=space_id,
            text_content=text_content,
            file_path=file_path,
            metadata=metadata,
        )
