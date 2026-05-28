"""GoodMem List Memories operation."""

from typing import Any

from deepeval_goodmem._base import GoodMemOperation


class GoodMemListMemories(GoodMemOperation):
    """List memories within a GoodMem space.

    Supports pagination via ``next_token`` and filtering by processing status
    or metadata expression.
    """

    name: str = "goodmem_list_memories"
    description: str = (
        "List memories in a GoodMem space, with optional pagination, status "
        "filtering, and metadata filter expressions."
    )

    def _run(
        self,
        space_id: str,
        max_results: int | None = None,
        next_token: str | None = None,
        status_filter: str | None = None,
        include_content: bool = False,
        filter_expression: str | None = None,
        sort_by: str | None = None,
        sort_order: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """List memories in a space.

        Args:
            space_id: The space UUID.
            max_results: Page size.
            next_token: Pagination token.
            status_filter: Processing status filter.
            include_content: Whether to include original content.
            filter_expression: Metadata filter expression.
            sort_by: Field to sort by. One of ``created_at`` or ``updated_at``.
            sort_order: Sort direction. One of ``ASCENDING`` or ``DESCENDING``.
            **kwargs: Additional keyword arguments (unused).

        Returns:
            Dictionary with the memories list and ``nextToken``.
        """
        return self._client().list_memories(
            space_id=space_id,
            max_results=max_results,
            next_token=next_token,
            status_filter=status_filter,
            include_content=include_content,
            filter_expression=filter_expression,
            sort_by=sort_by,
            sort_order=sort_order,
        )
