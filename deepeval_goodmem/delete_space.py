"""GoodMem Delete Space operation."""

from typing import Any

from deepeval_goodmem._base import GoodMemOperation


class GoodMemDeleteSpace(GoodMemOperation):
    """Permanently delete a GoodMem space.

    Removes the space and any data associated with it. This is irreversible.
    """

    name: str = "goodmem_delete_space"
    description: str = (
        "Permanently delete a GoodMem space and any data associated with it. "
        "This action cannot be undone."
    )

    def _run(self, space_id: str, **kwargs: Any) -> dict[str, Any]:
        """Delete a space by ID.

        Args:
            space_id: The space UUID.
            **kwargs: Additional keyword arguments (unused).

        Returns:
            Dictionary with the deletion result.
        """
        return self._client().delete_space(space_id=space_id)
