"""GoodMem List Spaces operation."""

from typing import Any

from deepeval_goodmem._base import GoodMemOperation


class GoodMemListSpaces(GoodMemOperation):
    """List all GoodMem spaces in the account.

    Returns each space with its ID, name, labels, embedder configuration,
    and access settings.
    """

    name: str = "goodmem_list_spaces"
    description: str = (
        "List all GoodMem spaces. Returns each space with its ID, name, "
        "embedder configuration, and access settings."
    )

    def _run(self, **kwargs: Any) -> dict[str, Any]:
        """List all spaces.

        Args:
            **kwargs: Additional keyword arguments (unused).

        Returns:
            Dictionary with the list of spaces.
        """
        spaces = self._client().list_spaces()
        return {
            "success": True,
            "spaces": spaces,
            "totalSpaces": len(spaces),
        }
