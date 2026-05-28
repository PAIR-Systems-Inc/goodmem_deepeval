"""GoodMem Get Space operation."""

from typing import Any

from deepeval_goodmem._base import GoodMemOperation


class GoodMemGetSpace(GoodMemOperation):
    """Fetch a specific GoodMem space by its ID.

    Returns the full space record including embedder configuration,
    chunking config, labels, and access settings.
    """

    name: str = "goodmem_get_space"
    description: str = (
        "Fetch a specific GoodMem space by its ID, including its name, "
        "labels, embedder configuration, and chunking settings."
    )

    def _run(self, space_id: str, **kwargs: Any) -> dict[str, Any]:
        """Fetch a space by ID.

        Args:
            space_id: The space UUID.
            **kwargs: Additional keyword arguments (unused).

        Returns:
            Dictionary with the space data.
        """
        space = self._client().get_space(space_id=space_id)
        return {"success": True, "space": space}
