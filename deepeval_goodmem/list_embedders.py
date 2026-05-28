"""GoodMem List Embedders operation."""

from typing import Any

from deepeval_goodmem._base import GoodMemOperation


class GoodMemListEmbedders(GoodMemOperation):
    """List all available GoodMem embedder models.

    Embedders convert text into vector representations used for similarity
    search. Pass the returned embedder ID when creating a new space.
    """

    name: str = "goodmem_list_embedders"
    description: str = (
        "List all available GoodMem embedder models. Use the returned "
        "embedder ID when creating a new space."
    )

    def _run(self, **kwargs: Any) -> dict[str, Any]:
        """List all embedders.

        Args:
            **kwargs: Additional keyword arguments (unused).

        Returns:
            Dictionary with the list of embedders.
        """
        embedders = self._client().list_embedders()
        return {
            "success": True,
            "embedders": embedders,
            "totalEmbedders": len(embedders),
        }
