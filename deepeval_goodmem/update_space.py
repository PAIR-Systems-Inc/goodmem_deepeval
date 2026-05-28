"""GoodMem Update Space operation."""

from typing import Any

from deepeval_goodmem._base import GoodMemOperation


class GoodMemUpdateSpace(GoodMemOperation):
    """Update mutable fields on a GoodMem space.

    Only ``name``, ``publicRead``, and labels are mutable. Embedders and
    chunking config are immutable after creation.
    """

    name: str = "goodmem_update_space"
    description: str = (
        "Update mutable fields on a GoodMem space (name, publicRead, labels). "
        "Embedders and chunking config cannot be changed after creation."
    )

    def _run(
        self,
        space_id: str,
        name: str | None = None,
        public_read: bool | None = None,
        replace_labels: dict[str, str] | None = None,
        merge_labels: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Update a space.

        Args:
            space_id: The space UUID.
            name: New space name.
            public_read: New publicRead value.
            replace_labels: Replacement label set.
            merge_labels: Labels to merge into the existing set.
            **kwargs: Additional keyword arguments (unused).

        Returns:
            Dictionary with the updated space.
        """
        space = self._client().update_space(
            space_id=space_id,
            name=name,
            public_read=public_read,
            replace_labels=replace_labels,
            merge_labels=merge_labels,
        )
        return {
            "success": True,
            "spaceId": space.get("spaceId", space_id),
            "space": space,
            "message": "Space updated successfully",
        }
