"""GoodMem Delete Memory operation."""

from typing import Any

from deepeval_goodmem._base import GoodMemOperation


class GoodMemDeleteMemory(GoodMemOperation):
    """Permanently delete a GoodMem memory and its associated data.

    Removes the memory record, its chunks, and vector embeddings. This
    action cannot be undone.
    """

    name: str = "goodmem_delete_memory"
    description: str = (
        "Permanently delete a GoodMem memory and its associated chunks "
        "and vector embeddings."
    )

    def _run(self, memory_id: str, **kwargs: Any) -> dict[str, Any]:
        """Delete a memory by ID.

        Args:
            memory_id: The UUID of the memory to delete.
            **kwargs: Additional keyword arguments (unused).

        Returns:
            Dictionary with the deletion result.
        """
        return self._client().delete_memory(memory_id=memory_id)
