"""DeepEval plugin for GoodMem, the retrieval-augmented generation (RAG) memory backend for AI agents."""

from deepeval_goodmem._base import GoodMemOperation
from deepeval_goodmem._client import GoodMemClient
from deepeval_goodmem.create_memory import GoodMemCreateMemory
from deepeval_goodmem.create_space import GoodMemCreateSpace
from deepeval_goodmem.delete_memory import GoodMemDeleteMemory
from deepeval_goodmem.delete_space import GoodMemDeleteSpace
from deepeval_goodmem.get_memory import GoodMemGetMemory
from deepeval_goodmem.get_space import GoodMemGetSpace
from deepeval_goodmem.list_embedders import GoodMemListEmbedders
from deepeval_goodmem.list_memories import GoodMemListMemories
from deepeval_goodmem.list_spaces import GoodMemListSpaces
from deepeval_goodmem.retrieve_memories import GoodMemRetrieveMemories
from deepeval_goodmem.update_space import GoodMemUpdateSpace

__all__ = [
    "GoodMemClient",
    "GoodMemCreateMemory",
    "GoodMemCreateSpace",
    "GoodMemDeleteMemory",
    "GoodMemDeleteSpace",
    "GoodMemGetMemory",
    "GoodMemGetSpace",
    "GoodMemListEmbedders",
    "GoodMemListMemories",
    "GoodMemListSpaces",
    "GoodMemOperation",
    "GoodMemRetrieveMemories",
    "GoodMemUpdateSpace",
]
